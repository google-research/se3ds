# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Basic GAN manager class."""

import abc
import os
import random
import time
from typing import Optional

from absl import logging
import gin
import numpy as np
from se3ds import constants
from se3ds.utils import ema
from se3ds.utils import eval_metric
from se3ds.utils import image_grid
from se3ds.utils import logger
from se3ds.utils import pano_utils
from se3ds.utils import parameter_overview
from se3ds.utils import task_manager
import tensorflow as tf

_HOURS_IN_DAY = 24
_SECS_IN_HOUR = 3600


@gin.configurable(denylist=["strategy", "model_dir"])
class GANManager(abc.ABC):
  """Basic Class for GAN for managing training and evaluation.

  Attributes:
    strategy: A tf.distribute.Strategy instance of distributed processing.
    model_dir: A string for saving the checkpoint and log.
    image_size: The size of the generated images.
    seed: The random seed used for the training/eval.
    optimizer_type: A string indicates the type of the optimizer.
    beta1: Adam optimizer hyperparameter beta1.
    beta2: Adam optimizer huyperparmeter beta2.
    g_lr: The learning rate of the generator.
    d_lr: The learning rate of the discriminator.
    global_batch_size: The global batch size for all the replicas.
      Set separately for different train/test/test_text mode.
    parallel_calls: The number of parallel processing calls for data pipeline.
    log_every_steps: The number of steps to log the result for tensorboard.
    save_every_steps: The number of steps to save the model.
    eval_every_steps: The number of steps for evaluation.
    num_epochs: The number of training epochs.
    d_step_per_g_step: The number of training updates for the discriminator
      for each generator update.
    num_batched_steps: Number of training steps to be batched into a single
      tf.function call. May speed up training.
    show_num: The number of images for the generated image grid.
    shuffle_buffer_size: the number of data in the shuffled buffer.
    generator_fn: Function (or constructor) to return the generator
    discriminator_fn: Function (or constructor) to return the discriminator.
    train_dataset_glob: Glob string indicates for the tfrecords for train set.
    test_dataset_glob: Glob string indictes for the tfrecords for test set.
    eval_size: Total number of eval images.
    generator: The Generator model.
    discriminator: The discriminator model.
    ema_generator: The EMA Generator model.
    g_optimizer: The generator optimizer.
    d_optimizer: The discriminagtor optimizer.
    train_ds: The training data.
    eval_ds: The evaluation data.
    train_batch_size: Batch size used for training  (TPU/GPU).
    test_batch_size: Batch size used for evaluation (TPU/GPU).
    train_batch_size_all_steps: Total Batch size for all g_d_steps.
    train_num: Total number of training data.
    eval_num: Total number of the test data.
    train_steps_per_epoch: Number of training steps per epoch.
    test_steps_per_epoch: Number of test steps per epoch.
    global_step: Current training step.
    ema_decay: Decay rate for ema variables.
    ema_init_step: The start step for ema calculation.
    ckpt: Checkpoints for saving.
    ckpt_manager: Auxiliary object for saving checkpoint.
    metrics: A dictionary for saving training/evaluation metrics.
    test_split: Validation split to run on. Can be val_seen or val_unseen.
    eval_seq_len: How many frames to write outputs for.
    predict_depth: If true, the model predicts depth alongside RGB.
  """

  def __init__(
      self,
      strategy: tf.distribute.Strategy,
      model_dir: str,
      image_size: int = 128,
      seed: int = 1,
      optimizer_type: str = "adam",
      beta1: float = 0.0,
      beta2: float = 0.999,
      g_lr: float = 0.0002,
      d_lr: float = 0.0002,
      train_batch_size: int = 128,
      test_batch_size: int = 128,
      parallel_calls: int = tf.data.experimental.AUTOTUNE,
      log_every_steps: int = 1000,
      save_every_steps: int = 2000,
      eval_every_steps: int = 2000,
      num_epochs: int = 100,
      d_step_per_g_step: int = 1,
      num_batched_steps: int = 5,
      show_num: int = 16,
      shuffle_buffer_size: int = 1000,
      ema_decay: float = 0.999,
      ema_init_step: int = 0,
      generator_fn=None,
      discriminator_fn=None,
      train_dataset_glob: Optional[str] = None,
      test_dataset_glob: Optional[str] = None,
      eval_size: Optional[int] = 10000,
      test_split: str = "val_seen",
      eval_seq_len: int = 4,
      predict_depth: bool = False,
  ):
    """Initializes the gan manager."""

    self.strategy = strategy
    self.model_dir = model_dir
    self.image_size = image_size
    self.seed = seed
    self.optimizer_type = optimizer_type
    self.beta1 = beta1
    self.beta2 = beta2
    self.g_lr = g_lr
    self.d_lr = d_lr
    self.global_batch_size = train_batch_size
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.parallel_calls = parallel_calls
    self.log_every_steps = log_every_steps
    self.save_every_steps = save_every_steps
    self.eval_every_steps = eval_every_steps
    self.num_epochs = num_epochs
    self.d_step_per_g_step = d_step_per_g_step
    self.num_batched_steps = num_batched_steps
    self.show_num = show_num
    self.shuffle_buffer_size = shuffle_buffer_size
    self.ema_decay = ema_decay
    self.ema_init_step = ema_init_step
    self.generator_fn = generator_fn
    self.discriminator_fn = discriminator_fn
    self.train_dataset_glob = train_dataset_glob
    self.test_dataset_glob = test_dataset_glob
    self.eval_size = eval_size
    self.test_split = test_split
    self.eval_seq_len = eval_seq_len
    self.predict_depth = predict_depth
    if seed > 0:
      random.seed(seed)
      np.random.seed(seed)
      tf.random.set_seed(seed)

  def _build_model(self):
    """Creates Generator, Discriminator and EMA Generator."""
    self.generator = self.generator_fn(image_size=self.image_size)
    self.discriminator = self.discriminator_fn(image_size=self.image_size)
    self.ema_generator = self.generator_fn(image_size=self.image_size)

  def _build_optimizer(self):
    """Creates optimizers for both Generator and Discriminator."""
    if self.optimizer_type == "adam":
      self.g_optimizer = tf.keras.optimizers.Adam(
          lr=self.g_lr, beta_1=self.beta1, beta_2=self.beta2)
      self.d_optimizer = tf.keras.optimizers.Adam(
          lr=self.d_lr, beta_1=self.beta1, beta_2=self.beta2)
    else:
      raise NotImplementedError

  def _get_dataset(self):
    """Gets the dataset for the model."""
    raise NotImplementedError

  def _create_data(self):
    """Creates Data for training and eval."""

    logging.info("Creating Data...")
    train_batch_size = self.global_batch_size * self.d_step_per_g_step
    test_batch_size = self.global_batch_size
    train_ds, test_ds = self._get_dataset()

    self.train_ds = train_ds.input_fn(
        split="train",
        global_batch_size=train_batch_size,
        strategy=self.strategy,
        shuffle=True,
        shuffle_buffer_size=self.shuffle_buffer_size,
        file_pattern=self.train_dataset_glob,
        seed=self.seed,
        parallel_calls=self.parallel_calls)
    self.eval_ds = test_ds.input_fn(
        split=self.test_split,
        global_batch_size=test_batch_size,
        strategy=self.strategy,
        shuffle=False,
        file_pattern=self.test_dataset_glob,
        seed=self.seed,
        parallel_calls=self.parallel_calls)
    display_ds = test_ds.input_fn(
        split=self.test_split,
        global_batch_size=test_batch_size,
        strategy=self.strategy,
        shuffle=False,
        file_pattern=self.test_dataset_glob,
        seed=self.seed,
        parallel_calls=1)

    self.train_ds = iter(self.train_ds)
    self.eval_ds = iter(self.eval_ds)
    self.display_batch = next(iter(display_ds))
    self.train_batch_size_all_steps = train_batch_size
    self.train_num = train_ds.num_examples["train"]
    self.eval_num = test_ds.num_examples[self.test_split]
    self.train_steps_per_epoch = self.train_num // train_batch_size
    self.test_steps_per_epoch = self.eval_num // test_batch_size
    logging.info("train_num %s, eval_num %s", self.train_num, self.eval_num)

  def test(self, unit_test: bool = False):
    """Test function for the pipeline."""

    logging.info("Start Testing...")
    self.global_batch_size = self.test_batch_size
    self._create_data()
    task_manager_csv = task_manager.TaskManagerWithCsvResults(
        self.model_dir, score_file=f"scores_{self.test_split}.csv")
    test_logger = logger.UniversalLogger(self.model_dir, step=0)
    eval_size = self.eval_size or self.eval_num
    metric = eval_metric.EvalMetric(
        ds=self.eval_ds,
        eval_num=eval_size,
        batch_size=self.test_batch_size,
        strategy=self.strategy,
        avg_num=1)

    with self.strategy.scope():
      self._create_obj()

    checkpoints = task_manager_csv.unevaluated_checkpoints(
        timeout=_HOURS_IN_DAY * _SECS_IN_HOUR,
        num_batched_steps=self.num_batched_steps,
        eval_every_steps=self.eval_every_steps)
    # Create a dummy checkpoint for unit testing. Model is randomly initialized.
    if unit_test:
      checkpoints = ["test-1"]
    for checkpoint_path in checkpoints:
      if not unit_test:
        try:
          with self.strategy.scope():
            self.ckpt.restore(checkpoint_path)
        except FileNotFoundError:
          logging.info("Could not find %s", checkpoint_path)
          continue
      step = self.global_step.numpy()
      image_dict, output_dict = self._get_image_grid("eval")
      image_dict_updated = {
          f"{k}/{self.test_split}": v for k, v in image_dict.items()
      }

      # Save image RGB outputs.
      image_output_dir = os.path.join(self.model_dir,
                                      f"images/{self.test_split}/{step}")
      outputs_to_save = {
          "rgb": "ema_generated_image",
          "depth": "ema_pred_depth",
      }

      for suffix, k in outputs_to_save.items():
        image_outputs = output_dict[k]  # (N, H, W, 3)
        num_examples = image_outputs.shape[0] // self.eval_seq_len
        for example_idx in range(num_examples):
          for frame_idx in range(self.eval_seq_len):
            frame_dir = os.path.join(image_output_dir, str(frame_idx))
            tf.io.gfile.makedirs(frame_dir)
            actual_idx = example_idx * self.eval_seq_len + frame_idx
            frame_image = tf.image.convert_image_dtype(
                image_outputs[actual_idx, ...], tf.uint8)
            with tf.io.gfile.GFile(
                os.path.join(frame_dir, f"{example_idx}_{suffix}.png"),
                "wb") as wf:
              image_data = tf.io.encode_png(frame_image).numpy()
              wf.write(image_data)

      time_start = time.time()
      fid, _, rmse = metric.calculate_fid_score(self.generator)
      ema_fid, _, ema_rmse = metric.calculate_fid_score(self.ema_generator)

      time_end = time.time()
      duration = (time_end - time_start) / 60.0
      logging.info("Step %d, Eval Time %.2f minutes", step, duration)
      result_dict = {}
      for i in fid:
        curr_result_dict = dict(
            fid=fid[i],
            ema_fid=ema_fid[i],
            rmse=rmse[i],
            ema_rmse=ema_rmse[i],
        )
        curr_result_dict = {f"{k}@{i}": v for k, v in curr_result_dict.items()}
        result_dict.update(curr_result_dict)
      # Prepend eval/ to all reported stats.
      result_dict_updated = {
          f"{self.test_split}/eval_image/{k}": v
          for k, v in result_dict.items()
      }
      test_logger.log_scalars(step, **result_dict_updated)
      test_logger.log_images(step, **image_dict_updated)
      task_manager_csv.add_eval_result(checkpoint_path, result_dict_updated, -1)

  def _create_or_load_checkpoint(self):
    """Creates and maybe loads checkpoint."""

    self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
    if self.ckpt_manager.latest_checkpoint:
      logging.info("Restored from %s", self.ckpt_manager.latest_checkpoint)
    else:
      logging.info("Initializing from scratch.")

  def _create_obj(self):
    """Creates necessary model and optimizer for train and eval."""
    self.global_step = tf.Variable(
        0, trainable=False, name="global_step", dtype=tf.int64)
    self._build_model()
    self._build_optimizer()
    self._create_metrics()
    self.ckpt = tf.train.Checkpoint(
        g_optimizer=self.g_optimizer,
        d_optimizer=self.d_optimizer,
        generator=self.generator,
        discriminator=self.discriminator,
        ema_generator=self.ema_generator,
        global_step=self.global_step)

    self.ckpt_manager = tf.train.CheckpointManager(
        checkpoint=self.ckpt, directory=self.model_dir, max_to_keep=200)

  def _split_input_dict(self, input_dict, splits):
    """Splits input dicts for multiple dicts for each step of training."""

    output = []
    split_dict = {}
    for key, item in input_dict.items():
      item_list = tf.split(item, splits)
      split_dict[key] = item_list
    for i in range(splits):
      current_dict = {}
      for key in input_dict.keys():
        current_dict[key] = split_dict[key][i]
      output.append(current_dict)
    return output

  @abc.abstractmethod
  def train_d(self, inputs):
    """Learns the Discriminator updates."""
    pass

  @abc.abstractmethod
  def train_g_d(self, inputs):
    """Learns both the Generator and Discriminator updates."""
    pass

  @tf.function
  def train_cluster(self, steps=1):
    """Groups Generator and Discriminator updates into tf.function."""
    def step_fn(inputs):
      input_list = self._split_input_dict(inputs, self.d_step_per_g_step)
      for i in range(self.d_step_per_g_step - 1):
        self.train_d(input_list[i])
      self.train_g_d(input_list[-1])
    for _ in tf.range(steps):
      self.strategy.run(step_fn, args=(next(self.train_ds),))

  def train(self):
    """Training function for the pipeline."""

    logging.info("Create Data and Model...")
    self.global_batch_size = self.train_batch_size
    self._create_data()
    if self.num_epochs != -1:
      num_train_steps = self.num_epochs * self.train_steps_per_epoch
    else:
      num_train_steps = 1
    logging.info(
        "Total training steps %s, %s steps per epoch",
        num_train_steps, self.train_steps_per_epoch)
    with self.strategy.scope():
      self._create_obj()
      self._create_or_load_checkpoint()
    start_step = self.global_step.numpy()
    train_logger = logger.UniversalLogger(
        self.model_dir, step=start_step, num_train_steps=num_train_steps)
    task_manager_csv = task_manager.TaskManagerWithCsvResults(self.model_dir)
    # self.train_one_step_test()
    logging.info("Start Training...")
    for step in range(start_step, num_train_steps, self.num_batched_steps):
      self.train_cluster(self.num_batched_steps)
      if step % self.log_every_steps < self.num_batched_steps:
        result_dict = self._save_metrics_to_dict()
        train_logger.log_scalars(step, **result_dict)
        self._reset_metrics()
      if step > self.num_batched_steps and (
          step % self.save_every_steps < self.num_batched_steps):
        self.ckpt_manager.save(step)
        image_dict, _ = self._get_image_grid("train")

        train_logger.log_images(step, **image_dict)
      self.global_step.assign_add(self.num_batched_steps)
    self.ckpt_manager.save(num_train_steps)
    task_manager_csv.mark_training_done()

  @tf.function
  def _get_image_grid(self, name_prefix="train"):
    """Generates image grid for visualization."""

    def step_fn(inputs):
      all_generated = []
      all_ema_generated = []
      all_pred_depth = []
      all_ema_pred_depth = []
      all_projected = []
      all_images = []
      all_depths = []
      all_blurred_mask = []
      all_proj_mask = []
      if name_prefix == "train":
        inputs = self._split_input_dict(inputs, self.d_step_per_g_step)
        inputs = inputs[0]
        # Output without blurred edges.
        # inputs["blurred_mask"] = tf.zeros_like(inputs["blurred_mask"])
        _, _, _, depth_out, _, _, generated = self.generator(
            inputs=[inputs, None], training=False)
        _, _, _, ema_depth_out, _, _, ema_generated = self.ema_generator(
            inputs=[inputs, None], training=False)
        all_images.append(inputs["image"])
        all_depths.append(inputs["depth"])
        all_generated.append(generated)
        all_ema_generated.append(ema_generated)
        all_pred_depth.append(depth_out)
        all_ema_pred_depth.append(ema_depth_out)
        all_projected.append(inputs["proj_image"])
        all_blurred_mask.append(inputs["blurred_mask"])
        all_proj_mask.append(inputs["proj_mask"])
      else:
        for mode in ["normal", "ema"]:
          # All depth_scale within a batch should be the same.
          depth_scale = inputs["depth_scale"][0]
          batch_size, _, height, width, _ = inputs["image"].shape
          memory_coords = tf.zeros((batch_size, 4, 0))
          memory_feats = tf.zeros((batch_size, 0, 3), dtype=tf.int32)
          prev_rgb_tensor = None
          for frame_idx in range(self.eval_seq_len):
            rgb_tensor = inputs["image"][:, frame_idx, ...]  # (N, H, W, 3)
            if mode == "normal":
              all_images.append(rgb_tensor)
              all_depths.append(inputs["depth"][:, frame_idx, ...])

            if not self.predict_depth or frame_idx == 0:
              depth_tensor = inputs["depth"][:, frame_idx, ...]  # (N, H, W, 1)

            relative_position = inputs["position"][:, frame_idx, ...]
            relative_coords = memory_coords - relative_position[..., None]
            pred_depth, pred_rgb = (
                pano_utils.project_feats_to_equirectangular(
                    memory_feats,
                    relative_coords,
                    height,
                    width,
                    void_class=constants.INVALID_RGB_VALUE,
                    depth_scale=depth_scale))
            pred_mask = tf.cast(
                tf.math.logical_and(
                    tf.math.logical_and(pred_depth > 0, pred_depth < 1),
                    tf.math.reduce_all(
                        pred_rgb != constants.INVALID_RGB_VALUE,
                        axis=-1),
                ), tf.float32)[..., None]
            pred_depth = pred_depth[..., None]
            pred_rgb = tf.clip_by_value(
                tf.cast(pred_rgb / 255, tf.float32), 0, 1)
            blurred_mask = tf.zeros_like(pred_depth)

            if prev_rgb_tensor is None:
              prev_rgb_tensor = tf.zeros_like(rgb_tensor)
            first_frame = tf.ones((batch_size,))
            if frame_idx > 0:
              first_frame = tf.zeros((batch_size,))
            generator_inputs = {
                "prev_image": prev_rgb_tensor,
                "proj_image": pred_rgb,
                "proj_mask": pred_mask,
                "proj_depth": pred_depth,
                "blurred_mask": blurred_mask,
                "first_frame": first_frame,
                # TODO: Update according to actual ID for eval dataset.
                # Generate for MP3D.
                "dataset_type": inputs["dataset_type"]
            }

            if mode == "ema":
              _, _, _, depth_out, _, _, generated = self.ema_generator(
                  inputs=[generator_inputs, None], training=False)
              if self.predict_depth and frame_idx > 0:
                depth_tensor = depth_out

              all_ema_pred_depth.append(depth_tensor)
              all_ema_generated.append(generated)
            else:
              _, _, _, depth_out, _, _, generated = self.generator(
                  inputs=[generator_inputs, None], training=False)
              if self.predict_depth and frame_idx > 0:
                depth_tensor = depth_out

              all_pred_depth.append(depth_tensor)
              all_generated.append(generated)

            if frame_idx == 0:
              prev_rgb_tensor = rgb_tensor
              # For groundtruth, we mask out the blurred regions.
              rgb_tensor = pano_utils.mask_pano(
                  rgb_tensor,
                  masked_region_value=constants.INVALID_RGB_VALUE)
            else:  # Feed generated points back to the generator.
              rgb_tensor = generated
              prev_rgb_tensor = rgb_tensor

            # Add points to point cloud memory.
            pc_rgb_tensor = tf.cast(rgb_tensor * 255, tf.int32)
            pc_rgb_tensor = tf.clip_by_value(
                pc_rgb_tensor, constants.INVALID_RGB_VALUE, 255)
            xyz1, feats = pano_utils.equirectangular_to_pointcloud(
                pc_rgb_tensor,
                depth_tensor[..., 0],
                void_class=0,
                depth_scale=depth_scale)
            xyz1 += relative_position[..., None]
            memory_coords = tf.concat([memory_coords, xyz1], axis=2)
            memory_feats = tf.concat([memory_feats, feats], axis=1)

            if mode == "normal":
              all_projected.append(pred_rgb)
              all_blurred_mask.append(blurred_mask)
              all_proj_mask.append(pred_mask)

      generated = tf.concat(all_generated, axis=0)
      ema_generated = tf.concat(all_ema_generated, axis=0)
      pred_depth = tf.tile(tf.concat(all_pred_depth, axis=0), [1, 1, 1, 3])
      ema_pred_depth = tf.tile(
          tf.concat(all_ema_pred_depth, axis=0), [1, 1, 1, 3])
      image = tf.concat(all_images, axis=0)
      depth = tf.tile(tf.concat(all_depths, axis=0), [1, 1, 1, 3])
      projected = tf.concat(all_projected, axis=0)
      blurred_mask = tf.tile(tf.concat(all_blurred_mask, axis=0), [1, 1, 1, 3])
      proj_mask = tf.tile(tf.concat(all_proj_mask, axis=0), [1, 1, 1, 3])
      return (generated, ema_generated, pred_depth, ema_pred_depth, image,
              depth, projected, blurred_mask, proj_mask)

    if name_prefix == "train":
      inputs = next(self.train_ds)
    else:
      inputs = self.display_batch

    (generated, ema_generated, pred_depth, ema_pred_depth, image, depth,
     projected, blurred_mask, proj_mask) = self.strategy.run(
         step_fn, args=(inputs,))
    image_dict_generated = image_grid.get_grid_image_dict(
        generated, self.show_num, self.strategy, name_prefix + "_raw_generated")
    image_dict_ema_generated = image_grid.get_grid_image_dict(
        ema_generated, self.show_num,
        self.strategy, name_prefix + "_ema_generated")
    depth_dict_generated = image_grid.get_grid_image_dict(
        pred_depth, self.show_num, self.strategy, name_prefix + "_pred_depth")
    depth_dict_ema_generated = image_grid.get_grid_image_dict(
        ema_pred_depth, self.show_num, self.strategy,
        name_prefix + "_ema_pred_depth")
    image_dict_real = image_grid.get_grid_image_dict(
        image, self.show_num, self.strategy, name_prefix + "_real_img")
    depth_dict_real = image_grid.get_grid_image_dict(
        depth, self.show_num, self.strategy, name_prefix + "_real_depth")
    image_dict_projected = image_grid.get_grid_image_dict(
        projected, self.show_num, self.strategy, name_prefix + "_projected")
    image_dict_blurred_mask = image_grid.get_grid_image_dict(
        blurred_mask, self.show_num, self.strategy, name_prefix + "_blur_bbox")
    image_dict_proj_mask = image_grid.get_grid_image_dict(
        proj_mask, self.show_num, self.strategy, name_prefix + "_proj_mask")
    image_dict = {
        **image_dict_generated,
        **image_dict_ema_generated,
        **depth_dict_generated,
        **depth_dict_ema_generated,
        **image_dict_real,
        **depth_dict_real,
        **image_dict_projected,
        **image_dict_blurred_mask,
        **image_dict_proj_mask,
    }

    output_dict = {
        "ema_generated_image": tf.concat(
            self.strategy.experimental_local_results(ema_generated), axis=0),
        "ema_pred_depth": tf.concat(
            self.strategy.experimental_local_results(ema_pred_depth), axis=0),
    }
    return image_dict, output_dict

  def _create_metrics(self):
    """Creates metrics for training and evaluation."""

    self.metrics = {}
    self.metrics["gen_loss"] = tf.keras.metrics.Mean(name="gen_loss")
    self.metrics["disc_loss"] = tf.keras.metrics.Mean(name="disc_loss")

  def _reset_metrics(self):
    """Resets metrics after each logging period."""

    for key in self.metrics:
      self.metrics[key].reset_states()

  def _save_metrics_to_dict(self):
    """Saves metrics to dictionary for logging purpose."""

    output_dict = {}
    for key, value in self.metrics.items():
      if np.any(np.isnan(value.result().numpy())):
        raise ValueError(f"NaN losses recorded for {key}.")
      output_dict[key] = value.result().numpy()
    return output_dict

  def update_ema_model(self):
    if self.global_step >= self.ema_init_step:
      if self.global_step >= self.ema_init_step + self.num_batched_steps:
        ema.update_ema_variables(
            self.ema_generator.variables,
            self.generator.variables,
            self.ema_decay)
      else:
        self.assign_ema_model_first_time()

  def assign_ema_model_first_time(self):
    ema.assign_ema_vars_from_initial_values(
        self.ema_generator.variables,
        self.generator.variables)

  def print_summary(self):
    """Prints summary for the model."""

    logging.info(self.generator.summary())
    logging.info(self.discriminator.summary())
    logging.info("**************************")
    logging.info("Generator Variables")
    parameter_overview.log_parameter_overview(self.generator)
    logging.info("Discriminator Variables")
    parameter_overview.log_parameter_overview(self.discriminator)

  def train_one_step_test(self):
    """Conducts one test training step and log variables."""
    self.train_cluster(1)
    logging.info("Finished test training step")
    self.print_summary()
