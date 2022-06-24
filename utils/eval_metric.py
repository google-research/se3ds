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

"""EvalMetric class for to calculate FID and Inception Scores for GANs.


Typical usage example:
>>> eval_metric = EvalMetric(
      ds=ds,
      eval_num=10000,
      batch_size=256,
      strategy=strategy)
or
>>> eval_metric = EvalMetric(
      ds=ds,
      eval_num=10000,
      batch_size=256,
      strategy=strategy,
      arg_num=3,
      num_splits=1)

strategy should  be Union [tf.distribute.OneDeviceStrategy,
                           tf.distribute.MirroredStrategy,
                           tf.distribute.experimental.TPUStrategy].

ds should be an instance of tf.data.Dataset.


If the number of eval image need to be changed to another value, e.g., 30000.
>>> eval_metric.eval_num = 30000

If batch size need to be changed to another value, e.g., 512.
>>> eval_metric.batch_size = 512

If the number of calucation to get the mean and standard deviation for FID
and Inception Scoreneed to be changed to another value, e.g., 5.
>>> eval_metric.arg_num = 5

If for each calucation, the splits need to be changed to another value, e.g., 10
>>> eval_metric.num_splits = 10

fid, fid_std = eval_metric(generator_fn)
where generator_fn is a function for GAN Generator to generate synthetic images.
"""

from absl import logging
import numpy as np
from se3ds import constants
from se3ds.datasets import indoor_datasets
from se3ds.utils import inception_utils
from se3ds.utils import pano_utils
import tensorflow as tf


class EvalMetric:
  """Evaluation class for FID and Inception Score.

  Attributes:
    ds: A tf.data.Dataset object for providing the evaluation data.
    eval_num: A integer count for the number of images for calculating the
      inception score and FID, preferbaly bigger than 10000.
    batch_size: Batch size for each forward operation.
    strategy: A tf.distribute.Strategy object for TPU/GPU processing.
    avg_num: A integer count for the number of calucation to get the mean and
      standard deviation for FID and Inception Score.
    num_splits: A integer count for the
      number of splits for eval_num for the calculation.
    eval_seq_len: Sequence length of videos to perform evaluation on.
  """

  def __init__(
      self,
      ds: tf.data.Dataset,
      eval_num: int,
      batch_size: int,
      strategy: tf.distribute.Strategy,
      avg_num: int = 3,
      num_splits: int = 1,
      eval_seq_len: int = 5) -> None:
    """Creates a new metric evaluation object."""

    tf.config.set_soft_device_placement(True)
    self.ds = ds
    self.eval_num = eval_num
    self.batch_size = batch_size
    self.strategy = strategy
    self.avg_num = avg_num
    self.num_splits = num_splits
    self.eval_seq_len = eval_seq_len
    with self.strategy.scope():
      self._inception_model = inception_utils.inception_model()
    # Calculates pooling feature for real image only once to save time.
    self._pool = self._get_real_pool_for_evaluation()

  @tf.function
  def _get_real_pool(self):
    """Gets pooling features for real images per batch.

    Returns:
      pool: tf.float32 tensor of shape [batch_size, 2048].

    """
    def step_fn(inputs):
      image = inputs["original_image"]
      all_pool = {}
      for i in range(1, self.eval_seq_len):
        aug_image = indoor_datasets.augment(image[:, i, ...])
        aug_image = pano_utils.crop_pano(aug_image, resize_to_original=False)
        pool_val, _ = inception_utils.get_inception(
            aug_image, self._inception_model)
        all_pool[i] = pool_val
        # all_logits[i] = logits_val
      return all_pool

    pool = self.strategy.run(step_fn, args=(next(self.ds),))
    pool = {
        i: tf.concat(self.strategy.experimental_local_results(p), axis=0)
        for i, p in pool.items()
    }
    return pool

  @tf.function
  def _get_generated_pool(self, generator_fn):
    """Gets pooling/logits features for generated images per batch.

    Args:
      generator_fn: generator function to get generated images.

    Returns:
      pool: tf.float32 tensor of shape [batch_size, 2048]
      logits: tf.float32 tensor of shape [batch_size, 1000]
    """
    def step_fn(inputs):
      batch_size, _, height, width, _ = inputs["image"].shape
      memory_coords = tf.zeros((batch_size, 4, 0))
      memory_feats = tf.zeros((batch_size, 0, 3), dtype=tf.int32)
      prev_rgb_tensor = None
      all_generated = []
      all_depth_rmse = []
      # All depth_scale within a batch should be the same.
      depth_scale = inputs["depth_scale"][0]
      for frame_idx in range(self.eval_seq_len):
        target_depth = inputs["depth"][:, frame_idx, ...]  # (N, H, W, 1)
        rgb_tensor = inputs["image"][:, frame_idx, ...]  # (N, H, W, 3)
        segmentation_tensor = tf.cast(  # (N, H, W, D)
            inputs["segmentation"][:, frame_idx, ...], tf.float32)
        depth_tensor = inputs["depth"][:, frame_idx, ...]  # (N, H, W, 1)


        relative_position = inputs["position"][:, frame_idx, ...]
        relative_coords = memory_coords - relative_position[..., None]
        pred_depth, pred_rgb = (
            pano_utils.project_feats_to_equirectangular(
                memory_feats, relative_coords, height, width,
                constants.INVALID_RGB_VALUE, depth_scale))

        pred_mask = tf.cast(
            tf.math.logical_and(
                tf.math.logical_and(pred_depth > 0, pred_depth < 1),
                tf.math.reduce_all(
                    pred_rgb != constants.INVALID_RGB_VALUE, axis=-1),
            ), tf.float32)[..., None]
        pred_depth = pred_depth[..., None]
        pred_rgb = tf.clip_by_value(tf.cast(pred_rgb / 255, tf.float32), 0, 1)
        # For eval, generate without blurred pixels.
        blurred_mask = tf.zeros_like(pred_depth)

        segmentation_tensor = tf.cast(segmentation_tensor, tf.int32)
        one_hot_tensor = tf.one_hot(segmentation_tensor[..., 0],
                                    constants.NUM_MP3D_CLASSES)
        if prev_rgb_tensor is None:
          prev_rgb_tensor = tf.zeros_like(rgb_tensor)
        first_frame = tf.ones((batch_size,))
        if frame_idx > 0:
          first_frame = tf.zeros((batch_size,))
        generator_inputs = {
            "one_hot_mask": one_hot_tensor,
            "depth": depth_tensor,
            "prev_image": prev_rgb_tensor,
            "proj_image": pred_rgb,
            "proj_mask": pred_mask,
            "proj_depth": pred_depth,
            "first_frame": first_frame,
            "blurred_mask": blurred_mask,
            # TODO: Update according to actual ID for eval dataset.
            # Generate for MP3D.
            "dataset_type": inputs["dataset_type"],
        }
        _, _, _, depth_out, _, _, generated = generator_fn(
            inputs=[generator_inputs, None], training=False)

        # Add points to point cloud memory.
        if frame_idx == 0:
          # For groundtruth, we mask out the blurred regions on the top/bottom.
          prev_rgb_tensor = rgb_tensor
          rgb_tensor = pano_utils.mask_pano(
              rgb_tensor,
              masked_region_value=constants.INVALID_RGB_VALUE)
        else:  # For future frames, add generated points.
          rgb_tensor = generated
          prev_rgb_tensor = generated
          if depth_out is not None:
            depth_tensor = depth_out

        # Compute RMSE.
        target_spatial_mask = tf.cast(
            tf.math.logical_and(target_depth > 0, target_depth < 1),
            tf.float32)
        assert depth_tensor.shape == target_depth.shape, (depth_tensor.shape,
                                                          target_depth.shape)
        depth_diff = (depth_tensor - target_depth)**2 * target_spatial_mask
        # Dimension 3 has channel size of 1
        depth_diff = tf.reduce_sum(depth_diff, axis=(1, 2, 3)) / tf.maximum(
            tf.reduce_sum(target_spatial_mask, axis=(1, 2, 3)), 1)
        depth_rmse = tf.sqrt(depth_diff)
        all_depth_rmse.append(depth_rmse)

        # Add points to point cloud memory.
        pc_rgb_tensor = tf.cast(rgb_tensor * 255, tf.int32)
        pc_rgb_tensor = tf.clip_by_value(
            pc_rgb_tensor, constants.INVALID_RGB_VALUE, 255)
        xyz1, feats = pano_utils.equirectangular_to_pointcloud(
            pc_rgb_tensor, depth_tensor[..., 0],
            constants.INVALID_RGB_VALUE,
            depth_scale)
        xyz1 += relative_position[..., None]
        memory_coords = tf.concat([memory_coords, xyz1], axis=2)
        memory_feats = tf.concat([memory_feats, feats], axis=1)
        all_generated.append(generated)

      pool_val, logits_val = {}, {}
      rmse = {}
      for frame_idx in range(1, self.eval_seq_len):
        aug_image = indoor_datasets.augment(all_generated[frame_idx])
        aug_image = pano_utils.crop_pano(aug_image, resize_to_original=False)
        curr_pool_val, curr_logits_val = inception_utils.get_inception(
            aug_image, self._inception_model)
        pool_val[frame_idx] = curr_pool_val
        logits_val[frame_idx] = curr_logits_val
        rmse[frame_idx] = all_depth_rmse[frame_idx]
      return pool_val, logits_val, rmse

    pool, logits, rmse = self.strategy.run(step_fn, args=(next(self.ds),))
    pool = {
        i: tf.concat(self.strategy.experimental_local_results(p), axis=0)
        for i, p in pool.items()
    }
    logits = {
        i: tf.concat(self.strategy.experimental_local_results(l), axis=0)
        for i, l in logits.items()
    }
    rmse = {
        i: tf.concat(self.strategy.experimental_local_results(l), axis=0)
        for i, l in rmse.items()
    }
    return pool, logits, rmse

  def _get_real_pool_for_evaluation(self):
    """Gets numpy arrays for pooling features and logits for real images."""

    logging.info("Get pool for %d samples", self.eval_num)
    n_iter = (self.eval_num // self.batch_size) + 1
    pool = {i: [] for i in range(1, self.eval_seq_len)}
    for j in range(n_iter):
      pool_val = self._get_real_pool()
      for i in range(1, self.eval_seq_len):
        pool[i].append(pool_val[i].numpy())
      if j % 10 == 0:
        logging.info("Real pool: %d / %d", j, n_iter)
    pool_total = {
        k: np.concatenate(v, 0)[0:self.eval_num] for k, v in pool.items()
    }
    logging.info("Active evaluation size for real data: %d",
                 pool_total[1].shape[0])
    return pool_total

  def _get_generated_pool_for_evaluation(self, generator_fn):
    """Gets numpy arrays for pooling features and logits for generated images."""

    n_iter = (self.eval_num // self.batch_size) + 1
    pool = {i: [] for i in range(1, self.eval_seq_len)}
    logits = {i: [] for i in range(1, self.eval_seq_len)}
    rmse = {i: [] for i in range(1, self.eval_seq_len)}
    for j in range(n_iter):
      pool_val, logits_val, rmse_scores = self._get_generated_pool(generator_fn)
      for i in range(1, self.eval_seq_len):
        pool[i].append(pool_val[i].numpy())
        logits[i].append(logits_val[i].numpy())
        rmse[i].append(rmse_scores[i].numpy())
      if j % 10 == 0:
        logging.info("Generated pool: %d / %d", j, n_iter)

    pool_total = {
        k: np.concatenate(v, 0)[0:self.eval_num] for k, v in pool.items()
    }
    logits_total = {
        k: np.concatenate(v, 0)[0:self.eval_num] for k, v in logits.items()
    }
    rmse_total = {
        k: np.concatenate(v, 0)[0:self.eval_num] for k, v in rmse.items()
    }
    logging.info("Active evaluation size for generated data: %d",
                 pool_total[1].shape[0])
    return pool_total, logits_total, rmse_total

  def calculate_fid_score(self, generator_fn):
    """Calculates Inception score and FID.

    Args:
      generator_fn: A function for GAN Generator to generate synthetic images.

    Returns:
      fid: The average FID score for the generated images.
      fid_std: The standard deviation of FID for the generated images.
    """

    fid_list = {i: [] for i in range(1, self.eval_seq_len)}
    rmse_list = {i: [] for i in range(1, self.eval_seq_len)}
    logging.info("Calculate Generator Statistics")
    for _ in range(self.avg_num):
      generated_pool, _, rmse_total = self._get_generated_pool_for_evaluation(
          generator_fn)
      for i in range(1, self.eval_seq_len):
        fid = inception_utils.calculate_fid(generated_pool[i], self._pool[i])
        fid_list[i].append(fid)
        rmse_list[i].append(np.mean(rmse_total[i]))

    fid = {k: np.mean(v) for k, v in fid_list.items()}
    fid_std = {k: np.std(v) for k, v in fid_list.items()}
    rmse = {k: np.mean(v) for k, v in rmse_list.items()}

    return fid, fid_std, rmse
