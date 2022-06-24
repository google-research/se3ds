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
"""Tests for se3ds.trainers.se3ds_trainer."""

import datetime
import os

from absl import flags
from absl import logging
from absl.testing import parameterized
import gin
from se3ds.trainers import se3ds_trainer
import tensorflow as tf

FLAGS = flags.FLAGS


class SE3DSGANTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the Generator and Discriminator of the cond_model."""

  def setUp(self):
    super().setUp()
    gin.clear_config()

    # When the test is run on machines with TPUs or GPUs (go/forge-for-ml), we
    # can test with a corresponding strategy.
    tpus = tf.config.experimental.list_logical_devices(device_type="TPU")
    gpus = tf.config.experimental.list_logical_devices(device_type="GPU")
    if tpus:
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver("")
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif gpus:
      strategy = tf.distribute.MirroredStrategy([g.name for g in gpus])
    else:
      strategy = tf.distribute.OneDeviceStrategy("CPU")

    root_dir = os.path.join(FLAGS.test_srcdir, "datasets/")
    train_file_glob = os.path.join(root_dir, "testdata/train.tfrecord")
    test_file_glob = os.path.join(root_dir, "testdata/val.tfrecord")
    assert tf.io.gfile.glob(
        train_file_glob
    ), f"Training files could not be found: {tf.io.gfile.glob(train_file_glob)}"

    self.strategy = strategy
    self.train_file_glob = train_file_glob
    self.test_file_glob = test_file_glob

  def _get_empty_directory(self):
    sub_dir = str(datetime.datetime.now().microsecond)
    empty_dir = os.path.join(FLAGS.test_tmpdir, sub_dir)
    assert not tf.io.gfile.exists(empty_dir)
    return empty_dir

  def parse_config(self, d_step_per_g_step, num_batched_steps, model,
                   predict_depth):
    gin.parse_config(f"""
    GANManager.generator_fn = {model}
    GANManager.discriminator_fn = @image_models.SNMultiScaleDiscriminator
    GANManager.log_every_steps = 1
    GANManager.save_every_steps = 1
    GANManager.eval_every_steps = 1
    GANManager.shuffle_buffer_size = 2
    GANManager.train_batch_size = 2
    GANManager.test_batch_size = 2
    GANManager.d_step_per_g_step = {d_step_per_g_step}
    GANManager.num_batched_steps = {num_batched_steps}
    GANManager.eval_size = 2
    GANManager.image_size = 128
    R2RImageDataset.image_size = 128
    R2RVideoDataset.image_size = 128
    image_models.SNMultiScaleDiscriminator.n_dis = 1
    image_models.SNMultiScaleDiscriminator.dis_dims = 2
    image_models.SNMultiScaleDiscriminator.n_layers = 2
    image_models.ResNetGenerator.gen_dims = 2
    image_models.ResNetGenerator.z_dim = 2
    image_models.ResNetGenerator.image_size = 128
    image_models.ResNetGenerator.conv_mode = 'normal'
    image_models.ResNetGenerator.context_layer = 'none'
    GAN.predict_depth = {predict_depth}
    se3ds_trainer.GAN.dis_use_pred_depth = {predict_depth}
    GAN.lambda_gan = 1
    GAN.lambda_kld = 0.05
    GAN.lambda_wc = 1.0
    GAN.lambda_depth = 1.0
    """)

  @parameterized.parameters(
      (1, 1, "@image_models.ResNetGenerator", True))
  def test_train(self, d_step_per_g_step, num_batched_steps, model,
                 predict_depth):
    self.parse_config(d_step_per_g_step, num_batched_steps, model,
                      predict_depth)
    model_dir = model_dir = os.path.join(FLAGS.test_tmpdir, "model")
    model = se3ds_trainer.GAN(
        strategy=self.strategy,
        model_dir=model_dir,
        train_dataset_glob=self.train_file_glob,
        test_dataset_glob=self.test_file_glob,
        num_epochs=-1,
        eval_size=2)
    model.train()
    files_in_model_dir = tf.io.gfile.listdir(model_dir)
    logging.info("files_in_model_dir: %s", files_in_model_dir)
    expected_model_files = ["ckpt-1.index"]
    self.assertAllInSet(expected_model_files, files_in_model_dir)

  @parameterized.parameters(
      ("@image_models.ResNetGenerator", True))
  def test_test(self, model, predict_depth):
    self.parse_config(
        d_step_per_g_step=1,
        num_batched_steps=1,
        model=model,
        predict_depth=predict_depth)
    model_dir = model_dir = os.path.join(FLAGS.test_tmpdir, "model")
    model = se3ds_trainer.GAN(
        strategy=self.strategy,
        model_dir=model_dir,
        train_dataset_glob=self.train_file_glob,
        test_dataset_glob=self.test_file_glob,
        num_epochs=-1,
        eval_size=2)
    model.test(unit_test=True)


if __name__ == "__main__":
  tf.test.main()
