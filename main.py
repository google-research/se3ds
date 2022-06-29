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
"""Starts training and (possibly) evaluation jobs."""
import enum

from absl import app
from absl import flags
from absl import logging
import gin
from se3ds.trainers import se3ds_trainer
import tensorflow as tf

FLAGS = flags.FLAGS


@enum.unique
class _Mode(enum.Enum):
  TRAIN = "TRAIN"
  TEST = "TEST"
  TEST_UNSEEN = "TEST_UNSEEN"


flags.DEFINE_string("model_dir", None, "Directory to save trained model in.")
flags.DEFINE_bool("use_tpu", False, "Whether to run on TPU or not.")
flags.DEFINE_enum_class("mode", _Mode.TRAIN, _Mode, "job status")

flags.DEFINE_multi_string("gin_config", [],
                          "List of paths to the config files.")
flags.DEFINE_multi_string("gin_bindings", [],
                          "Newline separated list of Gin parameter bindings.")


def main(unused_argv):
  gin.parse_config_files_and_bindings(FLAGS.gin_config, FLAGS.gin_bindings)
  if FLAGS.use_tpu:
    tpu_worker = ""  # CHANGEME: Fill in with your TPU worker.
    job_name = ""  # CHANGEME: Fill in with your job name.
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_worker, job_name=job_name)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
  else:
    for device_type in ["GPU", "CPU"]:
      devices = tf.config.experimental.list_logical_devices(
          device_type=device_type)
      if devices:
        break
    if len(devices) == 1:
      strategy = tf.distribute.OneDeviceStrategy(devices[0])
    else:
      strategy = tf.distribute.MirroredStrategy(devices)
  logging.info("FLAGS.use_tpu %s", FLAGS.use_tpu)
  logging.info("Distribution strategy: %s", strategy)

  model = se3ds_trainer.GAN(strategy=strategy, model_dir=FLAGS.model_dir)
  if FLAGS.mode is _Mode.TRAIN:
    model.train()
  elif FLAGS.mode is _Mode.TEST:
    model.test()
  elif FLAGS.mode is _Mode.TEST_UNSEEN:
    model = se3ds_trainer.GAN(
        strategy=strategy, model_dir=FLAGS.model_dir, test_split="val_unseen")
    model.test()


if __name__ == "__main__":
  flags.mark_flag_as_required("model_dir")
  app.run(main)
