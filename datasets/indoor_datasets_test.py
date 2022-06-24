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

"""Tests for se3ds.datasets.indoor_datasets."""

import itertools
import os

from absl import flags
from absl.testing import parameterized
from se3ds import constants
from se3ds.datasets import indoor_datasets
import tensorflow as tf

FLAGS = flags.FLAGS


class IndoorDatasetTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the indoor_datasetss file."""

  def setUp(self):
    super(IndoorDatasetTest, self).setUp()

    root_dir = os.path.join(FLAGS.test_srcdir, "datasets/testdata/")
    self.image_filenames = os.path.join(root_dir, "train.tfrecord")
    self.video_filenames = os.path.join(root_dir, "val.tfrecord")

  @parameterized.parameters(list(itertools.product(
      (1, 2), ("train",), (256, 512), (True, False))))
  def test_spade_image_preprocessing(self, batch_size, split, image_size,
                                     random_crop):
    """Tests that preprocessing outputs the correct image shapes."""
    dataset = indoor_datasets.R2RImageDataset(
        image_size=image_size,
        return_filename=True,
        horizontal_mask_ratio=0.1,
        vertical_mask_ratio=0.1,
        random_crop=random_crop)
    dataset_handler = dataset.input_fn(
        split=split,
        global_batch_size=batch_size,
        file_pattern=self.image_filenames,
        num_epochs=1)
    out_image_size = image_size

    for ex in dataset_handler:
      self.assertAllEqual(ex["image"].shape,
                          (batch_size, out_image_size, out_image_size * 2, 3))
      self.assertAllInRange(ex["image"], 0.0, 1.0)
      self.assertAllEqual(ex["proj_image"].shape,
                          (batch_size, out_image_size, out_image_size * 2, 3))
      self.assertAllInRange(ex["proj_image"], 0.0, 1.0)
      self.assertAllEqual(ex["proj_mask"].shape,
                          (batch_size, out_image_size, out_image_size * 2, 1))
      self.assertAllInSet(ex["proj_mask"], (0.0, 1.0))
      self.assertAllEqual(ex["segmentation"].shape,
                          (batch_size, out_image_size, out_image_size * 2, 1))
      self.assertAllEqual(ex["depth"].shape,
                          (batch_size, out_image_size, out_image_size * 2, 1))
      self.assertAllInRange(ex["depth"], 0, 1)
      self.assertAllEqual(ex["filename"].shape, (batch_size,))

  @parameterized.parameters(list(itertools.product(
      (1, 2), ("train", "val"), (256, 512),)))
  def test_spade_video_preprocessing(self, batch_size, split, image_size):
    """Tests that preprocessing outputs the correct video shapes."""
    dataset = indoor_datasets.R2RVideoDataset(
        image_size=image_size, return_filename=True, horizontal_mask_ratio=0.1)
    dataset_handler = dataset.input_fn(
        split=split,
        global_batch_size=batch_size,
        file_pattern=self.video_filenames,
        num_epochs=1)

    for ex in dataset_handler:
      self.assertAllEqual(ex["image"].shape,
                          (batch_size, constants.PANO_VIDEO_LENGTH,
                           dataset.image_size, dataset.image_size * 2, 3))
      self.assertAllInRange(ex["image"], 0.0, 1.0)
      self.assertAllEqual(ex["segmentation"].shape,
                          (batch_size, constants.PANO_VIDEO_LENGTH,
                           dataset.image_size, dataset.image_size * 2, 1))
      self.assertAllEqual(ex["depth"].shape,
                          (batch_size, constants.PANO_VIDEO_LENGTH,
                           dataset.image_size, dataset.image_size * 2, 1))
      self.assertAllInRange(ex["depth"], 0, 1)

if __name__ == "__main__":
  tf.test.main()
