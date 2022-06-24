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

"""Tests for run_perturbation."""

from absl import flags
from absl.testing import parameterized
import numpy as np
from se3ds import constants
from se3ds.inference import perturbation_utils
import tensorflow as tf

FLAGS = flags.FLAGS


class RunPerturbationTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the run_perturbation file."""

  @parameterized.parameters((0.5, 0.5, 1.0), (0.3, 0.5, 0.0))
  def test_proportion_invalid(self, distance, depth_distance,
                              expected_proportion_invalid):
    """Tests get_proportion_invalid_for_depth method."""
    height, width = 64, 128
    test_depth_image = tf.fill((height, width),
                               depth_distance / constants.DEPTH_SCALE)
    position_offset = tf.constant([0.0, distance, 0.0])
    proportion_invalid = perturbation_utils.get_proportion_invalid_for_depth(
        position_offset, test_depth_image)
    self.assertEqual(proportion_invalid, expected_proportion_invalid)

  def test_proportion_invalid_offset_forward(self):
    """Tests get_proportion_invalid_for_depth method."""
    height, width = 64, 128
    padding = 10
    position_offset = [0.0, 0.5, 0.0]
    test_depth_image = np.full((height, width), 1.0)
    # Set a small region in the traversal direction to be close.
    test_depth_image[height // 2 - padding:height // 2 + padding,
                     width // 2 - padding:width // 2 + padding] = 0.0
    test_depth_image = tf.constant(test_depth_image, dtype=tf.float32)

    position_offset = tf.constant(position_offset, dtype=tf.float32)
    proportion_invalid = perturbation_utils.get_proportion_invalid_for_depth(
        position_offset, test_depth_image)
    self.assertGreater(proportion_invalid, 0.0)

    # Check that only the traversal region is checked.
    test_depth_image = np.full((height, width), 1.0)
    test_depth_image[:padding, :padding] = 0.0
    test_depth_image = tf.constant(test_depth_image, dtype=tf.float32)

    position_offset = tf.constant(position_offset, dtype=tf.float32)
    proportion_invalid = perturbation_utils.get_proportion_invalid_for_depth(
        position_offset, test_depth_image)
    self.assertEqual(proportion_invalid, 0.0)

  def test_proportion_invalid_offset_diagonal(self):
    """Tests get_proportion_invalid_for_depth method with diagonal movement."""
    height, width = 64, 128
    padding = 10
    position_offset = [0.5, 0.5, 0.0]
    test_depth_image = np.full((height, width), 1.0)
    # Set a small region in the traversal direction to be close.
    height_start = int(height * 3 / 4)
    width_start = int(width * 3 / 4)
    test_depth_image[height_start - padding:height_start + padding,
                     width_start - padding:width_start + padding] = 0.0
    test_depth_image = tf.constant(test_depth_image, dtype=tf.float32)

    position_offset = tf.constant(position_offset, dtype=tf.float32)
    proportion_invalid = perturbation_utils.get_proportion_invalid_for_depth(
        position_offset, test_depth_image)
    self.assertGreater(proportion_invalid, 0.0)

    # Check that only the traversal region is checked.
    test_depth_image = np.full((height, width), 1.0)
    test_depth_image[:padding, :padding] = 0.0
    test_depth_image = tf.constant(test_depth_image, dtype=tf.float32)

    position_offset = tf.constant(position_offset, dtype=tf.float32)
    proportion_invalid = perturbation_utils.get_proportion_invalid_for_depth(
        position_offset, test_depth_image)
    self.assertEqual(proportion_invalid, 0.0)


if __name__ == '__main__':
  tf.test.main()
