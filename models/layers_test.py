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

"""Tests for se3ds.models.layers."""

import os

from absl import flags
from absl.testing import parameterized
from se3ds.models import layers
import tensorflow as tf

FLAGS = flags.FLAGS


class LayersTest(tf.test.TestCase, parameterized.TestCase):
  """Tests on the image_models file."""

  @parameterized.parameters((1, 128, 1), (2, 256, 2))
  def test_resstack(self, batch_size, image_size, strides):
    """Tests that ResStack model outputs correct shapes."""
    input_dim = 32
    expansion = 4
    blocks = 2
    output_dim = expansion * input_dim
    test_model = layers.ResStack(input_dim, input_dim, blocks, strides,
                                 expansion)
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, input_dim), dtype=tf.float32)
    test_output, test_mask = test_model(test_input)
    self.assertEqual(
        test_output.shape,
        (batch_size, image_size // strides, image_size // strides, output_dim))
    self.assertEqual(
        test_mask.shape,
        (batch_size, image_size // strides, image_size // strides, 1))

  @parameterized.parameters((1, 64, 1), (2, 128, 2))
  def test_resstack_transposed(self, batch_size, image_size, strides):
    """Tests that ResStackTranspose model outputs correct shapes."""
    input_dim = 32
    output_dim = 16
    blocks = 2
    test_model = layers.ResStackTranspose(input_dim, output_dim, blocks,
                                          strides)
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, input_dim), dtype=tf.float32)
    test_output = test_model(test_input)
    self.assertEqual(
        test_output.shape,
        (batch_size, image_size * strides, image_size * strides, output_dim))

  def test_resstack_masking(self):
    """Tests that ResStack models mask inputs correctly."""
    batch_size, image_size, input_dim, output_dim = 2, 128, 32, 16
    test_model = layers.ResStack(input_dim, output_dim, blocks=2, strides=1)
    test_input = tf.random.uniform(
        (batch_size, image_size, image_size, input_dim), dtype=tf.float32)
    # Create binary mask for testing. Left half of mask is zeroed out.
    test_mask_range = tf.range(image_size, dtype=tf.float32)
    test_mask = tf.cast(test_mask_range > image_size // 2, tf.float32)
    test_mask = tf.tile(
        test_mask[None, :, None, None], [batch_size, 1, image_size, 1])

    test_output, _ = test_model(test_input, mask=test_mask)

    # Change a pixel in the left region of the input. This should not affect the
    # output since this area is masked.
    test_input2 = test_input.numpy()
    test_input2[:, 0, 0, :] = 1
    test_input2 = tf.constant(test_input2)

    test_output2, _ = test_model(test_input2, mask=test_mask)

    self.assertAllEqual(test_output, test_output2)

  @parameterized.parameters((4, 8, 8, 3, 2, 64), (4, 8, 8, 3, 2, 32),
                            (4, 16, 8, 1, 1, 32))
  def test_spectral_conv(self, batch_size, input_dims, output_dims, kernel_size,
                         strides, input_size):
    """Tests that spectral convolution outputs are of the correct shapes."""
    # Require TPU / GPU to run grouped convolutions.
    spectral_conv = layers.SpectralConv(
        output_dims, kernel_size=kernel_size, strides=strides)
    # Use a regular conv to determine shape.
    normal_conv = tf.keras.layers.Conv2D(
        output_dims, kernel_size=kernel_size, strides=strides)

    test_input = tf.random.uniform(
        (batch_size, input_size, input_size, input_dims))
    test_output = spectral_conv(test_input)
    normal_output = normal_conv(test_input)
    self.assertAllEqual(test_output.shape, normal_output.shape)

  @parameterized.parameters((1, 3, 2), (4, 5, 1))
  def test_partial_conv(self, batch_size, kernel_size, strides):
    output_dims = 16
    input_dims = 32
    input_size = 32
    partial_conv = layers.PartialConv(
        output_dims, kernel_size=kernel_size, strides=strides)
    spectral_partial_conv = layers.PartialSpectralConv(
        output_dims, kernel_size=kernel_size, strides=strides)
    normal_conv = tf.keras.layers.Conv2D(
        output_dims, kernel_size=kernel_size, strides=strides)
    test_input = tf.random.uniform(
        (batch_size, input_size, input_size, input_dims))
    test_mask = tf.random.uniform((batch_size, input_size, input_size, 1))
    test_mask = tf.cast(test_mask > 0.5, tf.float32)
    test_output, _ = partial_conv(test_input, test_mask)
    normal_output = normal_conv(test_input)
    self.assertAllEqual(test_output.shape, normal_output.shape)

    test_partial_spectral_out, _ = spectral_partial_conv(test_input, test_mask)
    self.assertAllEqual(test_partial_spectral_out.shape, normal_output.shape)

    # Test that output is equivalent to a regular convolution in the absence of
    # a mask.
    normal_out2 = tf.nn.conv2d(
        input=test_input, filters=partial_conv.kernel, strides=strides,
        padding=partial_conv.padding.upper())
    test_output2, _ = partial_conv(test_input)
    self.assertAllClose(test_output2, normal_out2)

  @parameterized.parameters((True, 'CONSTANT'), (False, 'CONSTANT'),
                            (True, 'SYMMETRIC'))
  def test_pad_layer(self, circular_pad, mode):
    pad = layers.PadLayer(2, circular_pad, mode=mode)
    test_input = tf.constant([[1.0, 3.0, 2.0, 2.0], [1.0, 1.0, 2.0, 2.0],
                              [1.0, 1.0, 2.0, 2.0], [2.0, 0.0, 3.0, 3.0]],
                             dtype=tf.float32)
    test_input = tf.reshape(test_input, (1, 4, 4, 1))
    test_output = pad(test_input)
    if mode == 'CONSTANT':
      if circular_pad:
        expected_output = tf.constant(
            [[[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
              [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
              [[2.], [2.], [1.], [3.], [2.], [2.], [1.], [3.]],
              [[2.], [2.], [1.], [1.], [2.], [2.], [1.], [1.]],
              [[2.], [2.], [1.], [1.], [2.], [2.], [1.], [1.]],
              [[3.], [3.], [2.], [0.], [3.], [3.], [2.], [0.]],
              [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
              [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]]],
            dtype=tf.float32)
      else:
        expected_output = tf.constant(
            [[[[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
              [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
              [[0.], [0.], [1.], [3.], [2.], [2.], [0.], [0.]],
              [[0.], [0.], [1.], [1.], [2.], [2.], [0.], [0.]],
              [[0.], [0.], [1.], [1.], [2.], [2.], [0.], [0.]],
              [[0.], [0.], [2.], [0.], [3.], [3.], [0.], [0.]],
              [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
              [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]]],
            dtype=tf.float32)
    elif mode == 'SYMMETRIC' and circular_pad:
      expected_output = tf.constant(
          [[[[2.], [2.], [1.], [1.], [2.], [2.], [1.], [1.]],
            [[2.], [2.], [1.], [3.], [2.], [2.], [1.], [3.]],
            [[2.], [2.], [1.], [3.], [2.], [2.], [1.], [3.]],
            [[2.], [2.], [1.], [1.], [2.], [2.], [1.], [1.]],
            [[2.], [2.], [1.], [1.], [2.], [2.], [1.], [1.]],
            [[3.], [3.], [2.], [0.], [3.], [3.], [2.], [0.]],
            [[3.], [3.], [2.], [0.], [3.], [3.], [2.], [0.]],
            [[2.], [2.], [1.], [1.], [2.], [2.], [1.], [1.]]]],
          dtype=tf.float32)
    self.assertAllClose(test_output, expected_output)


if __name__ == '__main__':
  tf.test.main()
