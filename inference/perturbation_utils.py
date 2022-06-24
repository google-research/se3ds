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

"""Utility functions for perturbation."""
import math

import numpy as np
from se3ds import constants
import tensorflow as tf


def get_proportion_invalid_for_depth(position_offset: tf.Tensor,
                                     depth_image: tf.Tensor,
                                     distance_padding: float = 0.10):
  """Returns the proportion of collided pixels when moving in a given direction.

  Args:
    position_offset: (3,) tensor of relative xyz position to move towards.
    depth_image: (H, W) image with values in [0, 1] describing a depth map.
    distance_padding: Maximum threshold in meters between camera and an object.

  Returns:
    proportion_invalid: Proportion of collided pixels.
  """
  distance = tf.reduce_sum(position_offset**2)**0.5
  height, width = depth_image.shape

  # Compute heading / elevation of travel.
  heading = tf.math.atan2(-position_offset[0], -position_offset[1])
  # Map to [0, 2pi] domain.
  heading = heading + (2 * math.pi) * tf.cast(heading <= 0,
                                              tf.float32) % (2 * math.pi)
  if heading < 0:
    heading += 2 * math.pi
  heading_proportion = heading / (2 * math.pi)

  delta_xy = math.sqrt(position_offset[1]**2 + position_offset[0]**2)
  elevation = tf.math.atan2(delta_xy, -position_offset[2])
  # Map to [0, pi] domain.
  elevation = elevation + math.pi * tf.cast(elevation <= 0,
                                            tf.float32) % math.pi
  if elevation < 0:
    elevation += math.pi
  elevation_proportion = elevation / math.pi

  heading_start = int(heading_proportion * width)
  elevation_start = int(elevation_proportion * height)

  # Look around 30º heading and 60º elevation to check for collisions.
  threshold_width = int(30 / 360 * width)
  threshold_height = int(60 / 180 * height)
  depth_consider_region = depth_image[
      max(0, elevation_start -
          threshold_height):min(height, elevation_start + threshold_height),
      max(0, heading_start -
          threshold_width):min(width, heading_start + threshold_width)]
  proportion_invalid = np.mean(
      depth_consider_region * constants.DEPTH_SCALE < distance +
      distance_padding)
  return proportion_invalid
