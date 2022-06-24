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

"""Utility functions for image grid generation."""

import math
from typing import  Union, List, Dict
from absl import logging

import tensorflow as tf


def images_to_grid(images: tf.Tensor) -> tf.Tensor:
  """Transfer batch images to image grid."""

  ny, nx, h, w, c = images.shape
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, ny * h, nx * w, c])
  return images


def get_grid_image(
    x: tf.Tensor,
    show_num: int,
    strategy: tf.distribute.Strategy) -> tf.Tensor:
  """Concatenate image in each replica together for image grid."""
  x = strategy.experimental_local_results(x)
  x = tf.concat(x, axis=0)
  if x.shape[0] < show_num:
    logging.info("show_num is cut by the small batch size to %s", x.shape[0])
    show_num = x.shape[0]
  x = tf.cast(x[0:show_num] * 255.0, tf.uint8)
  x = tf.squeeze(x)
  h_num = int(math.sqrt(show_num))
  w_num = int(show_num / h_num)
  grid_num = h_num * w_num
  _, height, width, channel = x.shape.as_list()
  x = tf.reshape(x[0:grid_num], (h_num, w_num, height, width, channel))
  x = images_to_grid(x)
  return x


def get_grid_image_dict(
    images: Union[tf.Tensor, List[tf.Tensor], Dict[str, tf.Tensor]],
    show_num: int,
    strategy: tf.distribute.Strategy,
    name_prefix: str) -> Dict[str, tf.Tensor]:
  """Concatenate image in each replica together for image grid as dict.

  Args:
    images: Images in each replica, it can be tf.Tensor, list or dictionary.
    show_num: The number of images used for generating image_grid.
    strategy: tf.distribute.Strategy for merging images from each replica.
    name_prefix: The prefix of the generated image grid.

  Returns:
    out_dict: The generated image grid.
  """

  def _get_grid_image(x, show_num):
    x = strategy.experimental_local_results(x)
    x = tf.concat(x, axis=0)
    if x.shape[0] < show_num:
      logging.info("show_num is cut by the small batch size to %d", x.shape[0])
      show_num = x.shape[0]
    x = tf.cast(x[0:show_num] * 255.0, tf.uint8)
    x = tf.squeeze(x)
    h_num = int(math.sqrt(show_num))
    w_num = int(show_num / h_num)
    grid_num = h_num * w_num
    _, height, width, channel = x.shape.as_list()
    x = tf.reshape(x[0:grid_num], (h_num, w_num, height, width, channel))
    x = images_to_grid(x)
    return x
  out_dict = {}
  if isinstance(images, list):
    for i in range(len(images)):
      index_name = name_prefix + "_" + str(i)
      out_dict[index_name] = _get_grid_image(images[i], show_num)
  elif isinstance(images, dict):
    for key, value in images.items():
      out_dict[name_prefix + "_" + key] = _get_grid_image(value, show_num)
  else:
    out_dict[name_prefix] = _get_grid_image(images, show_num)
  return out_dict


