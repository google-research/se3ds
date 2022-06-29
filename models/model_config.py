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

"""Config object for SE3DS models."""

from typing import Optional

from se3ds import constants


class SE3DSConfig:
  """Parameters used to configure SE3DS models."""
  batch_size: int = 1
  ckpt_path: Optional[str] = constants.CKPT_UNSEEN
  hidden_dims: int = 128
  random_noise: bool = True
  z_dim: int = 32
  circular_pad: bool = True
  depth_scale: float = constants.DEPTH_SCALE
  gen_dims: int = 128
  image_height: int = 512
  h_fov: float = 0.17
  resnet_version: str = '101'
  use_blurred_mask: bool = True


def get_config() -> SE3DSConfig:
  """Returns the Val-Unseen config for SE3DS."""
  config = SE3DSConfig()
  config.ckpt_path = constants.CKPT_UNSEEN
  config.resnet_version = '101'
  return config


def get_re10k_config() -> SE3DSConfig:
  """Returns the Val-Unseen config for SE3DS."""
  config = SE3DSConfig()
  config.ckpt_path = constants.CKPT_RE10K
  config.resnet_version = '101'
  config.use_blurred_mask = False
  return config


def get_test_config() -> SE3DSConfig:
  """Returns config used for unit tests."""
  config = SE3DSConfig()
  config.ckpt_path = None
  config.hidden_dims = 4
  config.z_dim = 4
  config.gen_dims = 4
  return config
