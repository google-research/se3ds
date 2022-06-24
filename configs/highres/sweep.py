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
"""Sweep module."""


def get_experiment(exp_name, c, h):
  """Returns the parameter exploration for `exp_name`."""
  del exp_name, c
  return h.product(
      [
          h.from_dicts([
              {
                  # U-Net model
                  "GANManager.generator_fn": "@image_models.ResNetGenerator",
              },
          ]),
          h.from_dicts([
              {
                  # 512px panoramas on pf_4x4x8 (train) and pf_2x4x4 (eval)
                  "GANManager.image_size": 512,
                  "R2RImageDataset.image_size": 512,
                  "R2RVideoDataset.image_size": 512,
                  "GANManager.train_batch_size": 128,
                  "GANManager.test_batch_size": 64,
                  "GANManager.eval_every_steps": 4000,
                  "image_models.ResNetGenerator.resnet_version": "101",
                  "image_models.SNMultiScaleDiscriminator.n_dis": 2,
                  "image_models.SNMultiScaleDiscriminator.n_layers": 6,
              },
          ]),
          h.sweep("GANManager.seed", [0]),
      ],
      name="_gin")
