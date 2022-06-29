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

"""Image models used in SE3DS."""

from typing import List, Optional, Tuple

import gin
from se3ds import constants
from se3ds.models import layers
import tensorflow as tf
from tensorflow_addons import layers as tfa_layers


# Generator models.
@gin.configurable
class ResNetGenerator(tf.keras.Model):
  """ResNet generator model with partial convs."""

  def __init__(self,
               image_size: int = 256,
               gen_dims: int = 96,
               z_dim: int = 128,
               resnet_version: str = '50',
               context_layer: str = 'convs',
               conv_mode: str = 'spectral',
               use_blurred_mask: bool = True):
    """Initializes a ResNet generator."""
    super().__init__()
    self.hidden_dims = gen_dims
    self.resnet_version = resnet_version
    self.z_dim = z_dim
    self.circular_pad = True
    if context_layer not in ['convs', 'none']:
      raise NotImplementedError
    self.context_layer = context_layer
    self.use_blurred_mask = use_blurred_mask

    conv_fn = tf.keras.layers.Conv2D
    if conv_mode == 'spectral':
      conv_fn = layers.SpectralConv

    self.sigmoid = tf.keras.layers.Activation('sigmoid')
    self.encoder = ResNetEncoder(
        image_size=image_size,
        hidden_dims=self.hidden_dims,
        resnet_version=self.resnet_version,
        flatten_output=False,
        circular_pad=self.circular_pad,
        conv_fn=conv_fn)
    self.decoder = ResNetDecoder(
        output_dim=self.hidden_dims,
        image_size=image_size,
        hidden_dims=self.hidden_dims,
        resnet_version=self.resnet_version,
        flatten_output=False,
        circular_pad=self.circular_pad,
        conv_fn=conv_fn)
    self.depth_decoder = ResNetDecoder(
        output_dim=self.hidden_dims,
        image_size=image_size,
        hidden_dims=self.hidden_dims,
        resnet_version=self.resnet_version,
        flatten_output=False,
        circular_pad=self.circular_pad,
        conv_fn=conv_fn)
    # Use separate convs for final RGB and depth outputs.
    self.rgb_conv = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        conv_fn(self.hidden_dims, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        conv_fn(self.hidden_dims, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        conv_fn(3, kernel_size=3, strides=1, padding='VALID'),
    ])
    self.depth_conv = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        conv_fn(self.hidden_dims, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        conv_fn(self.hidden_dims, kernel_size=3, strides=1, padding='VALID'),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.experimental.SyncBatchNormalization(),
        layers.PadLayer(1, circular_pad=self.circular_pad),
        conv_fn(1, kernel_size=3, strides=1, padding='VALID'),
    ])

    if self.context_layer == 'convs':
      self.global_context_layer = tf.keras.Sequential([
          tf.keras.layers.experimental.SyncBatchNormalization(),
          layers.PadLayer(1, circular_pad=self.circular_pad),
          layers.SpectralConv(
              self.hidden_dims * 4, kernel_size=3, strides=1, padding='VALID'),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          layers.PadLayer(1, circular_pad=self.circular_pad),
          layers.SpectralConv(
              self.hidden_dims * 8, kernel_size=3, strides=1, padding='VALID'),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          layers.PadLayer(1, circular_pad=self.circular_pad),
          layers.SpectralConv(
              self.hidden_dims * 4, kernel_size=3, strides=1, padding='VALID'),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          layers.PadLayer(1, circular_pad=self.circular_pad),
          layers.SpectralConv(
              self.hidden_dims * 4, kernel_size=3, strides=1,
              padding='VALID'),
      ])
    else:
      self.global_context_layer = None

  def call(self,
           inputs,
           sample_noise: bool = False,
           training=None) -> List[tf.Tensor]:
    """Performs a forward pass to generate an image.

    Args:
      inputs: List of two inputs. The first item is the condition, the second is
        the random noise (if any).
      sample_noise: If True, samples noise from the prior distribution. If this
        is False, it uses mu directly.
      training: Whether the model is in training mode.

    Returns:
      out_x: Tensor of shape (N, H, W, 3) with values in [0, 1] representing a
        generated image.
    """
    if sample_noise:
      raise ValueError('This model does not support noise sampling!')
    cond, _ = inputs
    guidance_image = cond['proj_image']
    guidance_depth = cond['proj_depth']
    guidance_mask = cond['proj_mask']
    blurred_mask = cond['blurred_mask']

    if self.use_blurred_mask:
      combined_input = tf.concat(
          [guidance_image, guidance_depth, blurred_mask], axis=-1)
    else:
      combined_input = tf.concat(
          [guidance_image, guidance_depth], axis=-1)

    hidden_spatial, skip = self.encoder(combined_input, guidance_mask,
                                        training=training)
    batch_size, hidden_height, hidden_width, _ = hidden_spatial.shape

    if self.global_context_layer is not None:
      hidden_spatial = self.global_context_layer(
          hidden_spatial, training=training)

    kld_loss = tf.zeros((batch_size, hidden_height, hidden_width, self.z_dim))
    mu_p = tf.zeros((batch_size, hidden_height, hidden_width, self.z_dim))
    logvar_p = tf.zeros((batch_size, hidden_height, hidden_width, self.z_dim))

    out = self.decoder(hidden_spatial, skip)
    depth_out = self.depth_decoder(hidden_spatial, skip)

    seg_out = tf.zeros(
        guidance_depth.shape[:-1] + (constants.NUM_MP3D_CLASSES,),
        guidance_depth.dtype)
    depth_seg_out = tf.zeros_like(seg_out)
    rgb_out = self.rgb_conv(out)
    depth_out = self.depth_conv(depth_out)

    # Cast RGB to [0, 1]
    rgb_out = tf.math.tanh(rgb_out)
    rgb_out = (rgb_out + 1) / 2

    depth_out = tf.clip_by_value(depth_out, 0, 1)
    return [
        mu_p, logvar_p, kld_loss, depth_out, seg_out, depth_seg_out, rgb_out
    ]


@gin.configurable
class ResNetEncoder(tf.keras.Model):
  """Encoder architecture for ResNet image model.

  Modified from "RedNet: Residual Encoder-Decoder Network for indoor RGB-D
    Semantic Segmentation": https://arxiv.org/abs/1806.01054"
  """

  def __init__(self,
               image_size: int,
               hidden_dims: int = 64,
               resnet_version: str = '50',  # either 50, 101, or 152
               flatten_output: bool = True,
               circular_pad: bool = False,
               conv_fn: tf.keras.layers.Layer = tf.keras.layers.Conv2D):  # pytype: disable=annotation-type-mismatch  # typed-keras
    super(ResNetEncoder, self).__init__()

    # If model is not fully convolutional, check that image size is valid.
    if flatten_output and image_size not in [128, 256]:
      raise ValueError(f'image_size should be one of {[128, 256]}.')

    self.image_size = image_size
    self.flatten_output = flatten_output

    self.pad1 = layers.PadLayer(3, circular_pad=circular_pad)
    self.conv1 = layers.PartialConv(
        hidden_dims, 7, strides=2, padding='VALID')
    self.act1 = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU()
    ])

    if resnet_version == '50':
      filters = [3, 4, 6, 3]
    elif resnet_version == '101':
      filters = [3, 4, 23, 3]
    elif resnet_version == '152':
      filters = [3, 8, 36, 3]
    else:
      raise ValueError('resnet_version should be one of ["50", "101", "152"], '
                       f'got {resnet_version} instead.')

    self.stack1 = layers.ResStack(
        hidden_dims,
        hidden_dims,
        filters[0],
        circular_pad=circular_pad,
        conv_fn=conv_fn)
    self.stack2 = layers.ResStack(
        hidden_dims,
        hidden_dims * 2,
        filters[1],
        strides=2,
        circular_pad=circular_pad,
        conv_fn=conv_fn)
    self.stack3 = layers.ResStack(
        hidden_dims * 2,
        hidden_dims * 4,
        filters[2],
        strides=2,
        circular_pad=circular_pad,
        conv_fn=conv_fn)
    self.stack4 = layers.ResStack(
        hidden_dims * 4,
        hidden_dims * 8,
        filters[3],
        strides=2,
        circular_pad=circular_pad,
        conv_fn=conv_fn)
    self.flatten = tf.keras.layers.Flatten()
    self.maxpool = tf.keras.Sequential(
        [tf.keras.layers.MaxPool2D(padding='SAME')])
    self.final_pad = layers.PadLayer(1, circular_pad=circular_pad)
    self.final_conv = layers.PartialConv(
        hidden_dims * 4, kernel_size=3, strides=1, padding='VALID')
    self.final_act = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU(),
    ])

  def call(self,
           x: tf.Tensor,
           mask: Optional[tf.Tensor] = None,
           training=None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    update_mask = mask
    if update_mask is not None:
      update_mask = self.pad1(update_mask)

    out_x = self.pad1(x, training=training)

    out_x, update_mask = self.conv1(out_x, update_mask)
    out_x = self.act1(out_x, training=training)
    b1 = out_x
    out_x, update_mask = self.maxpool(out_x), self.maxpool(update_mask)
    out_x, update_mask = self.stack1(out_x, update_mask, training=training)
    s1 = out_x
    out_x, update_mask = self.stack2(out_x, update_mask, training=training)
    s2 = out_x
    out_x, update_mask = self.stack3(out_x, update_mask, training=training)
    s3 = out_x
    out_x, update_mask = self.stack4(out_x, update_mask, training=training)
    out_x = self.final_pad(out_x, training=training)
    update_mask = self.final_pad(update_mask, training=training)
    out_x, update_mask = self.final_conv(out_x, update_mask)
    out_x = self.final_act(out_x, training=training)
    if self.flatten_output:
      out_x = self.flatten(out_x)
    return out_x, [b1, s1, s2, s3]


@gin.configurable
class ResNetDecoder(tf.keras.Model):
  """Decoder architecture for ResNet image model.

  Modified from "RedNet: Residual Encoder-Decoder Network for indoor RGB-D
    Semantic Segmentation": https://arxiv.org/abs/1806.01054"
  """

  def create_agent(self, hidden_dims: int,
                   conv_fn: tf.keras.layers.Layer = tf.keras.layers.Conv2D):  # pytype: disable=annotation-type-mismatch  # typed-keras
    agent = conv_fn(
        hidden_dims,
        kernel_size=1,
        strides=1,
        padding='SAME',
        use_bias=False)
    agent_act = tf.keras.Sequential([
        tf.keras.layers.experimental.SyncBatchNormalization(),
        tf.keras.layers.ReLU()
    ])
    return agent, agent_act

  def __init__(
      self,
      output_dim: int,
      image_size: int,
      hidden_dims: int = 64,
      resnet_version: str = '50',
      flatten_output: bool = True,
      circular_pad: bool = False,
      partial_conv: bool = True,
      conv_fn: tf.keras.layers.Layer = tf.keras.layers.Conv2D):  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Inits a ResNetDecoder.

    Args:
      output_dim: Output dimensionality at the last layer.
      image_size: Input image size in pixels.
      hidden_dims: Base channel dimensions of intermediate hidden layers.
      resnet_version: ResNet architecture to use. Either '50', '101', or '152'.
      flatten_output:  If false, encoder and decoder are fully convolutional
        networks.
      circular_pad: Whether to apply circular padding.
      partial_conv: Whether to use partial convolutions instead of regular ones.
      conv_fn: Conv2D function to use.
    """
    super(ResNetDecoder, self).__init__()

    if flatten_output and image_size not in [128, 256]:
      raise ValueError(f'image_size should be one of {[128, 256]}.')
    self.image_size = image_size
    self.flatten_output = flatten_output
    self.partial_conv = partial_conv
    if self.partial_conv:
      self.agent_fn = layers.PartialConv
      if conv_fn == layers.SpectralConv:
        self.agent_fn = layers.PartialSpectralConv
    else:
      self.agent_fn = conv_fn

    if self.flatten_output:
      self.upc = tf.keras.Sequential([
          tf.keras.layers.Conv2DTranspose(
              hidden_dims * 2, kernel_size=4, strides=1),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          tf.keras.layers.LeakyReLU(alpha=0.2),
          tf.keras.layers.UpSampling2D(),
      ])
    else:
      self.upc = tf.keras.Sequential([
          conv_fn(hidden_dims * 2, kernel_size=1, strides=1, padding='SAME'),
          tf.keras.layers.experimental.SyncBatchNormalization(),
          tf.keras.layers.LeakyReLU(alpha=0.2),
          tf.keras.layers.UpSampling2D(),
      ])

    if resnet_version == '50':
      filters = [6, 4, 3, 3]  # [3, 4, 6, 3]
    elif resnet_version == '101':
      filters = [23, 4, 3, 3]
    elif resnet_version == '152':
      filters = [36, 8, 3, 3]
    else:
      raise ValueError('resnet_version should be one of ["50", "101", "152"], '
                       f'got {resnet_version} instead.')

    if self.flatten_output and self.image_size == 256:
      self.deconv1 = layers.ResStackTranspose(
          hidden_dims * 8,
          hidden_dims * 4,
          filters[0],
          strides=2,
          circular_pad=circular_pad,
          conv_fn=conv_fn)
    else:
      self.deconv1 = layers.ResStackTranspose(
          hidden_dims * 8,
          hidden_dims * 4,
          filters[0],
          strides=1,
          circular_pad=circular_pad,
          conv_fn=conv_fn)
    self.deconv2 = layers.ResStackTranspose(
        hidden_dims * 4,
        hidden_dims * 2,
        filters[1],
        strides=2,
        circular_pad=circular_pad,
        conv_fn=conv_fn)
    self.deconv3 = layers.ResStackTranspose(
        hidden_dims * 2,
        hidden_dims,
        filters[2],
        strides=2,
        circular_pad=circular_pad,
        conv_fn=conv_fn)
    self.deconv4 = layers.ResStackTranspose(
        hidden_dims,
        hidden_dims,
        filters[3],
        strides=2,
        circular_pad=circular_pad,
        conv_fn=conv_fn)

    self.agent0, self.agent0_act = self.create_agent(hidden_dims, self.agent_fn)
    self.agent1, self.agent1_act = self.create_agent(hidden_dims, self.agent_fn)
    self.agent2, self.agent2_act = self.create_agent(hidden_dims * 2,
                                                     self.agent_fn)
    self.agent3, self.agent3_act = self.create_agent(hidden_dims * 4,
                                                     self.agent_fn)
    self.agent4, self.agent4_act = self.create_agent(hidden_dims * 8,
                                                     self.agent_fn)

    self.final_conv = layers.ResStackTranspose(
        hidden_dims, hidden_dims, 3, circular_pad=circular_pad)
    self.final_deconv = tf.keras.layers.Conv2DTranspose(
        output_dim, kernel_size=2, strides=2, padding='SAME')

  def call(self, x: tf.Tensor, skip, masks=None, training=None) -> tf.Tensor:
    if masks is None:
      masks = [None] * len(skip)

    out_x = self.upc(x)
    # TODO: Consider using mask from the lowest level of the encoder.
    if self.partial_conv:
      out_x, _ = self.agent4(out_x)  # (8, 8)
    else:
      out_x = self.agent4(out_x)  # (8, 8)
    out_x = self.agent4_act(out_x)
    out_x = self.deconv1(out_x)
    if self.partial_conv:
      shortcut_out, _ = self.agent3(skip[3], masks[3])
    else:
      shortcut_out = self.agent3(skip[3])
    shortcut_out = self.agent3_act(shortcut_out)
    out_x = out_x + shortcut_out  # (16, 16)

    out_x = self.deconv2(out_x)
    if self.partial_conv:
      shortcut_out, _ = self.agent2(skip[2], masks[2])
    else:
      shortcut_out = self.agent2(skip[2])
    shortcut_out = self.agent2_act(shortcut_out)
    out_x = out_x + shortcut_out  # (32, 32)

    out_x = self.deconv3(out_x)
    if self.partial_conv:
      shortcut_out, _ = self.agent1(skip[1], masks[1])
    else:
      shortcut_out = self.agent1(skip[1])
    shortcut_out = self.agent1_act(shortcut_out)
    out_x = out_x + shortcut_out  # (64, 64)

    out_x = self.deconv4(out_x)
    if self.partial_conv:
      shortcut_out, _ = self.agent0(skip[0], masks[0])
    else:
      shortcut_out = self.agent0(skip[0])
    shortcut_out = self.agent0_act(shortcut_out)
    out_x = out_x + shortcut_out  # (128, 128)

    out_x = self.final_conv(out_x)
    out_x = self.final_deconv(out_x)  # (256, 256)
    return out_x


# Discriminator models.
class SNPatchDiscriminator(tf.keras.Model):
  """Spectral normalized PatchGAN discriminator."""

  def __init__(self,
               kernel_size: int = 4,
               dis_dims: int = 64,
               n_layers: int = 4,
               circular_pad: bool = False):
    """Initializes a spectral normalized PatchGAN discriminator.

    Args:
      kernel_size: Kernel size of convolutions.
      dis_dims: Baseline dimensions for convolutional layers.
      n_layers: Number of layers in this discriminator.
      circular_pad: Whether to apply circular padding.
    """
    super().__init__()

    self.discriminator_groups = [
        tf.keras.Sequential([
            layers.PadLayer(kernel_size // 2, circular_pad=circular_pad),
            tf.keras.layers.Conv2D(
                dis_dims,
                kernel_size=kernel_size,
                strides=2,
                padding='VALID'),
            tf.keras.layers.LeakyReLU(alpha=0.2)
        ])
    ]

    previous_dim = dis_dims
    for i in range(1, n_layers):
      current_dim = min(previous_dim * 2, 512)
      self.discriminator_groups.append(
          tf.keras.Sequential([
              layers.PadLayer(kernel_size // 2, circular_pad=circular_pad),
              layers.SpectralConv(
                  current_dim,
                  kernel_size=kernel_size,
                  strides=2 if (i != n_layers-1) else 1,
                  padding='VALID',
                  activation=None),
              tfa_layers.InstanceNormalization(),
              tf.keras.layers.LeakyReLU(alpha=0.2),
          ])
      )
      previous_dim = current_dim

    # Final classification layer
    self.discriminator_groups.append(
        tf.keras.layers.Conv2D(
            1, kernel_size=kernel_size, strides=1, padding='SAME'))

  def call(self, x: tf.Tensor) -> List[tf.Tensor]:
    """Forward pass of PatchGAN discriminator.

    Args:
      x: Tensor of shape (N, H, W, C).

    Returns:
      results: List of Tensors of intermediate features. Each object in the list
        is of shape (N, H, W, C).
    """
    results = []
    prev_out = x
    for model in self.discriminator_groups:
      out = model(prev_out)
      results.append(out)
      prev_out = out
    return results


@gin.configurable
class SNMultiScaleDiscriminator(tf.keras.Model):
  """Spectral normalized multiscale PatchGAN discriminator."""

  def __init__(self,
               image_size: int = 256,
               n_dis: int = 2,
               kernel_size: int = 4,
               dis_dims: int = 96,
               n_layers: int = 5,
               circular_pad: bool = False):
    """Initializes a spectral normalized PatchGAN discriminator.

    Args:
      image_size: Size of image inputs. This is not used in this model as it is
        fully convolutional, but remains here for compatibility with the
        trainer.
      n_dis: Number of discriminators to use.
      kernel_size: Kernel size of convolutions in sub-discriminators.
      dis_dims: Baseline dims for convolutional layers in sub-discriminators.
      n_layers: Number of layers in each sub-discriminator.
      circular_pad: Whether to apply circular padding.
    """
    super().__init__()
    del image_size  # Not used - this model is fully convolutional.

    self.discriminators = []
    for _ in range(n_dis):
      self.discriminators.append(
          SNPatchDiscriminator(
              kernel_size=kernel_size,
              dis_dims=dis_dims,
              n_layers=n_layers,
              circular_pad=circular_pad))

  def call(self, inputs: tf.Tensor) -> List[List[tf.Tensor]]:
    """Forward pass of multiscale discriminator.

    Args:
      inputs: Tensor of shape (N, H, W, C).

    Returns:
      results: List of list of Tensors of intermediate features. The first
        Tensor in the list has shape (N, H', W', C'), and each Tensor after is
        downsampled by a factor of 2, e.g. (N, H'//2, W'//2, C'), and so on.
    """
    result = []
    prev_out = inputs
    for model in self.discriminators:
      out = model(prev_out)
      result.append(out)

      # Downsample for next discriminator.
      prev_out = tf.nn.avg_pool(prev_out, ksize=3, strides=2, padding='SAME')
    return result
