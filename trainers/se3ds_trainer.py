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
"""Trainer for the SE3DS model."""

from typing import Dict

import gin
from se3ds.datasets import indoor_datasets
from se3ds.models import image_models  # pylint: disable=unused-import
from se3ds.trainers import gan_manager
import tensorflow as tf


def _clip_grad(grad, grad_clip_norm: float = 5.0):
  grad = [
      None if g is None else tf.clip_by_norm(g, grad_clip_norm)
      for g in grad
  ]
  return grad


def _kld_loss(mu: tf.Tensor, logvar: tf.Tensor):
  return -0.5 * tf.reduce_sum(1 + logvar - (mu ** 2) - tf.math.exp(logvar))


def _wc_loss(generated_images: tf.Tensor, real_images: tf.Tensor,
             mask: tf.Tensor):
  """Computes world consistency loss.

  Args:
    generated_images: tf.Tensor of shape [N, H, W, 3] with values in [0, 1].
    real_images: tf.Tensor of shape [N, H, W, 3] with values in [0, 1].
    mask: Binary valued tf.Tensor of shape [N, H, W, 1].

  Returns:
    wc_loss: Scalar tensor indicating loss value.
  """
  wc_loss = tf.abs(generated_images - real_images)
  wc_loss = tf.reduce_sum(
      wc_loss * mask, axis=(1, 2, 3)) / generated_images.shape[-1]
  wc_loss = wc_loss / tf.maximum(tf.reduce_sum(mask, axis=(1, 2, 3)), 1)
  return wc_loss


def _discriminator_loss(
    real_logit: tf.Tensor, fake_logit: tf.Tensor) -> tf.Tensor:
  """Calculates hinge loss for the Discriminator."""
  real_loss = tf.nn.relu(1.0 - real_logit)
  fake_loss = tf.nn.relu(1.0 + fake_logit)
  total_loss = real_loss + fake_loss
  return total_loss


def _generator_loss(
    fake_logit: tf.Tensor) -> tf.Tensor:
  """Calculates hinge loss for the Generator."""
  loss = -fake_logit
  return loss


@gin.configurable(denylist=["strategy", "model_dir"])
class GAN(gan_manager.GANManager):
  """One stage GAN."""

  def __init__(self,
               lambda_gan,
               lambda_kld,
               lambda_wc,
               lambda_depth,
               *args,
               dis_use_pred_depth=True,
               mask_blurred=False,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.lambda_gan = lambda_gan
    self.lambda_kld = lambda_kld
    self.lambda_wc = lambda_wc
    self.lambda_depth = lambda_depth
    self.dis_use_pred_depth = dis_use_pred_depth
    self.mask_blurred = mask_blurred

  def _get_dataset(self):
    """Gets the dataset for the model."""
    train_ds = indoor_datasets.R2RImageDataset(image_size=self.image_size)
    test_ds = indoor_datasets.R2RVideoDataset()
    return train_ds, test_ds

  def _create_obj(self):
    super()._create_obj()

    self.ce_loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction="none")

  def _create_metrics(self):
    """Creates metrics for training and evaluation."""

    self.metrics = {}
    self.metrics["gen/gen_loss"] = tf.keras.metrics.Mean(name="gen_loss")
    self.metrics["dis/disc_loss"] = tf.keras.metrics.Mean(name="disc_loss")
    self.metrics["dis/grad_norm"] = tf.keras.metrics.Mean(name="dis_grad_norm")
    self.metrics["gen/gen_feat_loss"] = tf.keras.metrics.Mean(
        name="gen_feat_loss")
    self.metrics["gen/gen_gan_loss"] = tf.keras.metrics.Mean(
        name="gen_gan_loss")
    self.metrics["gen/depth_loss"] = tf.keras.metrics.Mean(name="depth_loss")
    self.metrics["gen/seg_loss"] = tf.keras.metrics.Mean(name="gen_seg_loss")
    self.metrics["gen/depth_seg_loss"] = tf.keras.metrics.Mean(
        name="gen_depth_seg_loss")
    self.metrics["gen/depth_seg_consistency"] = tf.keras.metrics.Mean(
        name="gen_depth_seg_consistency")
    self.metrics["gen/kld_loss"] = tf.keras.metrics.Mean(name="kld_loss")
    self.metrics["gen/kld_nan"] = tf.keras.metrics.Mean(name="kld_nan")
    self.metrics["gen/wc_loss"] = tf.keras.metrics.Mean(name="wc_loss")
    self.metrics["gen/grad_norm"] = tf.keras.metrics.Mean(name="gen_grad_norm")

  def train_g_d(self, inputs: Dict[str, tf.Tensor]) -> None:
    """Learns the Discriminator and Generator updates.

    Args:
      inputs: a mapping contains key-value pair as:
        "image": tf.float32 tensor of shape (B, H, W, C), normalized to have
          values in [0, 1].
        "embedding": tf.float32 tensor of shape (B, MAX_LEN, D) for word
          embedding.
        "max_len": tf.int32 tensor of shape (B, 1) for sentence length.
        "sentence_embedding": tf.float32 tensor of shape (B, D) for the sentence
          embedding.
    """
    image = inputs["image"]
    if not self.mask_blurred:
      inputs["blurred_mask"] = tf.zeros_like(inputs["blurred_mask"])
    blurred_mask = inputs["blurred_mask"]  # binary (N, H, W, 1)

    # (N, H, W, 1)
    target_spatial_mask = tf.cast(
        tf.math.logical_and(
            inputs["depth"] > 0, inputs["depth"] < 1), tf.float32)
    num_spatial_pixels = tf.reduce_sum(target_spatial_mask, axis=(1, 2, 3))
    num_spatial_pixels = tf.maximum(num_spatial_pixels, 1)

    combined_input = inputs["depth"]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      (_, _, kld_loss, depth_out, seg_logits, depth_seg_logits,
       generated) = self.generator(
           inputs=[inputs, None], training=True)

      depth_loss = 0.
      if self.predict_depth:
        depth_loss = tf.abs(depth_out - inputs["depth"]) * target_spatial_mask
        depth_loss = tf.reduce_sum(
            depth_loss, axis=(1, 2, 3)) / num_spatial_pixels
        depth_loss = self.lambda_depth * tf.reduce_mean(depth_loss)

      seg_loss = 0.
      depth_seg_loss = 0.
      depth_seg_consistency_loss = 0.

      kld_nan = tf.reduce_sum(tf.cast(tf.math.is_nan(kld_loss), tf.float32))
      kld_loss = tf.where(
          tf.math.is_nan(kld_loss), tf.zeros_like(kld_loss), kld_loss)
      kld_loss = self.lambda_kld * tf.reduce_mean(kld_loss)
      wc_loss = self.lambda_wc * _wc_loss(
          generated, inputs["proj_image"], inputs["proj_mask"] *
          (1 - blurred_mask))

      # Input to discimrinator is concatenation of image and condition.
      if self.dis_use_pred_depth:
        fake_input = tf.concat([generated, depth_out], axis=-1)
      else:
        fake_input = tf.concat([generated, combined_input], axis=-1)
      real_input = tf.concat([image, combined_input], axis=-1)
      all_input = tf.concat([fake_input, real_input], axis=0)

      # The output is a list of list of Tensors.
      # The first list contains the outputs of each sub-discriminator in the
      # multiscale discriminator. Each list within that first list contains a
      # list of feature maps for the corresponding sub-discriminator.
      logit_outputs = self.discriminator(all_input, training=True)

      fake_logit_list, real_logit_list = [], []
      for sub_logit_list in logit_outputs:
        assert isinstance(
            sub_logit_list,
            list), "Discriminator is expected to output a list of lists."
        sub_fake_logit_list = []
        sub_real_logit_list = []
        for out in sub_logit_list:
          fake_logit, real_logit = tf.split(out, 2)
          sub_fake_logit_list.append(fake_logit)
          sub_real_logit_list.append(real_logit)
        fake_logit_list.append(sub_fake_logit_list)
        real_logit_list.append(sub_real_logit_list)

      gen_loss = 0
      disc_loss = 0

      for sub_fake_logit, sub_real_logit in zip(fake_logit_list,
                                                real_logit_list):
        # Each sub_fake_logit and sub_real_logit is a list of intermediate
        # feature maps. The last feature map is the final real / fake decision.
        # sub_fake_logit[-1].shape: (4, 10, 18, 1)
        # sub_real_logit[-1].shape: (4, 10, 18, 1)
        curr_gen_loss = _generator_loss(sub_fake_logit[-1])
        curr_gen_loss = tf.reduce_mean(curr_gen_loss)
        curr_disc_loss = _discriminator_loss(sub_real_logit[-1],
                                             sub_fake_logit[-1])
        curr_disc_loss = tf.reduce_mean(curr_disc_loss)
        gen_loss += curr_gen_loss
        disc_loss += curr_disc_loss

      # Normalize by number of intermediate feature maps.
      num_feature_maps = len(fake_logit_list)
      gen_loss = self.lambda_gan * gen_loss / num_feature_maps
      disc_loss = self.lambda_gan * disc_loss / num_feature_maps

      scaled_disc_loss = disc_loss / self.strategy.num_replicas_in_sync
      combined_gen_loss = (
          gen_loss + kld_loss + wc_loss +
          depth_loss + seg_loss + depth_seg_loss + depth_seg_consistency_loss)
      scaled_gen_loss = combined_gen_loss / self.strategy.num_replicas_in_sync

    gradients_of_gen = gen_tape.gradient(
        scaled_gen_loss, self.generator.trainable_variables)
    gradients_of_gen = _clip_grad(gradients_of_gen)
    gen_grad_norms = tf.reduce_mean([tf.norm(g) for g in gradients_of_gen])
    gen_grad_norms = tf.where(
        tf.math.is_nan(gen_grad_norms), tf.zeros_like(gen_grad_norms),
        gen_grad_norms)

    gradients_of_discriminator = disc_tape.gradient(
        scaled_disc_loss, self.discriminator.trainable_variables)
    gradients_of_discriminator = _clip_grad(gradients_of_discriminator)
    dis_grad_norms = tf.reduce_mean(
        [tf.norm(g) for g in gradients_of_discriminator])
    dis_grad_norms = tf.where(
        tf.math.is_nan(dis_grad_norms), tf.zeros_like(dis_grad_norms),
        dis_grad_norms)

    self.g_optimizer.apply_gradients(
        zip(gradients_of_gen, self.generator.trainable_variables))
    self.d_optimizer.apply_gradients(
        zip(gradients_of_discriminator, self.discriminator.trainable_variables)
    )
    if self.global_step == 0:
      self.ema_generator(inputs=[inputs, None], training=True)
    self.update_ema_model()
    self.metrics["dis/disc_loss"].update_state(disc_loss)
    self.metrics["dis/grad_norm"].update_state(dis_grad_norms)
    self.metrics["gen/gen_gan_loss"].update_state(gen_loss)
    self.metrics["gen/gen_loss"].update_state(combined_gen_loss)
    self.metrics["gen/depth_loss"].update_state(depth_loss)
    self.metrics["gen/seg_loss"].update_state(seg_loss)
    self.metrics["gen/depth_seg_loss"].update_state(depth_seg_loss)
    self.metrics["gen/depth_seg_consistency"].update_state(
        depth_seg_consistency_loss)
    self.metrics["gen/kld_loss"].update_state(kld_loss)
    self.metrics["gen/kld_nan"].update_state(kld_nan)
    self.metrics["gen/wc_loss"].update_state(wc_loss)
    self.metrics["gen/grad_norm"].update_state(gen_grad_norms)

  def train_d(self, inputs: Dict[str, tf.Tensor]) -> None:
    """Learns the Discriminator updates.

    Args:
      inputs: a mapping contains key-value pair as:
        "image": tf.float32 tensor of shape (B, H, W, C), normalized to have
          values in [0, 1].
        "embedding": tf.float32 tensor of shape (B, MAX_LEN, D) for word
          embedding.
        "max_len": tf.int32 tensor of shape (B, 1) for sentence length.
        "sentence_embedding": tf.float32 tensor of shape (B, D) for the sentence
          embedding.
    """
    image = inputs["image"]
    if not self.mask_blurred:
      inputs["blurred_mask"] = tf.zeros_like(inputs["blurred_mask"])
    combined_input = inputs["depth"]
    _, _, _, depth_out, _, _, generated = self.generator(
        inputs=[inputs, None], training=True)

    with tf.GradientTape() as disc_tape:
      # Input to discriminator is concatenation of image and condition.
      if self.dis_use_pred_depth:
        fake_input = tf.concat([generated, depth_out], axis=-1)
      else:
        fake_input = tf.concat([generated, combined_input], axis=-1)
      real_input = tf.concat([image, combined_input], axis=-1)
      all_input = tf.concat([fake_input, real_input], axis=0)
      logit_outputs = self.discriminator(all_input, training=True)

      fake_logit_list, real_logit_list = [], []
      for sub_logit_list in logit_outputs:
        assert isinstance(
            sub_logit_list,
            list), "Discriminator is expected to output a list of lists."
        sub_fake_logit_list = []
        sub_real_logit_list = []
        for out in sub_logit_list:
          fake_logit, real_logit = tf.split(out, 2)
          sub_fake_logit_list.append(fake_logit)
          sub_real_logit_list.append(real_logit)
        fake_logit_list.append(sub_fake_logit_list)
        real_logit_list.append(sub_real_logit_list)

      disc_loss = 0

      for sub_fake_logit, sub_real_logit in zip(fake_logit_list,
                                                real_logit_list):
        curr_disc_loss = _discriminator_loss(sub_real_logit[-1],
                                             sub_fake_logit[-1])
        curr_disc_loss = tf.reduce_mean(curr_disc_loss)
        disc_loss += curr_disc_loss

      # Normalize by number of intermediate feature maps.
      num_feature_maps = len(fake_logit_list)
      disc_loss = self.lambda_gan * disc_loss / num_feature_maps
      disc_loss = disc_loss / self.strategy.num_replicas_in_sync

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, self.discriminator.trainable_variables)
    gradients_of_discriminator = _clip_grad(gradients_of_discriminator)
    self.d_optimizer.apply_gradients(
        zip(gradients_of_discriminator, self.discriminator.trainable_variables)
    )
