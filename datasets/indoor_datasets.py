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

"""R2R image and video datasets preprocessed for SE3DS training."""

import enum
import os
from typing import Optional

from absl import logging
import gin
from se3ds import constants
from se3ds.datasets import base_dataset
import tensorflow as tf


class DatasetType(enum.Enum):
  MP3D = 0
  GIBSON = 1  # Unused
  RE10K = 2


def augment(
    x: tf.Tensor,
    random_roll_range: Optional[int] = None,
    random_flip: bool = True,
    seed: Optional[int] = None,
) -> tf.Tensor:
  """Randomly augments the input image.

  Args:
    x: Input image tensor of shape (N, H, W, C) with values in [0, 1].
    random_roll_range: Maximum number of pixels by which to randomly roll image.
      If not provided, rolls up to the entire width of the image.
    random_flip: Whether to random flip the image.
    seed: Random seed for augmenation.
  Returns:
    The augmented image tensor.
  """
  if len(x.shape) != 4:
    raise ValueError(
        f"Expected input to have shape of length 4, instead got {len(x.shape)}")
  random_roll_range = random_roll_range or (x.shape[2] // 2)
  roll_amount = tf.random.uniform((), -random_roll_range, random_roll_range,
                                  dtype=tf.int32, seed=seed)
  x = tf.roll(x, roll_amount, axis=2)
  if random_flip:
    x = tf.image.random_flip_left_right(
        x, seed=seed)
  return x


@gin.configurable
class R2RImageDataset(base_dataset.BaseDataset):
  """Dataset class for preprocessing data for training Pathdreamer models.

  Attributes:
    image_size: Height of images to be output.
    preprocessed_image_height: Height of images in the preprocessed TFRecords.
    z_dim: Dimension of noise vector.
    num_classes: Number of semantic class labels in segmentation outputs.
    data_dir: Directory to load dataset TFRecords from.
    return_filename: If True, returns the filename and scan_id of examples as
      strings. Should be set to False if running on TPU.
    horizontal_mask_ratio: Ratio to randomly mask outputs horizontally, to act
      as data augmentation to simulate sparse inputs. A value of 0.1 for example
      indicates that up to 10% of the image may be set to zeros during training.
    vertical_mask_ratio: Ratio to randomly mask outputs vertically, similar to
      horizontal_mask_ratio.
    random_roll_and_flip: If True, applies random horizontal roll and flip on
      the output examples.
    random_crop: If True, randomly crops the output image to (image_size,
      image_size * 2) panoramas.
    random_resize_max: Maximum value to upsample panoramas by before cropping.
      Only used if random_crop is true, otherwise the panorama is not resized.
    pad_minval: Minimum amount of padding around the visible mask. Value can
      range in [-1,1].
    pad_maxval: Maximum amount of padding around the visible mask. Value can
      range in [-1,1].
  """

  def __init__(
      self,
      image_size: int = 256,
      preprocessed_image_height: int = 512,
      z_dim: int = 64,
      num_classes: int = constants.NUM_MP3D_CLASSES,
      data_dir: str = "data/train/",
      return_filename: bool = False,
      horizontal_mask_ratio: float = 0.5,
      vertical_mask_ratio: float = 0.5,
      random_roll_and_flip: bool = True,
      random_crop: bool = True,
      random_resize_max: float = 2.0,
      pad_minval: float = -0.05,
      pad_maxval: float = 0.1,
      re_10k_crop: bool = False,
      **kwargs,
  ):
    super().__init__(
        image_size=image_size, z_dim=z_dim, num_classes=num_classes, **kwargs)
    self.data_dir = data_dir
    self.return_filename = return_filename
    self.preprocessed_image_height = preprocessed_image_height
    self.horizontal_mask_ratio = horizontal_mask_ratio
    self.vertical_mask_ratio = vertical_mask_ratio
    self.random_roll_and_flip = random_roll_and_flip
    self.random_crop = random_crop
    self.random_resize_max = random_resize_max
    self.pad_minval = pad_minval
    self.pad_maxval = pad_maxval
    self.re_10k_crop = re_10k_crop

  def _parse(self, example):
    """Returns a parsed, preprocessed, and batched `tf.data.Dataset`.

    Args:
      example: Scalar string tensor representing bytes data.

    Returns:
      outputs: Dict of string feature names to Tensor feature values:
        "image": RGB image tf.float32 tensor of shape (H, W, 3), normalized to
          [0, 1].
        "prev_image": RGB image tf.float32 tensor of shape (H, W, 3),
          normalized to [0, 1]. This represents the previous frame.
        "proj_image": RGB image tf.float32 tensor of shape (H, W, 3),
          normalized to [0, 1]. This represents the guidance RGB image.
        "proj_mask": tf.float32 binary tensor of shape (H, W) with 1 indicating
          valid pixels and 0 indicating masked out pixels.
        "segmentation": tf.int32 tensor of shape (H, W) containing semantic
          segmentation labels assigned to each pixel.
        "depth": tf.float32 tensor of shape (H, W) containing depth values in
          [0, 1] assigned to each pixel.
        "scan_id": tf.string tensor indicating Matterport3D scan ID.
        "filename": tf.string tensor indicating image filename.
    """

    features = {
        "scan_id":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "dataset_type":
            tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "depth_scale":
            tf.io.FixedLenFeature([], dtype=tf.float32, default_value=10.0),
        "image/encoded":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/perspective_encoded":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/filename":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/depth":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/perspective_depth":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/visible_mask":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/blurred_mask":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "image/segmentation/class/encoded":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "proj/encoded":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "proj/depth":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "proj/mask":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "bbox":
            tf.io.FixedLenFeature([4],
                                  dtype=tf.float32,
                                  default_value=[0.0, 0.0, 0.0, 0.0])
    }

    preprocessed_shape = (self.preprocessed_image_height,
                          self.preprocessed_image_height * 2)
    decoded_example = tf.io.parse_single_example(example, features)
    output = {}
    output["dataset_type"] = decoded_example["dataset_type"]
    output["image"] = tf.image.decode_png(
        decoded_example["image/encoded"], channels=3)
    output["image"].set_shape(preprocessed_shape + (3,))
    output["image"] = tf.image.convert_image_dtype(output["image"], tf.float32)

    output["proj_image"] = tf.image.decode_png(
        decoded_example["proj/encoded"], channels=3)
    output["proj_image"].set_shape(preprocessed_shape + (3,))
    output["proj_image"] = tf.image.convert_image_dtype(output["proj_image"],
                                                        tf.float32)

    output["proj_depth"] = tf.image.decode_png(
        decoded_example["proj/depth"], channels=1, dtype=tf.uint16)[:, :, 0]
    output["proj_depth"] = tf.image.convert_image_dtype(output["proj_depth"],
                                                        tf.float32)
    output["proj_depth"].set_shape(preprocessed_shape)

    output["proj_mask"] = tf.image.decode_png(
        decoded_example["proj/mask"], channels=1)[:, :, 0]
    output["proj_mask"] = tf.cast(
        tf.clip_by_value(output["proj_mask"], 0, 1), tf.float32)
    output["proj_mask"].set_shape(preprocessed_shape)

    # Create binary tensor with 1s where blurred values are.
    output["blurred_mask"] = tf.image.decode_png(
        decoded_example["image/blurred_mask"], channels=1)
    output["blurred_mask"] = tf.cast(
        tf.clip_by_value(output["blurred_mask"], 0, 1), tf.float32)
    output["blurred_mask"].set_shape(preprocessed_shape + (1,))

    output["segmentation"] = tf.image.decode_png(
        decoded_example["image/segmentation/class/encoded"], channels=1)[:, :,
                                                                         0]
    output["segmentation"] = tf.cast(output["segmentation"], tf.int32)
    output["segmentation"].set_shape(preprocessed_shape)

    # Scalar that is 1 if segmentation is valid and 0 otherwise.
    output["segmentation_valid"] = tf.cast(
        tf.reduce_any(output["segmentation"] != 0), tf.float32)

    output["depth"] = tf.image.decode_png(
        decoded_example["image/depth"], channels=1, dtype=tf.uint16)[:, :, 0]
    output["depth"] = tf.image.convert_image_dtype(output["depth"], tf.float32)
    output["depth"].set_shape(preprocessed_shape)

    output["depth_scale"] = decoded_example["depth_scale"]

    if output["dataset_type"] == DatasetType.RE10K.value:
      output["visible_mask"] = tf.image.decode_png(
          decoded_example["image/visible_mask"], channels=1)
      output["visible_mask"] = tf.cast(
          tf.clip_by_value(output["visible_mask"], 0, 1), tf.float32)
      output["visible_mask"].set_shape(preprocessed_shape + (1,))
      output["blurred_mask"] = 1 - output["visible_mask"]
    else:
      output["visible_mask"] = tf.zeros(preprocessed_shape + (1,), tf.float32)

    output["bbox"] = decoded_example["bbox"]

    if self.return_filename:
      output["filename"] = decoded_example["image/filename"]
      output["scan_id"] = decoded_example["scan_id"]
    return output

  def get_file_patterns(self,
                        split: Optional[str] = None,
                        file_pattern: Optional[str] = None):
    """Pass the file pattern to find corresponding files for given split."""

    if not file_pattern:
      if split not in ("train", "val", "val_unseen", "val_seen", "test"):
        if split in ["val_seen", "val_unseen"]:
          split = "val"
        raise ValueError(
            f"Expected split to be one of ['train', 'val'], got {split}")
      file_pattern = os.path.join(self.data_dir, f"{split}*.tfrecord")
    return file_pattern

  def _transform_fn(self, features, seed, random_generator):
    image = features["image"]
    proj_image = features["proj_image"]
    segmentation = tf.cast(features["segmentation"][..., None], tf.float32)
    depth = features["depth"][..., None]
    proj_depth = features["proj_depth"][..., None]
    proj_mask = features["proj_mask"][..., None]
    blurred_mask = features["blurred_mask"]
    height, width, _ = proj_mask.shape

    resize_size = (self.image_size, self.image_size * 2)
    # Randomly perturb resize size.
    if self.random_crop:
      random_multiplier = tf.random.uniform((), 1.0, self.random_resize_max)
      resize_size = (int(self.image_size * random_multiplier),
                     int(self.image_size * 2 * random_multiplier))

    # Perform random masking.
    if self.horizontal_mask_ratio > 0:
      mask_ratio = tf.random.uniform((), 0, self.horizontal_mask_ratio)
      keep_ratio = 1 - mask_ratio
      image_start = tf.random.uniform((), 0, width)
      image_end = (image_start + width * keep_ratio) % width
      if image_start > image_end:
        mask = tf.math.logical_or(
            tf.range(width, dtype=tf.float32) > image_start,
            tf.range(width, dtype=tf.float32) < image_end)[None, :, None]
      else:
        mask = tf.math.logical_and(
            tf.range(width, dtype=tf.float32) > image_start,
            tf.range(width, dtype=tf.float32) < image_end)[None, :, None]
      proj_mask = proj_mask * tf.cast(mask, proj_mask.dtype)
    if self.vertical_mask_ratio > 0:
      mask_ratio = tf.random.uniform((), 0, self.vertical_mask_ratio)
      image_height = height * (1 - mask_ratio)
      image_start = tf.random.uniform((), 0, height - image_height)
      mask = tf.math.logical_and(
          tf.range(height, dtype=tf.float32) > image_start,
          tf.range(height, dtype=tf.float32) < image_start + image_height)[:,
                                                                           None,
                                                                           None]
      proj_mask = proj_mask * tf.cast(mask, proj_mask.dtype)

    # Batch images and semantics to speed up preprocessing.
    images = image
    semantics = tf.concat(
        [segmentation, depth, proj_depth, proj_mask, blurred_mask, proj_image],
        axis=-1)

    images = tf.image.resize(images, resize_size)
    images = tf.clip_by_value(images, 0.0, 1.0)
    semantics = tf.image.resize(semantics, resize_size, method="nearest")

    if self.random_crop or self.random_roll_and_flip:
      aug_features = tf.concat([images, semantics], axis=-1)

      # Perform random horizontal roll and flips.
      if self.random_roll_and_flip:
        random_roll_range = int(
            float(self.image_size) * 2 * self.random_resize_max)
        aug_features = augment(aug_features[None, ...],
                               random_roll_range)[0, ...]

      if self.random_crop:
        _, _, channels = aug_features.shape
        crop_size = (self.image_size, self.image_size * 2, channels)
        cropped = tf.image.random_crop(aug_features, size=crop_size)
        aug_features = tf.cast(cropped, aug_features.dtype)

      images, semantics = tf.split(
          aug_features,
          [images.shape[-1], semantics.shape[-1]],
          axis=-1)

    image = images
    (segmentation, depth, proj_depth, proj_mask, blurred_mask,
     proj_image) = tf.split(
         semantics, [
             segmentation.shape[-1],
             depth.shape[-1],
             proj_depth.shape[-1],
             proj_mask.shape[-1],
             blurred_mask.shape[-1],
             proj_image.shape[-1],
         ],
         axis=-1)

    output = dict(
        image=image,
        proj_image=proj_image,
        proj_mask=proj_mask,
        proj_depth=proj_depth,
        segmentation=tf.cast(segmentation, tf.int32),
        segmentation_valid=features["segmentation_valid"],
        depth=depth,
        depth_scale=features["depth_scale"],
        blurred_mask=blurred_mask,
        dataset_type=features["dataset_type"],
        bbox=features["bbox"]
    )

    if self.return_filename:
      output.update({"filename": features["filename"]})
    if self.z_generator == "cpu_generator":
      z = random_generator.normal(shape=(self.z_dim,))
      output.update({"z": z})
    elif self.z_generator == "cpu_random":
      z = tf.random.normal((self.z_dim,))
      output.update({"z": z})
      logging.info("Random z is generated by CPU tf.random.")
    else:
      logging.info("Random z is generated by the device (TPU/GPU).")
    return output

  def _transform_fn_re10k(self, features, seed, random_generator):
    image = features["image"]
    height, width, _ = image.shape

    proj_image = features["proj_image"]
    segmentation = tf.cast(features["segmentation"][..., None], tf.float32)
    depth = features["depth"][..., None]
    proj_depth = features["proj_depth"][..., None]
    proj_mask = features["proj_mask"][..., None]
    blurred_mask = features["blurred_mask"]

    # Perform random masking.
    if self.horizontal_mask_ratio > 0:
      mask_ratio = tf.random.uniform((), 0, self.horizontal_mask_ratio)
      keep_ratio = 1 - mask_ratio
      image_start = tf.random.uniform((), 0, width)
      image_end = (image_start + width * keep_ratio) % width
      if image_start > image_end:
        mask = tf.math.logical_or(
            tf.range(width, dtype=tf.float32) > image_start,
            tf.range(width, dtype=tf.float32) < image_end)[None, :, None]
      else:
        mask = tf.math.logical_and(
            tf.range(width, dtype=tf.float32) > image_start,
            tf.range(width, dtype=tf.float32) < image_end)[None, :, None]
      proj_mask = proj_mask * tf.cast(mask, proj_mask.dtype)
    if self.vertical_mask_ratio > 0:
      mask_ratio = tf.random.uniform((), 0, self.vertical_mask_ratio)
      image_height = height * (1 - mask_ratio)
      image_start = tf.random.uniform((), 0, height - image_height)
      mask = tf.math.logical_and(
          tf.range(height, dtype=tf.float32) > image_start,
          tf.range(height, dtype=tf.float32) < image_start + image_height)[:,
                                                                           None,
                                                                           None]
      proj_mask = proj_mask * tf.cast(mask, proj_mask.dtype)

    # Batch images and semantics to speed up preprocessing.
    images = image
    semantics = tf.concat(
        [segmentation, depth, proj_depth, proj_mask, blurred_mask], axis=-1)

    if self.re_10k_crop:
      rows = tf.math.count_nonzero(
          1 - blurred_mask[..., 0], axis=0, keepdims=None,
          dtype=tf.bool)  # return true if any pixels in the given row is true
      columns = tf.math.count_nonzero(
          1 - blurred_mask[..., 0], axis=1, keepdims=None, dtype=tf.bool)

      def indices_by_value(value):
        return tf.cast(
            tf.where(tf.equal(value, True))[:, -1], tf.float32
        )  # return all the indices where mask is present along given axis

      # pad_prcntg -- a value in [0,1] indicates how much padding to add to the
      # crop region.
      pad_prcntg = tf.random.uniform(
          shape=[], minval=self.pad_minval, maxval=self.pad_maxval)

      x_shift = tf.random.uniform(
          shape=[],
          minval=-0.5 * tf.abs(pad_prcntg),
          maxval=0.5 * tf.abs(pad_prcntg))
      y_shift = tf.random.uniform(
          shape=[],
          minval=-0.5 * tf.abs(pad_prcntg),
          maxval=0.5 * tf.abs(pad_prcntg))

      y_min = indices_by_value(columns)[
          0] / height - pad_prcntg + y_shift  # first true pixel along axis
      y_max = indices_by_value(columns)[
          -1] / height + pad_prcntg + y_shift  # last true pixel along axis
      x_min = indices_by_value(rows)[0] / width
      x_max = indices_by_value(rows)[-1] / width

      # normalized coordinates have aspect ratio built in.
      new_h = y_max - y_min
      pad_w = (new_h - (x_max - x_min)) / 2
      x_max = x_max + pad_w + x_shift
      x_min = x_min - pad_w + x_shift

      y_min = tf.cast(y_min * height, tf.int32)
      x_min = tf.cast(x_min * width, tf.int32)
      y_max = tf.cast((y_max) * height, tf.int32)
      x_max = tf.cast((x_max) * width, tf.int32)

      # if bbox outside image.
      y_min = tf.math.maximum(0, y_min)
      x_min = tf.math.maximum(0, x_min)
      y_max = tf.math.minimum(y_max, height)
      x_max = tf.math.minimum(x_max, width)

      # if y_max == y_min, fix.
      # This could be due to weird issues like blank image.
      y_max = tf.math.maximum(y_min + 1, y_max)
      x_max = tf.math.maximum(x_min + 1, x_max)

      if self.random_crop:
        aug_features = tf.concat([images, semantics, proj_image], axis=-1)

        cropped = tf.image.crop_to_bounding_box(aug_features, y_min, x_min,
                                                y_max - y_min, x_max - x_min)
        aug_features = tf.cast(cropped, aug_features.dtype)

        images, semantics, proj_image = tf.split(
            aug_features,
            [images.shape[-1], semantics.shape[-1], proj_image.shape[-1]],
            axis=-1)

        resize_size = (int(self.image_size), int(self.image_size * 2))

        images = tf.image.resize(images, resize_size)
        images = tf.clip_by_value(images, 0.0, 1.0)
        semantics = tf.image.resize(semantics, resize_size, method="nearest")
        proj_image = tf.image.resize(proj_image, resize_size, method="nearest")

    image = images
    (segmentation, depth, proj_depth, proj_mask, blurred_mask) = tf.split(
        semantics, [
            segmentation.shape[-1],
            depth.shape[-1],
            proj_depth.shape[-1],
            proj_mask.shape[-1],
            blurred_mask.shape[-1],
        ],
        axis=-1)

    output = dict(
        image=image,
        proj_image=proj_image,
        proj_mask=proj_mask,
        proj_depth=proj_depth,
        segmentation=tf.cast(segmentation, tf.int32),
        segmentation_valid=features["segmentation_valid"],
        depth=depth,
        depth_scale=features["depth_scale"],
        blurred_mask=blurred_mask,
        dataset_type=features["dataset_type"],
        bbox=features["bbox"]
    )

    # copy the old bbox information if no random crop.
    if self.re_10k_crop and self.random_crop:
      output["bbox"] = [x_min, y_min, x_max, y_max]

    if self.return_filename:
      output.update({
          "filename": features["filename"],
      })
    if self.z_generator == "cpu_generator":
      z = random_generator.normal(shape=(self.z_dim,))
      output.update({"z": z})
    elif self.z_generator == "cpu_random":
      z = tf.random.normal((self.z_dim,))
      output.update({"z": z})
      logging.info("Random z is generated by CPU tf.random.")
    else:
      logging.info("Random z is generated by the device (TPU/GPU).")
    return output

  def _train_transform_fn(self, features, seed, random_generator):
    # if features["dataset_type"] == DatasetType.RE10K.value:
    #   return self._transform_fn_re10k(features, seed, random_generator)
    # else:
    return self._transform_fn(features, seed, random_generator)

  def _eval_transform_fn(self, features, seed, random_generator):
    # if features["dataset_type"] == DatasetType.RE10K.value:
    #   features = self._transform_fn_re10k(features, seed, random_generator)
    # else:
    features = self._transform_fn(features, seed, random_generator)

    features["one_hot_mask"] = tf.one_hot(
        tf.cast(features["segmentation"][..., 0], tf.int32), self.num_classes)
    return features

  def _train_batch_transform_fn(self, features, seed):
    """Implements batch transform for faster speed.

    https://www.tensorflow.org/guide/data_performance#vectorizing_mapping.

    Args:
      features: The feature dict.
      seed: The random seed used for randomness.

    Returns:
      The transformed feature dict.
    """
    image = features["image"]
    segmentation = features["segmentation"]
    depth = features["depth"]
    proj_image = features["proj_image"]
    proj_depth = features["proj_depth"]
    proj_mask = features["proj_mask"]
    blurred_mask = features["blurred_mask"]

    images = tf.concat([image, proj_image], axis=0)
    image, proj_image = tf.split(
        images, [image.shape[0], proj_image.shape[0]], axis=0)

    features.update({
        "image":
            image,  # TODO: Revisit if we need to *= (1- blurred-mask)
        "proj_image":
            proj_image * proj_mask,
        "proj_mask":
            proj_mask,
        "proj_depth":
            proj_depth * proj_mask,
        "depth":
            depth,
        "segmentation":
            tf.cast(segmentation, tf.int32),
        "one_hot_mask":
            tf.one_hot(
                tf.cast(segmentation[..., 0], tf.int32), self.num_classes),
        "blurred_mask":
            blurred_mask,
    })

    return features

  @property
  def num_examples(self):
    return {"train": 183_621, "val": 4671}


@gin.configurable
class R2RVideoDataset(base_dataset.BaseDataset):
  """Dataset class for preprocessing data for R2R."""

  def __init__(
      self,
      image_size: int = 256,
      preprocessed_image_height: int = 512,
      num_classes: int = constants.NUM_MP3D_CLASSES,
      data_dir: str = "data/val/",
      return_filename: bool = False,
      video_length: int = constants.PANO_VIDEO_LENGTH,
      horizontal_mask_ratio: float = 0.0,
      **kwargs,
  ):
    super().__init__(image_size=image_size, num_classes=num_classes, **kwargs)
    self.preprocessed_image_height = preprocessed_image_height
    self.data_dir = data_dir
    self.return_filename = return_filename
    self.video_length = video_length
    self.horizontal_mask_ratio = horizontal_mask_ratio

  def _parse(self, example):
    """Returns a parsed, preprocessed, and batched `tf.data.Dataset`.

    Args:
      example: Scalar string tensor representing bytes data.

    Returns:
      outputs: Dict of string feature names to Tensor feature values:
        "image": tf.float32 tensor of shape (T, H, W, C), normalized to
          have values in [0, 1]. This represents the RGB image sequence.
        "position": tf.string tensor indicating image filename.
        "mask": tf.float32 binary tensor of shape (T,) indicating whether a
          frame is valid or padding.
        "segmentation": tf.int32 tensor of shape (T, H, W) containing semantic
          segmentation labels assigned to each pixel. Values span
          [0, self.num_classes].
        "depth": tf.float32 tensor of shape (T, H, W) containing depth values in
          [0, 1] for the sequence.
        "scan_id": tf.string tensor indicating Matterport3D scan ID.
        "id": tf.string tensor indicating image ID.
    """

    features = {
        "id":
            tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "scan_id":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "dataset_type":
            tf.io.FixedLenFeature([], dtype=tf.int64, default_value=0),
        "depth_scale":
            tf.io.FixedLenFeature([],
                                  dtype=tf.float32,
                                  default_value=constants.DEPTH_SCALE),
        "video/num_frames":
            tf.io.FixedLenFeature([], dtype=tf.int64),
        "video/rgb":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "video/segmentations":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "video/depth":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "video/position":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
        "video/mask":
            tf.io.FixedLenFeature([], dtype=tf.string, default_value=""),
    }

    preprocessed_shape = (constants.PANO_VIDEO_LENGTH,
                          self.preprocessed_image_height,
                          self.preprocessed_image_height * 2)
    decoded_example = tf.io.parse_single_example(example, features)
    output = {}
    output["id"] = decoded_example["id"]
    output["dataset_type"] = decoded_example["dataset_type"]
    output["image"] = tf.io.parse_tensor(
        decoded_example["video/rgb"], out_type=tf.float32)
    output["image"].set_shape(preprocessed_shape + (3,))

    output["position"] = tf.io.parse_tensor(
        decoded_example["video/position"], out_type=tf.float32)
    output["position"] = tf.ensure_shape(
        output["position"], [constants.PANO_VIDEO_LENGTH, 4])

    output["mask"] = tf.io.parse_tensor(
        decoded_example["video/mask"], out_type=tf.float32)
    output["mask"] = tf.ensure_shape(output["mask"],
                                     [constants.PANO_VIDEO_LENGTH])

    output["segmentation"] = tf.io.parse_tensor(
        decoded_example["video/segmentations"], out_type=tf.uint8)
    output["segmentation"] = tf.ensure_shape(output["segmentation"],
                                             preprocessed_shape)

    output["pathdreamer_segmentation"] = tf.io.parse_tensor(
        decoded_example["video/pathdreamer_segmentations"], out_type=tf.int32)
    output["pathdreamer_segmentation"] = tf.ensure_shape(
        output["pathdreamer_segmentation"], preprocessed_shape)
    output["pathdreamer_segmentation"] = tf.cast(
        output["pathdreamer_segmentation"], tf.uint8)

    output["depth"] = tf.io.parse_tensor(
        decoded_example["video/depth"], out_type=tf.float32)
    output["depth"] = tf.ensure_shape(output["depth"], preprocessed_shape)

    output["pathdreamer_depth"] = tf.io.parse_tensor(
        decoded_example["video/pathdreamer_depth"], out_type=tf.float32)
    output["pathdreamer_depth"] = tf.ensure_shape(output["pathdreamer_depth"],
                                                  preprocessed_shape)

    output["depth_scale"] = decoded_example["depth_scale"]

    if self.return_filename:
      output["scan_id"] = decoded_example["scan_id"]
    return output

  def get_file_patterns(self,
                        split: Optional[str] = None,
                        file_pattern: Optional[str] = None):
    """Pass the file pattern to find corresponding files for given split."""

    if not file_pattern:
      if split not in ("train", "val_seen", "val_unseen"):
        raise ValueError(
            f"Expected split to be one of ['train', 'val_seen', 'val_unseen'], got {split}"
        )
      file_pattern = self.data_dir + (f"{split}*.tfrecord")
    return file_pattern

  def _transform_fn(self, features, seed, random_generator):
    image = tf.image.resize(features["image"],
                            (self.image_size, self.image_size * 2))
    segmentation = tf.image.resize(
        features["segmentation"][..., None],
        (self.image_size, self.image_size * 2),
        method="nearest")
    pathdreamer_segmentation = tf.image.resize(
        features["pathdreamer_segmentation"][..., None],
        (self.image_size, self.image_size * 2),
        method="nearest")
    depth = tf.image.resize(
        features["depth"][..., None], (self.image_size, self.image_size * 2),
        method="nearest")
    pathdreamer_depth = tf.image.resize(
        features["pathdreamer_depth"][..., None],
        (self.image_size, self.image_size * 2),
        method="nearest")

    if self.horizontal_mask_ratio > 0:
      width = self.image_size * 2
      mask_start = tf.random.uniform((), 0, width)
      mask_end = (mask_start + width * (1 - self.horizontal_mask_ratio)) % width
      if mask_start > mask_end:
        mask = tf.math.logical_or(
            tf.range(width, dtype=tf.float32) > mask_start,
            tf.range(width, dtype=tf.float32) < mask_end)[None, None, :, None]
      else:
        mask = tf.math.logical_and(
            tf.range(width, dtype=tf.float32) > mask_start,
            tf.range(width, dtype=tf.float32) < mask_end)[None, None, :, None]
      masked_image = image * tf.cast(mask, image.dtype)
    else:
      masked_image = image

    output = dict(
        id=features["id"],
        image=masked_image,
        original_image=image,
        position=features["position"],
        mask=features["mask"],
        segmentation=segmentation,
        pathdreamer_segmentation=pathdreamer_segmentation,
        depth=depth,
        pathdreamer_depth=pathdreamer_depth,
        depth_scale=features["depth_scale"],
        dataset_type=features["dataset_type"])

    if self.z_generator == "cpu_generator":
      z = random_generator.normal(shape=(self.z_dim,))
      output.update({"z": z})
      logging.info("random z is generated by CPU Generator")
    elif self.z_generator == "cpu_random":
      z = tf.random.normal((self.z_dim,))
      output.update({"z": z})
      logging.info("random z is generated by CPU Random")
    else:
      logging.info("random z is generated by on Device (TPU/GPU)")
    return output

  def _train_transform_fn(self, features, seed, random_generator):
    return self._transform_fn(features, seed, random_generator)

  def _eval_transform_fn(self, features, seed, random_generator):
    features = self._transform_fn(features, seed, random_generator)
    features["one_hot_mask"] = tf.one_hot(
        tf.cast(features["segmentation"][..., 0], tf.int32), self.num_classes)
    return features

  def _train_batch_transform_fn(self, features, seed):
    """Implements batch transform for faster speed.

    https://www.tensorflow.org/guide/data_performance#vectorizing_mapping.

    Args:
      features: The feature dict.
      seed: The random seed used for randomness.

    Returns:
      The transformed feature dict.
    """
    image = features["image"]
    batch_size, seq_len, height, width, _ = image.shape
    image = tf.reshape(image, (batch_size * seq_len, height, width, -1))
    image = tf.reshape(image, (batch_size, seq_len, height, width, -1))

    features.update({
        "image": image,
    })
    return features

  @property
  def num_examples(self):
    return {"train": 4675, "val_unseen": 783, "val_seen": 340}
