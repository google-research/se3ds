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

"""Base dataset builder class."""

import abc
import functools
from typing import Optional

import tensorflow.compat.v2 as tf


# Size of TFRecord reader buffer, per file. (The TFRecordDataset default, 256KB,
# is too small and makes TPU trainers input-bound.)
TFRECORD_READER_BUFFER_SIZE_BYTES = 64 * (1 << 20)  # 64 MiB


class BaseDataset(abc.ABC):
  """Base Dataset."""

  def __init__(self,
               image_size: int,
               num_classes: Optional[int] = None,
               z_dim: int = 128,
               z_generator: str = "cpu_generator"):
    """init function for BaseDataset.

    Args:
      image_size: Width and height of resized images.
      num_classes: Number of class labels in this dataset.
      z_dim: The dimension of random noise.
      z_generator: "cpu_generator" uses tf.random.Generator on cpu;
        "cpu_random" uses tf.random on cpu, otherwise use on device tf.random.
    """

    self.image_size = image_size
    self.num_classes = num_classes
    self.z_dim = z_dim
    self.z_generator = z_generator

  def as_dataset(self,
                 file_patterns: str,
                 shuffle: bool,
                 seed: Optional[int] = None,
                 parallel_calls: int = tf.data.experimental.AUTOTUNE):
    """Returns the split as a `tf.data.Dataset`."""

    assert tf.io.gfile.glob(file_patterns), (
        f"No data files matched {file_patterns}")
    files = tf.data.Dataset.list_files(
        file_patterns, shuffle=shuffle, seed=seed)

    dataset_type = functools.partial(
        tf.data.TFRecordDataset,
        buffer_size=TFRECORD_READER_BUFFER_SIZE_BYTES,
        num_parallel_reads=parallel_calls)
    dataset = files.interleave(
        dataset_type,
        cycle_length=parallel_calls,
        num_parallel_calls=parallel_calls,
        deterministic=True)

    processed_ds = dataset.map(
        self._parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return processed_ds

  @abc.abstractmethod
  def _parse(self, example):
    """Returns a parsed, decoded `tf.data.Dataset`.

    Args:
      example: Scalar string tensor representing bytes data.

    Returns:
      outputs: Dict of string feature names to Tensor feature values.
    """
    pass

  def input_fn(
      self,
      split: str,
      global_batch_size: int,
      strategy: Optional[tf.distribute.Strategy] = None,
      num_epochs: Optional[int] = None,
      shuffle: bool = False,
      shuffle_buffer_size: int = 1000,
      cache: bool = False,
      file_pattern: Optional[str] = None,
      seed: Optional[int] = 1,
      parallel_calls: int = tf.data.experimental.AUTOTUNE) -> tf.data.Dataset:
    """return a distributed tf.data.Dataset instance."""

    def dataset_fn(input_context: tf.distribute.InputContext):
      local_seed = seed  # Seed for this machine.
      if input_context:
        num_input_pipelines = input_context.num_input_pipelines
        input_id = input_context.input_pipeline_id
        generator = tf.random.Generator.from_seed(seed)
        gens = generator.split(num_input_pipelines)
      else:
        generator = tf.random.Generator.from_seed(seed)
        input_id = 0
        gens = generator.split(1)
      if local_seed is not None:
        local_seed += input_id
      file_patterns = self.get_file_patterns(split, file_pattern)
      ds = self.as_dataset(file_patterns, shuffle, local_seed, parallel_calls)
      if cache:
        ds = ds.cache()
      ds = ds.repeat(num_epochs)
      if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed=local_seed)
      if split == "train":
        map_fn = functools.partial(
            self._train_transform_fn,
            seed=local_seed,
            random_generator=gens[input_id])
      else:
        map_fn = functools.partial(
            self._eval_transform_fn,
            seed=local_seed,
            random_generator=gens[input_id])
      ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      if input_context:
        per_replica_batch_size = input_context.get_per_replica_batch_size(
            global_batch_size)
      else:
        per_replica_batch_size = global_batch_size
      ds = ds.batch(
          per_replica_batch_size,
          drop_remainder=True)
      if split == "train":
        batch_map_fn = functools.partial(
            self._train_batch_transform_fn, seed=local_seed)
        ds = ds.map(batch_map_fn).prefetch(tf.data.experimental.AUTOTUNE)
      else:
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
      return ds
    if strategy:
      return strategy.distribute_datasets_from_function(dataset_fn)
    else:
      return dataset_fn(input_context=None)

  @abc.abstractmethod
  def get_file_patterns(self, split, file_pattern):
    pass

  @property
  @abc.abstractmethod
  def num_examples(self):
    pass

  def _train_transform_fn(self, features, seed, random_generator=None):
    del seed, random_generator
    return features

  def _eval_transform_fn(self, features, seed, random_generator=None):
    del seed, random_generator
    return features

  def _train_batch_transform_fn(self, features, seed):
    """Implements batch transform for faster speed."""

    del seed
    return features
