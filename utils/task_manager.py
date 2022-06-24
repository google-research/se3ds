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

"""Auxiliary class and functions for eval jobs."""

import csv
import os
import re
import time
from typing import Optional, Dict, Any, Iterable

from absl import logging
import gin
import tensorflow as tf


class _DummyParserDelegate(gin.config_parser.ParserDelegate):
  """Dummy class required to parse Gin configs.

  Our use case (just get the config as dictionary) does not require real
  implementations the two methods.
  """

  def configurable_reference(self, scoped_name, evaluate):
    return scoped_name

  def macro(self, scoped_name):
    return scoped_name


def _parse_gin_config(config_path):
  """Parses a Gin config into a dictionary. All values are strings."""
  with tf.io.gfile.GFile(config_path) as f:
    config_str = f.read()
  parser = gin.config_parser.ConfigParser(config_str, _DummyParserDelegate())
  config = {}
  for statement in parser:
    if not isinstance(statement, gin.config_parser.ImportStatement):
      name = statement.selector + "." + statement.arg_name
      config[name] = statement.value
  return config


class TaskManager:
  """Class for checking the model folder repeately for evaluation."""

  def __init__(self,
               model_dir: str) -> None:
    self._model_dir = model_dir

  @property
  def model_dir(self) -> str:
    return self._model_dir

  def mark_training_done(self) -> None:
    with tf.io.gfile.GFile(
        os.path.join(self.model_dir, "TRAIN_DONE"), "w") as f:
      f.write("")

  def is_training_done(self) -> None:
    return tf.io.gfile.exists(os.path.join(self.model_dir, "TRAIN_DONE"))

  def add_eval_result(
      self,
      checkpoint_path: str,
      result_dict: Dict[str, Any],
      default_value: int = -1) -> None:
    pass

  def _get_checkpoints_with_results(self):
    return set()

  def unevaluated_checkpoints(self,
                              timeout: int = 3600 * 8,
                              num_batched_steps: int = 1,
                              eval_every_steps: Optional[int] = None,
                              ) -> Iterable[str]:
    """Generator for checkpoints without evaluation results.

    Args:
      timeout: Optional timeout for waiting for new checkpoints. Set this to
        do continious evaluation.
      num_batched_steps: Steps that are batched into a single tf.function.
        Required for computing correct evaluation checkpoints.
      eval_every_steps: Only evaluate checkpoints from steps divisible by this
                         integer.

    Yields:
      Path to checkpoints that have not yet been evaluated.
    """
    logging.info("Looking for checkpoints in %s", self._model_dir)
    evaluated_checkpoints = self._get_checkpoints_with_results()
    last_eval = time.time()
    while True:
      logging.info(
          "what is in %s:  are  %s",
          self._model_dir, tf.io.gfile.listdir(self._model_dir))
      unevaluated_checkpoints = []
      checkpoint_state = tf.train.get_checkpoint_state(self._model_dir)
      if checkpoint_state:
        checkpoints = set(checkpoint_state.all_model_checkpoint_paths)
      else:
        checkpoints = set()
      unevaluated_checkpoints = checkpoints - evaluated_checkpoints
      step_and_ckpt = sorted(
          (int(x.split("-")[-1]), x) for x in unevaluated_checkpoints)

      unevaluated_checkpoints = []
      for step, ckpt in step_and_ckpt:
        if eval_every_steps:
          if step > num_batched_steps and (
              step % eval_every_steps < num_batched_steps):
            unevaluated_checkpoints.append(ckpt)
        else:
          unevaluated_checkpoints.append(ckpt)

      logging.info(
          "Found checkpoints: %s\nEvaluated checkpoints: %s\n"
          "Unevaluated checkpoints: %s", checkpoints, evaluated_checkpoints,
          unevaluated_checkpoints)
      for checkpoint_path in unevaluated_checkpoints:
        yield checkpoint_path

      if unevaluated_checkpoints:
        evaluated_checkpoints |= set(unevaluated_checkpoints)
        last_eval = time.time()
        continue
      if time.time() - last_eval > timeout or self.is_training_done():
        break
      time.sleep(5)


class TaskManagerWithCsvResults(TaskManager):
  """Task Manager that writes results to a CSV file."""

  def __init__(self,
               model_dir: str,
               score_file: Optional[str] = None) -> None:
    super().__init__(model_dir)
    if score_file is None:
      score_file = os.path.join(model_dir, "scores.csv%r=3.2")
    else:
      score_file = os.path.join(model_dir, score_file + "%r=3.2")
    self._score_file = score_file

  def _get_checkpoints_with_results(self):
    """Return the checkpoints as set."""
    if not tf.io.gfile.exists(self._score_file):
      return set()
    with tf.io.gfile.GFile(self._score_file) as f:
      reader = csv.DictReader(f)
      return {r["checkpoint_path"] for r in reader}
    return set()

  def add_eval_result(self,
                      checkpoint_path: str,
                      result_dict: Dict[str, Any],
                      default_value: int) -> None:
    """Add eval result to the CSV file."""
    step = int(os.path.basename(checkpoint_path).split("-")[-1])
    config = self._get_config_for_step(step)
    csv_header = (
        ["checkpoint_path", "step"] + sorted(result_dict) + sorted(config))
    write_header = not tf.io.gfile.exists(self._score_file)
    if write_header:
      with tf.io.gfile.GFile(self._score_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
        writer.writeheader()
    row = dict(checkpoint_path=checkpoint_path, step=str(step), **config)
    for k, v in result_dict.items():
      if isinstance(v, float):
        v = "{:.3f}".format(v)
      row[k] = v
    with tf.io.gfile.GFile(self._score_file, "a") as f:
      writer = csv.DictWriter(f, fieldnames=csv_header, extrasaction="ignore")
      writer.writerow(row)

  def _get_config_for_step(self, step):
    """Returns the latest operative config for the global step as dictionary."""
    saved_configs = tf.io.gfile.glob(
        os.path.join(self.model_dir, "operative_config-*.gin"))
    if not saved_configs:
      return {}
    get_step = lambda fn: int(re.findall(r"operative_config-(\d+).gin", fn)[0])
    config_steps = [get_step(fn) for fn in saved_configs]
    assert config_steps
    last_config_step = sorted([s for s in config_steps if s <= step])[-1]
    config_path = os.path.join(
        self.model_dir, "operative_config-{}.gin".format(last_config_step))
    return _parse_gin_config(config_path)
