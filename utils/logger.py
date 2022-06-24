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

"""Provides a helper class to easily log summaries to various 'backends'.

Currently supported 'backends':
- Summary files
- Logs
"""

from typing import Callable, Optional

from absl import logging
import tensorflow as tf


class UniversalLogger:
  """Helper class for logging model outputs.

  This class writes scalar results to multiple destinations:
  - INFO logs for easy debugging.
  - Summary files in `workdir` for analysis in TensorBoard.
  """

  def __init__(self,
               workdir: str,
               step: int,
               num_train_steps: Optional[int] = None,
               logging_fn: Optional[Callable[[str], None]] = None):
    """Create a new logger object.

    Args:
      workdir: Path to model directory.
      step: Current step.
      num_train_steps: Total number for training steps. Required for reporting
        progress to XManager.
      logging_fn: Function used for writing logs. Defautls to `logging.info`.
        Set this to `print` when working from Colab.
    """
    self.summary_writer = tf.summary.create_file_writer(workdir)
    self._num_train_steps = num_train_steps
    self._print = logging_fn or logging.info
    self._measurement_series_cache = {}
    self._steps_per_sec_start_step = step

  def log_scalars(self, step: int, **kwargs):
    """Log scalars (given as keyword arguments)."""
    log_msg = ", ".join([f"{k} = {v:.3f}" for k, v in sorted(kwargs.items())])
    self._print(f"[{step}] {log_msg}")

    with self.summary_writer.as_default():
      for k, v in sorted(kwargs.items()):
        tf.summary.scalar(k, v, step=step)

  def log_images(self, step: int, max_outputs: int = 10, **kwargs):
    """Log images (given as keyword arguments)."""

    with self.summary_writer.as_default():
      for k, v in sorted(kwargs.items()):
        tf.summary.image(k, v, step=step, max_outputs=max_outputs)

  def _report_progress(self, step: int, steps_per_sec: float):
    """Report the progress and estimated time to finish to XManager."""
    if self._num_train_steps is None:
      return
    eta_seconds = (self._num_train_steps - step) / (steps_per_sec + 1e-7)
    message = (f"{100 * step / self._num_train_steps:.1f}% @{step}, "
               f"{steps_per_sec:.1f} steps/s, ETA: {eta_seconds / 60:.0f} min")
    self._print(f"Reporting progress: {message}")
