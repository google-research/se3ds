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

"""Helper function for creating and logging TF variable overviews."""

from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
import numpy as np
import tensorflow as tf

ModuleOrVariables = Union[tf.Module, List[tf.Variable]]


def flatten_dict(input_dict: Dict[str, Any],
                 *,
                 prefix: str = "",
                 delimiter: str = "/") -> Dict[str, Any]:
  """Flattens the keys of a nested dictionary."""
  output_dict = {}
  for key, value in input_dict.items():
    nested_key = f"{prefix}{delimiter}{key}" if prefix else key
    if isinstance(value, dict):
      output_dict.update(
          flatten_dict(value, prefix=nested_key, delimiter=delimiter))
    else:
      output_dict[nested_key] = value
  return output_dict


def count_parameters(params: Union[tf.Module, Dict[str, Any]]) -> int:
  """Returns the count of variables for the module or parameter dictionary."""
  if isinstance(params, tf.Module):
    return sum(np.prod(v.shape) for v in params.trainable_variables)  # pytype: disable=attribute-error
  params = flatten_dict(params)
  return sum(np.prod(v.shape) for v in params.values())


def get_params(module: tf.Module) -> Tuple[List[str], List[np.ndarray]]:
  """Returns the trainable variables of a module as flattened dictionary."""
  assert isinstance(module, tf.Module), module
  variables = sorted(module.trainable_variables, key=lambda v: v.name)
  return [v.name for v in variables], [v.numpy() for v in variables]


def get_parameter_overview(params: Union[tf.Module, Dict[str, np.ndarray]],
                           include_stats: bool = True,
                           max_lines: Optional[int] = None):
  """Returns a string with variables names, their shapes, count.

  Args:
    params: Dictionary with parameters as NumPy arrays. The dictionary can be
      nested. Alternatively a `tf.Module` can be provided, in which case the
      `trainable_variables` of the module will be used.
    include_stats: If True add columns with mean and std for each variable.
    max_lines: If not `None`, the maximum number of variables to include.

  Returns:
    A string with a table like in the example.

  +----------------+---------------+------------+
  | Name           | Shape         | Size       |
  +----------------+---------------+------------+
  | FC_1/weights:0 | (63612, 1024) | 65,138,688 |
  | FC_1/biases:0  |       (1024,) |      1,024 |
  | FC_2/weights:0 |    (1024, 32) |     32,768 |
  | FC_2/biases:0  |         (32,) |         32 |
  +----------------+---------------+------------+

  Total: 65,172,512
  """
  if isinstance(params, tf.Module):
    names, values = get_params(params)
  else:
    assert isinstance(params, dict)
    if params:
      params = flatten_dict(params)
      names, values = map(list, tuple(zip(*sorted(params.items()))))
    else:
      names, values = [], []

  class _Column:

    def __init__(self, name, values):
      self.name = name
      self.values = values
      self.width = max(len(v) for v in values + [name])

  columns = [
      _Column("Name", names),
      _Column("Shape", [str(v.shape) for v in values]),
      _Column("Size", [f"{np.prod(v.shape):,}" for v in values]),
  ]
  if include_stats:
    columns.extend([
        _Column("Mean", [f"{v.mean():.3}" for v in values]),
        _Column("Std", [f"{v.std():.3}" for v in values]),
    ])

  var_line_format = "|" + "".join(f" {{: <{c.width}s}} |" for c in columns)
  sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")
  header = var_line_format.replace(">", "<").format(*[c.name for c in columns])
  separator = sep_line_format.format(*["" for c in columns])

  lines = [separator, header, separator]
  for i in range(len(names)):
    if max_lines and len(lines) >= max_lines - 3:
      lines.append("[...]")
      break
    lines.append(var_line_format.format(*[c.values[i] for c in columns]))

  total_weights = count_parameters(params)
  lines.append(separator)
  lines.append("Total: {:,}".format(total_weights))
  return "\n".join(lines)


def log_parameter_overview(params: Union[tf.Module, Dict[str, np.ndarray]],
                           msg: Optional[str] = None):
  """Writes a table with variables name and shapes to INFO log.

  See get_parameter_overview for details.

  Args:
    params: Dictionary with parameters as NumPy arrays. The dictionary can be
      nested. Alternatively a `tf.Module` can be provided, in which case the
      `trainable_variables` of the module will be used.
    msg: Message to be logged before the overview.
  """
  table = get_parameter_overview(params)
  lines = [msg] if msg else []
  lines += table.split("\n")
  # The table can to large to fit into one log entry.
  for i in range(0, len(lines), 80):
    logging.info("\n%s", "\n".join(lines[i:i + 80]))
