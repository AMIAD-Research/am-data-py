"""
The module represents filters to apply to corpus_implementation_pandas.
"""

from typing import Dict, List

from abc import ABC, abstractmethod

import pandas as pd

class Filter(ABC):
  """
  Abstract class for filters.

  Filters are used to select a subpart of a corpus. The filter() method returns a boolean
  indicating whether an element should belong to the corpus.

  Param_template is used to select arguments of the filter() method from the annotation table.
  """
  
  def __init__(self,
    param_template: Dict[str,str]=None,
    **kwargs
  ) -> None:
    """
    Args:
        param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
    """

    self._param_template = param_template

  def get_param_template(self
  ) -> None:
      """
      Getter for param_template.
      """
      return self._param_template

  def set_param_template(self,
      param_template: Dict[str,str]
  ) -> None:
      """
      Setter for param_template.
      """
      self._param_template = param_template

  @abstractmethod
  def filter(self, value: object) -> bool:
    """
    Indicates if the value must be selected.
    Returns a bool: True if the value is selected, False instead
    """
    raise NotImplementedError

  def __call__(self,
      **kwargs
  ) -> object:
    output = self.filter(**kwargs)
    return output



class RangeFilter(Filter):
  """
  Filter that accepts values within a range defined with min_value and max_value.

  Param_template should contain the key value.


  * Example of config
  ```yaml
  _target_: amdata.filter.RangeFilter
  min_value: 2.0
  max_value: 3.0
  param_template:
    value: duration
  ```
  """

  def __init__(self, min_value: float = float('-inf'), max_value: float = float('inf'), param_template: Dict = None):
    """
    Args:
        min_value (_type_, optional): minimum accepted value. Defaults to float('-inf').
        max_value (_type_, optional): maximum accepted value. Defaults to float('inf').
        param_template (_type_, optional): dict with the key value. Defaults to None.
    """
    super().__init__(param_template)
    self.min_value = min_value
    self.max_value = max_value

  def filter(self, value) -> bool:
    """
    Filter value if outside of the range.

    Args:
        value (_type_): input value, will be converted to float

    Returns:
        bool: True if between min_value and max_value, False otherwise
    """
    value = float(value)
    return value >= self.min_value and value <= self.max_value
  


class AcceptFilter(Filter):
  """
  Filter that accepts a list of accepted values.

  Param_template should contain the key value, with the column name of the label.

  * Example of config
  ```yaml
  _target_: amdata.filter.AcceptFilter
  accepted_values:
    - CLASS_01
  param_template:
    value: class_binary
  ```

  """

  def __init__(self, accepted_values: List = None, param_template: Dict = None):
    """
    Args:
        accepted_values (list, optional): list of accepted values. Defaults to [].
        param_template (_type_, optional): : dict with the key value. Defaults to None.
    """
    super().__init__(param_template)
    self.accepted_values = accepted_values if accepted_values is not None else []

  def filter(self, value) -> bool:
    """
    Args:
        value (_type_): input value

    Returns:
        bool: whether the input value belongs to the list of accepted values
    """
    return value in self.accepted_values
  
class RejectFilter(Filter):
  """
  Filter that rejects a list of rejected values.

  Param_template should contain the key value, with the column name of the label.

  * Example of config
  ```yaml
  _target_: amdata.filter.RejectFilter
  rejected_values:
    - CLASS_05
    - CLASS_02
  param_template:
    value: class_10
  ```
  """

  def __init__(self, rejected_values: List = None, param_template: Dict = None):
    """
    Args:
        rejected_values (list, optional): list of rejectted values. Defaults to [].
        param_template (_type_, optional): : dict with the key value. Defaults to None.
    """
    super().__init__(param_template)
    self.rejected_values = rejected_values if rejected_values is not None else []

  def filter(self, value) -> bool:
    """
    Args:
        value (_type_): input value

    Returns:
        bool: whether the input value does not belong to the list of rejected values
    """
    return value not in self.rejected_values
  

