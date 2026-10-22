from typing import Dict, List
import pandas as pd

from abc import ABC, abstractmethod

class Mapper(ABC):
  """
  Abstract class for mappers.

  Mappers are used to define new annotations based on existing annotations.
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

  def set_param_template(self, param_template: Dict[str,str]) -> None:
      """
      Setter for param_template.
      """
      self._param_template = param_template

  @abstractmethod
  def get_data(self,**kwargs):
    raise NotImplementedError

  def __call__(self, **kwargs) -> object:
    output = self.get_data(**kwargs)
    return output

  @classmethod
  def from_table(cls, row: Dict, args: Dict) -> object:
      """
      Create a Mapper object from a table row.
      """
      kwargs = {}
      for key in args.keys():
          if not isinstance(args[key],list) and args[key] in row.keys():
              kwargs[key] = row[args[key]]
          else:
              kwargs[key] = args[key]

      return cls(**kwargs)

class LabelToIndexMapper(Mapper):
  """
  Mapper used to convert string labels into integer indices. 
  The mapping strategy depends on whether an explicit list of labels is provided:
  - If `labels` is given at initialization, the order of the list is preserved and indices are assigned accordingly.
  - If no `labels` are given, the mapper will build its mapping later via `set_labels`, and labels will be sorted by default unless `sort_labels` is set to False. 
  Attributes: 
    sort_labels (bool): Whether to sort labels when building the mapping. 
    _labels (List[str] | None): The original list of labels (may include duplicates). 
    _filter (callable | None): Optional filter applied to labels before indexing. 
    _output_key (str): Placeholder, not actively used (defaults to "index"). 
    _label_to_index (Dict[str, int]): Mapping from label to integer index. 
  """
  def __init__(self,
    labels: List=None,
    filter: object=None,
    output_key: str=None,
    param_template: Dict[str,str]=None,
    sort_labels: bool = True,   
  ) -> None:
    """ 
    Args: 
        labels (List, optional): List of labels defining the mapping. If provided, duplicates are removed while preserving order, and `sort_labels` is automatically set to False. Defaults to None. 
        filter (callable, optional): Function applied to the labels before building the mapping. Labels filtered out will not receive an index. Defaults to None. 
        output_key (str, optional): Placeholder argument, not currently used. Defaults to None. 
        param_template (Dict, optional): Template for extracting init arguments from corpus metadata. Defaults to None. 
        sort_labels (bool, optional): Whether to sort labels when building the mapping. If `labels` is provided, this is overridden to False. Defaults to True. 
    """
    super().__init__(param_template=param_template)

    self.sort_labels = sort_labels
    self._labels = None
    if labels is not None:
      self._labels = list(dict.fromkeys(labels))
      self.sort_labels = False 
      
      
    self._filter = filter
    self._output_key = output_key

    self._label_to_index = {}
    
    if self._labels is not None:
      self.set_labels(self._labels)

    if self._output_key is None:
      self._output_key = 'index'


  def set_labels(self,
    labels: List[str]
    ) -> None:
    """
    Set the mapping from label values to ints.
    Args:
        labels (List): list of labels.
    """
    
    self._labels = labels

    if self._filter is None:
      self._filtered_labels = self._labels
    else:
      self._filtered_labels = self._filter(self._labels)

    if self.sort_labels:
      self._filtered_labels = sorted(self._filtered_labels)

    self._label_to_index = {label: idx for idx, label in enumerate(self._filtered_labels)}


  def get_labels(self) -> List:
    """
    Getter for the list of labels (before filter).
    """
    return self._labels

  def get_data(self, label: str) -> int:
    """
    Convert a str label to an int.

    Args:
        label (str): input label

    Returns:
        int: output label, -1 if the input label is filtered or is not known by the mapper.
    """
    output = -1
    if label in self._label_to_index.keys():
      index = self._label_to_index[label]
      output = index
    else:
      print(f"{label=} not known by mapper")

    return output



class CSVLabelMapper(Mapper):
  """
  Mapper used to convert str to str, based on a csv with correspondence.
  """
  
  def __init__(self, csv_path: str, param_template: Dict[str,str]=None) -> None:
    """
    Args:
        csv_path (str): path of the csv with mapping
        param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
    """

    super().__init__(param_template=param_template)

    self._mapping = pd.read_csv(csv_path, names= ["id", "label"])
    self._mapping = self._mapping.set_index("id")
    print(self._mapping)

    self._labels = None
    self._label_to_index = {}
    
  def set_labels(self, labels: List[str]) -> None:
    """
    Set the mapping from label values to str.
    Args:
        labels (List): list of labels.
    """
    
    self._labels = labels

 
    self._label_to_index = {label: self._mapping.loc[label]["label"] for idx, label in enumerate(self._labels)}


  def get_labels(self) -> List:
    """
    Getter for the list of labels (before filter).
    """
    return self._labels

  def get_data(self, label: str) -> str | None:
    """
    Convert a str label to another str.

    Args:
        label (str): input label

    Returns:
        str: output label, None if the input label is filtered or is not known by the mapper.
    """
    output = None
    if label in self._label_to_index.keys():
      index = self._label_to_index[label]
      output = index

    return output


class CopyMapper(Mapper):
  """
  Mapper used to change the name of the label.
  """
  
  def __init__(self, param_template: Dict[str,str]=None) -> None:
    """
    Args:
        param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
    """

    super().__init__(param_template=param_template)
    self._labels = None

  def set_labels(self, labels: List[str]) -> None:
    """
    Set the labels.
    Args:
        labels (List): list of labels.
    """
    
    self._labels = labels


  def get_labels(self) -> List:
    """
    Getter for the list of labels (before filter).
    """
    return self._labels

  def get_data(self, label: str) -> str:
    """
    Return the same label

    Args:
        label (str): input label

    Returns:
        output label
    """
    return label
  


class ConstantMapper(Mapper):
  """
  Mapper used to create a column with a fixed value.
  """
  
  def __init__(self, value: object, param_template: Dict[str,str]=None) -> None:
    """
    Args:
      value: constant value of the mapper
      param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
    """

    super().__init__(param_template=param_template)
    self._labels = None
    self._value = value

  def set_labels(self, labels: List[str]) -> None:
    """
    Set the labels.
    Args:
        labels (List): list of labels.
    """
    
    self._labels = labels


  def get_labels(self) -> List:
    """
    Getter for the list of labels (before filter).
    """
    return self._labels

  def get_data(self, label: str=None) -> str :
    """
    Return the same label

    Args:
        label (str): input label

    Returns:
        output label
    """
    return self._value
  

class DictMapper(Mapper):
  """
  Mapper used to convert values with a dictionary.
  """
  
  def __init__(self,
    dictionary: Dict,
    param_template: Dict[str,str]=None,
  ) -> None:
    """
    Args:
      dictionary: conversion dictionary
      param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
    """

    super().__init__(param_template=param_template)
    self._labels = None
    self._dictionary = dictionary

  def set_labels(self, labels: List[str]) -> None:
    """
    Set the labels.
    Args:
        labels (List): list of labels.
    """
    
    self._labels = labels


  def get_labels(self) -> List:
    """
    Getter for the list of labels (before filter).
    """
    return self._labels

  def get_data(self, label: str) -> str :
    """
    Return the same label

    Args:
        label (str): input label

    Returns:
        output label
    """
    if label in self._dictionary:
      return self._dictionary[label]
    else:
      return label