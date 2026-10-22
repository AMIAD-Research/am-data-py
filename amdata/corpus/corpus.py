"""
The module represents a corpus with LR2 format.
"""


from typing import Dict, List

from abc import ABC

import pandas as pd

from amdata.corpus.corpus_imp import CorpusImplementation

class Corpus(ABC):
  """
  A corpus with LR2 format.

  A corpus contains:
    - a list (mandatory);
    - an annotation csv (mandatory, contains all data about an element of the
      corpus);
    - several folders containing files (media, annotations), the organization
      of these files is given by the annotation csv.
    
  We use the name MetaCorpus when you need to sample elements from other corpora for each element of the corpus (eg a MetaCorpus of 
  speaker verification trials, for each trial, you need to sample an element from a speaker model corpus and an element from a test utterance corpus).

  * Example of config
  ```yaml
  corpus:
    _target_: amdata.corpus.CorpusBase
    implementation:
      _target_: amdata.corpus.CorpusImplementationPandas
    directory: ${corpus_directory}/sre24
    list_filename: list/sre24eval-audioeval-trial.csv
    data_output:
      id: str
      model: str
      segment: str
      targettype: str
      target_index:
        _target_: amdata.mapper.LabelToIndexMapper
        param_template:
          label: targettype
      audio:
        _target_: amdata.corpus.CorpusBase
        implementation:
          _target_: amdata.corpus.CorpusImplementationPandas
        directory: ${corpus_directory}/sre24
        list_filename: list/sre24eval-audioeval.csv
        data_selection_key_parent: segment
        data_output:
          audio_path: str
          speaker: str
          language: str
          audio:
            _target_: amdata.media.Audio
            param_template:
              file_suffix: audio_path
            file_prefix: ${corpus_directory}/sre24
      speaker:
        _target_: amdata.corpus.CorpusBase
        implementation:
          _target_: amdata.corpus.CorpusImplementationPandas
        directory: ${corpus_directory}/sre24
        list_filename: list/sre24eval-audioeval-enrollment.csv
        data_selection_key_parent: model
        data_output:
          id: str
          segment: str
          segmentlist:
            _target_: amdata.serializer.StringSerializer
          param_template:
            input: segment
        audio:
          _target_: amdata.corpus.CorpusBase
          implementation:
            _target_: amdata.corpus.CorpusImplementationPandas
          directory: ${corpus_directory}/sre24
          list_filename: list/sre24eval-audioeval.csv
          data_selection_key_parent: segmentlist
          data_output:
            audio_path: str
            speaker: str
            language: str
            audio:
              _target_: amdata.media.Audio
              param_template:
                file_suffix: audio_path
              file_prefix: ${corpus_directory}/sre24
  ```
  """
  
  def __init__(self,
    directory: str,
    implementation: CorpusImplementation,
    list_filename: str = None,
    filters: List[object]=None,
    data_selection_key: str = "id",
    data_selection_key_parent: str = "id",
    data_output: Dict[str,str] = None,
    load: bool = True,
    init_from_corpus: str = "",
    name: str="",
    root: bool=True,
    cutters: List[object]=None,
    mappers: Dict=None,
    ) -> None:
    """
    Args:
        directory (str): root directory of the corpus.
        list_filename (str): filename of the list of elements of the corpus.
        implementation (CorpusImplementation): implementation object of the corpus.
        filters (object, optional): filters to apply to the corpus. Defaults to [].
        data_selection_key (str, optional): column of the ids in the current corpus. Defaults to "id".
        data_selection_key_parent (str, optional): column of the ids in the parent corpus, only used if it is a MetaCorpus. Defaults to "id".
        data_output (Dict[str,str], optional): dictionary of labels to output. The keys must correspond to: columns of the annotation file or to new information that is defined by an object (serializer, mapper or corpus). The values correspond to classes to native python types or to classes to read the media path or transform the annotations. If the value is another corpus, the data_selection_key_parent parameter of the child corpus is used to search for an id in the parent corpus. Defaults to None.
        load (bool, optional): load the corpus from files. Defaults to True.
        init_from_corpus (str, optional): name of a corpus from which to initialize the corpus. Used when the same corpus is used several time in a complex MetaCorpus. Defaults to "".
        name (str, optional): name of the corpus. Defaults to "".
        root (bool, optional): is the corpus a root corpus. Defaults to True.
        cutters (list[object], optional): list of cutters. Defaults to [].
        mappers (Dict, optional): dictionary of mappers. Default to {}.
    """ 

    
    self._directory = directory
    self._list_filename = list_filename

    self._filters = filters if filters is not None else []

    self._data_selection_key = data_selection_key
    self._data_selection_key_parent = data_selection_key_parent
    self._data_output = data_output

    self._load = load
    self._init_from_corpus = init_from_corpus

    self._name = name
    self._root = root

    self._cutters = cutters if cutters is not None else []
    self._mappers = mappers if mappers is not None else {}

    self._implementation = implementation
    self._implementation.set_directory(self._directory)
    self._implementation.set_list_filename(self._list_filename)
    self._implementation.set_filters(self._filters)
    self._implementation.set_data_selection_key(self._data_selection_key)
    self._implementation.set_data_selection_key_parent(self._data_selection_key_parent)

    self._implementation.set_data_output(self._data_output)
    self._implementation.set_name(self._name)
    self._implementation.set_root(self._root)
    self._implementation.set_init_from_corpus(self._init_from_corpus)

    self._implementation.set_cutters(self._cutters)
    self._implementation.set_mappers(self._mappers)
    

    # Load the corpus from files only if load is True
    # and if the init_from_corpus is not empty. In this case, the corpus internals will be shared or copied from another corpus.
    if self._load and self._init_from_corpus == "":
      self.load()
    
    self.setup()


  def get_implementation(self,
  ) -> CorpusImplementation:
    """
    Get the implementation of the corpus.

    Returns:
        CorpusImplementation: implementation of the corpus.
    """
    return self._implementation

  def get_name(self,
  ) -> str:
    """
    Get the name of the corpus.

    Returns:
        str: name of the corpus.
    """
    return self._implementation.get_name()

  def is_root(self,
  ) -> bool:
    """
    Check if the corpus is a root corpus.

    A root corpus is the root of a MetaCorpus.

    Returns:
        bool: True if the corpus is a root corpus, False otherwise.
    """
    return self._implementation.is_root()

  def get_init_from_corpus(self,
    ) -> str:
    """
    Get the name of the corpus from which the current corpus is initialized.

    Returns:
        str: name of the corpus from which the current corpus is initialized.
    """
    return self._implementation.get_init_from_corpus()


  def get_n(self,
  ) -> int:
    """
    Get the number of elements of the corpus

    Returns:
        int: number of elements.
    """
    return self._implementation.get_n()

  def get_ids(self,
  ) -> List:
    """
    Get a list of ids of elements in the corpus.

    Returns:
        List: list of ids of the corpus elements.
    """
    return self._implementation.get_ids()

  def get_annotation_keys(self,
  ) -> List[str]:
    """
    Get annotation keys. Annotation keys correspond to the columns of the annotation files.

    Returns:
        List[str]: list of annotation keys.
    """
    return self._implementation.get_annotation_keys()

  def get_annotation_from_selection(self,
        selection_values: List,
        selection_key: str = "id"
  ) -> Dict:
    """
    Get the annotations from a selection of values.

    Args:
        selection_values (List): values of the selection.
        selection_key (str): key of the selection. Default is "id".
    Returns:
        Dict: dictionary of values. Keys are given by the data_output parameter of the object. Only keys of builtin types are returned.
    """
    return self._implementation.get_annotation_from_selection(selection_values=selection_values, selection_key=selection_key)

  def get_annotation_from_index(self,
      loc: slice
  ) -> Dict:
    """
    Get the annotations from indices
    Args:
        loc (slice): indices of the elements to get.
    Returns:
        Dict: dictionary of values. Keys are given by the data_output parameter of the object. Only keys of builtin types are returned.
    """
    return self._implementation.get_annotation_from_index(loc)

  def get_data_selection_key(self,
  ) -> str:
    """
    Get the key used to select the data.

    Returns:
        str: key used to select the data.
    """
    return self._implementation.get_data_selection_key()


  def get_data_keys(self,
  ) -> List[str]:
    """
    Get the names of the columns in the list table.

    Returns:
        List[str]: sequence of column names.
    """
    return self._implementation.get_data_keys()


  def get_data_from_idx(self,
    idx: int | slice
  ) -> List[Dict]:
    """
    Get a data item corresponding to row index in the list table.

    Args:
        idx (int|slice): index of an element in the corpus

    Returns:
        List[Dict]: list of dictionaries, a dictionary if given for each input index. Keys are given by the data_output parameter of the object. Values can be builtin types, dictionaries for media files, or a corpus.
    """
    return self._implementation.get_data_from_idx(idx)


  def get_data_from_key(self,
    value: str
  ) -> List[Dict]:
    """
    Get a data item corresponding to the specified value of the data_selection_key of the corpus.

    Args:
        value (str): value of the data_selection_key corresponding to one element.

    Returns:
        List[Dict]: list (of length 1) with a dictionary. Keys are given by the data_output parameter of the object. Values can be builtin types, dictionaries for media files, or a corpus.
    """
    return self._implementation.get_data_from_key(value)
  
  def setup(self
  ) -> None:
    """
    Setup corpus internals.
    
    This method should be called after loading from files or from another corpus.
    """
    self._implementation.setup()

  def load(self
  ) -> None:
    """
    Load corpus internals from external files
    """
    self._implementation.load()

  def save(self,
           path: str
  ) -> None:
    """
    Save corpus internals to persistent files.
    """
    self._implementation.save(path)

class CorpusBase(Corpus):
  """
  Generic base corpus.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

class CorpusUnion(Corpus):
  """
  Union of several corpora.
  """

  def __init__(self, corpus_list:List, corpus_names:List=None, relative_path_columns:List[str]=None, corpus_column_name:str=None,**kwargs):
    """
    Args:
        corpus_list (List): list of corpus objects
        corpus_names (List): list of names of corpus
        relative_path_columns (List[str]): list of columns corresponding to the relative path of a file
        corpus_column_name (str, optional): name of the column with corpus id in the merged corpus. Defaults to None.
    """


    self._corpus_list = corpus_list
    self._corpus_names = corpus_names
    self._corpus_column_name = corpus_column_name
    self._relative_path_columns = relative_path_columns

    kwargs["directory"] = ""
    kwargs["list_filename"] = ""
    kwargs["load"] = False
    super().__init__(**kwargs)

  def setup(self
  ) -> None:
    """
    Setup corpus internals.
    
    This method should be called after loading from files or from another corpus.
    """
    self._implementation.set_corpus_list(corpus_list=self._corpus_list)
    self._implementation.set_corpus_names(corpus_names=self._corpus_names)
    self._implementation.set_corpus_column_name(corpus_column_name=self._corpus_column_name)
    self._implementation.set_relative_path_columns(relative_path_columns=self._relative_path_columns)
    self._implementation.setup()

