
from typing import Dict, Optional, List
from abc import ABC, abstractmethod

class CorpusImplementation(ABC):
    """
    Abstract class for implementing a Corpus.
    """
    
    def __init__(self,
        directory: str="",
        list_filename: str="",
        filters: List[object]=None,
        init_from_corpus: str="",
        name: str="",
        root: bool=True,
        cutters: List[object]=None,
        mappers: Dict[str,object]=None,
    ) -> None:
        """
        Args:
            directory (str, optional): directory of the corpus. Defaults to "".
            list_filename (str, optional): name of the list file. Defaults to "".
            filters (object, optional): filter for the corpus. Defaults to [].
            init_from_corpus (str, optional): name of the corpus to initialize from. Defaults to "".
            name (str, optional): name of the corpus. Defaults to "".
            root (bool, optional): True if the corpus is a root corpus. Defaults to True.
            cutters (List[object], optional): list of cutters to apply to the corpus.
                Defaults to [].
            mappers (Dict[str, object], optional): mapping of label transformations
                (key -> mapper object). Defaults to {}.
        """

        self._directory = directory
        self._list_filename = list_filename

        self._filters = filters if filters is not None else []

        self._name = name
        self._root = root

        self._init_from_corpus = init_from_corpus

        self._data_output = {}
        self._data_annotation_keys = []
        self._data_derived_keys = []
        self._cutters = cutters if cutters is not None else []
        self._mappers = mappers if mappers is not None else {}

    def get_name(self,
        ) -> str:
        """
        Get the name of the corpus.

        Returns:
            str: name of the corpus
        """
        return self._name

    def set_name(self,
        name: str
    ) -> None:
        """
        Set the corpus name.
        """
        self._name = name

    def set_cutters(self,
        cutters: List[object]
    ) -> None:
        """
        Set the cutter list.
        """
        self._cutters = cutters

    def set_mappers(self,
        mappers: Dict
    ) -> None:
        """
        Set the mapper dictionary.
        """
        self._mappers = mappers

    def is_root(self,
        ) -> bool:
        """
        Check if the corpus is a root corpus.
        Returns:
            bool: True if the corpus is a root corpus, False otherwise.
        """
        return self._root

    def set_root(self,
        root: bool
    ) -> None:
        """
        Set the root attribute.
        Args:
            root (bool): True if the corpus is a root corpus, False otherwise.
        """
        self._root = root

    def get_init_from_corpus(self,
    ) -> str:
        """
        Get the name of the corpus from which the current corpus is initialized.
        Returns:
            str: name of the corpus from which the current corpus is initialized.
        """
        return self._init_from_corpus

    def set_init_from_corpus(self,
        init_from_corpus: str
    ) -> None:
        """
        Set the init_from_corpus attribute.
        Args:
            init_from_corpus (str): name of the corpus to initialize from
        """
        self._init_from_corpus = init_from_corpus

    def get_directory(self
    ) -> str:
        """
        Get the directory of the corpus files.
        Returns:
            str: directory of the corpus.
        """
        return self._directory

    def set_directory(self,
        directory: str
    ) -> None:
        """
        Set the directory of the corpus files.
        Args:
            directory (str): directory of the corpus.
        """
        self._directory = directory

    def get_list_filename(self,
        ) -> str:
        """
        Get the name of the list file.
        Returns:
            str: name of the list file.
        """
        return self._list_filename

    def set_list_filename(self,
        list_filename: str
    ) -> None:
        """
        Set the name of the list file.
        Args:
            list_filename (str): name of the list file.
        """
        self._list_filename = list_filename

    def get_filters(self,
        ) -> Optional[object]:
        """
        Get the filters.
        Returns:
            object: filters.
        """
        return self._filters

    def set_filters(self,
        filters: List[object]
    ) -> None:
        """
        Set the filters
        Args:
            filters: filters
        """
        self._filters = filters

    @abstractmethod
    def get_n(self,
        ) -> int:
        """
        Get the number of elements of the corpus
        Returns:
            int: number of elements.
        """
        pass


    @abstractmethod
    def get_ids(self,
        ) -> List[int]:
        """
        Get a list of ids of elements in the corpus.

        Returns:
            List[int]: list of ids.
        """
        pass

    @abstractmethod
    def get_list_keys(self,
        ) -> List:
        """
        Get the names of the columns in the list table.

        Returns:
            List[str]: sequence of column names.
        """
        pass

    @abstractmethod
    def get_annotation_keys(self,
        ) -> List[str]:
        """
        Get annotation keys. Annotation keys correspond to the columns of the annotation files.

        Returns:
            List[str]: list of annotation keys.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    def get_data_keys(self,
        ) -> List:
        """
        Get the list of data keys.

        The data keys are the union of annotation keys (columns of the
        annotation table that map to builtin types) and derived keys
        (media, mappers, serializers, subcorpora).

        Returns:
            List: list of data keys.
        """
        return self._data_annotation_keys + self._data_derived_keys


    @abstractmethod
    def get_data_selection_key(self,
    ) -> str:
        """
        Get the key used to select the data.
        Returns:
            str: key used to select the data.
        """
        pass

    @abstractmethod
    def set_data_selection_key(self,
        key: str
    ) -> str:
        """
        Set the key used to select the data.
        Args:
            key (str): key used to select the data.
        """
        pass

    @abstractmethod
    def set_data_selection_key_parent(self,
        key: str
    ) -> str:
        """
        Set the parent key used to select the data.
        Args:
            key (str): key used to select the data.
        """
        pass



    @abstractmethod
    def get_data_from_key(self, value: str) -> List[Dict]:
        """
        Get all data items whose selection key matches the given value.

        The selection key is the column of the annotation table used to
        identify the row (typically ``"id"``). The value must correspond
        to a unique row in the annotation table.

        Args:
            value: value of the selection key to look up.

        Returns:
            List[Dict]: list of data items (one per matching row). Each
                dict contains the keys defined by ``data_output``.
        """
        pass

    @abstractmethod
    def get_data_from_idx(self,
        loc: int | slice
    ) ->  List[Dict]:
        """
        Get data items corresponding to row indices in the annotation table.

        Args:
            loc (int | slice): index or slice of indices of elements in
                the corpus.

        Returns:
            List[Dict]: list of data dicts. Keys are given by the
                ``data_output`` parameter of the object. Values can be
                builtin types, dictionaries for media files, or corpus
                objects.
        """
        pass

    def get_subcorpora(self
    ) -> List:
        """
        Get the list of defined corpora in by some data_output keys (recursively)

        Returns:
            List: list of subcorpora.
        """

        subcorpora = []
        if self._data_output is not None:
            for key in self._data_output.keys():
                if 'Corpus' in str(self._data_output[key].__class__):
                    subcorpora.append(self._data_output[key])
                    # recursivity
                    subsubcorpora = self._data_output[key].get_implementation().get_subcorpora()
                    for subcorpus in subsubcorpora:
                        if not subcorpus in subcorpora:
                            subcorpora.append(subcorpus) 

        return subcorpora

    @abstractmethod
    def setup(self
    ) -> None:
        """
        Setup the corpus internals.
        This method should be called after loading from files or from another corpus.
        """
        pass

    @abstractmethod
    def load(self
    ) -> None:
        """
        Load corpus internals from external files
        """
        pass

    @abstractmethod
    def save(self,
             path: str,
    ) -> None:
        """
        Save corpus internals to persistent files.
        """
        pass

