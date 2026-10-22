from pathlib import Path
from typing import Dict, Tuple, List

import os
from operator import itemgetter
import numpy as np
import pandas as pd

from amdata.corpus.corpus_imp import CorpusImplementation

from tqdm import tqdm

import ast


class CorpusImplementationPandas(CorpusImplementation):
    """
    Implementation of a Corpus with Pandas.


    Attributes:
        _table_list (pd.DataFrame): table corresponding to the list of the corpus
        _table_annotation (pd.DataFrame): table corresponding to the annotation of the corpus
        _table_index: not used for now
        _data_annotation_getter (itemgetter): itemgetter for annotation keys
        _data_annotation_keys (List[str]): subset of _data_output keys corresponding to builtins types, they correspond to columns of the annotation table
        _data_derived_keys (List[str]): other keys of data_output, corresponding to media, mappers, serializers or subcorpora
        _data_output (Dict[str,object]): dictionary with keys corresponding to keys and values corresponding to types or to objects (media, corpus, serializer, mapper, etc)
        _data_selection_key_parent: column of the ids in the parent corpus (if the corpus is the child corpus of a MetaCorpus).
        _data_selection_key: column of the ids in the  local corpus

    """

    def __init__(self, **kwargs):
        """
        Initialize a pandas-based corpus implementation.

        All keyword arguments are forwarded to the parent
        ``CorpusImplementation.__init__`` (directory, list_filename,
        filters, cutters, mappers, etc.). After the parent init, this
        class sets up its pandas-specific internal state:

            - ``_table_list``: list DataFrame (initially None)
            - ``_table_annotation``: annotation DataFrame (initially None)
            - ``_table_index``: unused
            - ``_data_selection_key_parent``: parent corpus key (None)
            - ``_data_selection_key``: local corpus selection key ("id")

        Args:
            **kwargs: arguments forwarded to ``CorpusImplementation.__init__``.
        """
        super().__init__(**kwargs)

        self._table_list = None
        self._table_annotation = None
        self._table_index = None

        self._data_selection_key_parent = None
        self._data_selection_key = "id"

    def apply_cutters(self):
        """
        Apply all cutters.
        """
        print("Apply cutters = ", self._cutters)
        for cutter in self._cutters:
            self.apply_cutter(cutter)

    def apply_mappers(self):
        """
        Apply all mappers.
        """
        for label_name in self._mappers:

            mapper = self._mappers[label_name]

            labels = mapper.get_labels()
            if labels is None:
                paramTemplate = mapper.get_param_template()
                if paramTemplate is not None:
                    labels = list(self._table_annotation[paramTemplate['label']].dropna().unique())
                    mapper.set_labels(labels)

            def call_mapper(row, mapper):
                args = {}
                if hasattr(mapper, 'get_param_template'):
                    paramTemplate = mapper.get_param_template()
                    if paramTemplate is not None:
                        for argKey,argValue in paramTemplate.items():
                            args[argKey] = row[argValue]
                return mapper(**args)
            self._table_annotation[label_name] = self._table_annotation.apply(lambda row: call_mapper(row=row, mapper=mapper), axis=1)

    
    def apply_filters(self):
        """
        Apply all filters.
        """

        if self._filters:
            for filter in self._filters:

                def call_filter(row, filter):
                    args = {}
                    if hasattr(filter, 'get_param_template'):
                        paramTemplate = filter.get_param_template()
                        for argKey,argValue in paramTemplate.items():
                            args[argKey] = row[argValue]
                    return filter(**args)
                row_mask = self._table_annotation.apply(lambda row: call_filter(row=row, filter=filter), axis=1)
                self._table_annotation = self._table_annotation[row_mask]


    def get_n(self,
        ) -> int:
        if self._table_annotation is None:
            raise ValueError("Annotation table not loaded")
        return len(self._table_annotation)

    def get_ids(self,
        ) -> List:
        return self._table_annotation['id'].values.tolist()

    def get_list_keys(self,
        ) -> List:
        return list(self._table_list.columns)

    def get_annotation_keys(self,
        ) -> List:
        return list(self._table_annotation.columns)

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

        if self._table_annotation is None:
            raise ValueError("Annotation table not loaded")

        if selection_key not in self._table_annotation.columns:
            raise ValueError(f"Key {selection_key} is not in the annotation table")

        subset = self._table_annotation[self._table_annotation[selection_key].isin(selection_values)]

        if subset.empty:
            raise ValueError(f"Annotation table does not contain the values for key {selection_key}")
            
        return subset.to_dict(orient='records')[0]

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
        if self._table_annotation is None:
            raise ValueError("Annotation table not loaded")
        # In order to preserve dataframe structure.
        if isinstance(loc, int) or isinstance(loc, np.integer):
            loc = [loc]
        subset = self._table_annotation.iloc[loc]
        return subset

    def get_data_selection_key(self,
    ) -> str:
        """
        Get the key used to select the data.
        Returns:
            str: key used to select the data.
        """
        return self._data_selection_key

    def set_data_selection_key(self,
        key: str
    ) -> str:
        """
        Set the key used to select the data.
        Args:
            key (str): key used to select the data.
        """
        self._data_selection_key = key

    
    def set_data_selection_key_parent(self,
        key: str
    ) -> str:
        """
        Set the key used to select the data in paretn corpus.
        Args:
            key (str): key used to select the data.
        """
        self._data_selection_key_parent = key

    def apply_cutter(self, cutter: object):
        """
        Apply a cutter.

        Args:
            cutter (object): cutter object
        """
        self._table_annotation = cutter(self._table_annotation)
        print("Applied cutter")
        print(self._table_annotation)



    def get_data_from_key(self, values: List) -> List[Dict]: # chaque méthode : deux étapes (annotation keys, et derived keys)
        if not isinstance(values, list):
            return self.get_data_from_key([values])
        if isinstance(values, list) and len(values)>1:
            return [self.get_data_from_key([val])[0] for val in values]
        value = values[0]

        if value is None:
            raise ValueError("value argument is None") 
        
        if self._table_annotation is None:
            raise ValueError("Annotation table not loaded")    
        row_subset = self._table_annotation[self._data_selection_key].isin([value])
        #test if sum(row_subset) < 1
        row = self._table_annotation[row_subset].iloc[0]
        data = row.to_dict()
        output = {}
        if self._data_annotation_getter:
            for key in self._data_output:
                if key in data.keys() and (type(self._data_output[key]).__module__ == "builtins" or self._data_output[key]=="list"):
                    val = data[key]
                    data_type = self._data_output[key]
                    if data_type == int:
                        output[key] = int(val)
                    elif data_type == float:
                        output[key] = float(val)
                    elif data_type == str:
                        output[key] = str(val)
                    elif data_type == list or data_type=="list":
                        output[key] = ast.literal_eval(val)
                    else:
                        output[key] = val
        
        for key in self._data_derived_keys:
            if 'CorpusBase' not in str(self._data_output[key].__class__):
                processor = self._data_output[key]
                args = {}
                if hasattr(self._data_output[key], 'get_param_template'):
                    paramTemplate = processor.get_param_template()
                    for argKey,argValue in paramTemplate.items():
                        args[argKey] = output[argValue]


                localOutput = processor(**args)
                if localOutput is not None:
                    output[key] = localOutput
        for key in self._data_derived_keys:
            if 'CorpusBase' in str(self._data_output[key].__class__):
                values = output[self._data_output[key]._data_selection_key_parent ]
                if isinstance(values, str) or isinstance(values, int) or isinstance(values, np.integer):
                    values = [values]
                output[key] = self._data_output[key].get_data_from_key(values)
        return [output]

    def get_data_from_idx(self, idx: slice) -> List[Dict]:
        if idx is None:
            raise ValueError("idx argument is None") 
        
        if self._table_annotation is None:
            raise ValueError("Annotation table not loaded")

        annotation_rows = self.get_annotation_from_index(idx)

        if isinstance(idx, int) or isinstance(idx, np.integer):
            idx_iterable = [idx]
        elif isinstance(idx, slice):
            idx_iterable = range(idx.start, idx.stop, idx.step)
        else:
            idx_iterable = idx

        output = []
        for _, row in annotation_rows.iterrows():
            data = row.to_dict()
            for key in self._data_output:
                if key in data.keys() and (type(self._data_output[key]).__module__ == "builtins" or self._data_output[key]=="list"):
                    val = data[key]
                    data_type = self._data_output[key]
                    if data_type == 'int':
                        data[key] = int(val)
                    elif data_type == 'float':
                        data[key] = float(val)
                    elif data_type == 'str':
                        data[key] = str(val)
                    elif data_type == 'list':
                        data[key] = ast.literal_eval(val)
                    else:
                        data[key] = val
            output.append(data)

        for key in self._data_derived_keys:
            if 'CorpusBase' not in str(self._data_output[key].__class__):
                for i, _ in enumerate(idx_iterable):
                    processor = self._data_output[key]
                    args = {}
                    paramTemplate = None
                    if hasattr(self._data_output[key], 'get_param_template'):
                        paramTemplate = processor.get_param_template()
                    if paramTemplate is  None:
                        localOutput = processor(**args)
                        if localOutput is not None:
                            output[i][key] = localOutput

        for key in self._data_derived_keys:
            if 'CorpusBase' not in str(self._data_output[key].__class__):
                for i, _ in enumerate(idx_iterable):
                    processor = self._data_output[key]
                    args = {}
                    if hasattr(self._data_output[key], 'get_param_template'):
                        paramTemplate = processor.get_param_template()
                        if paramTemplate is not None:
                            for argKey,argValue in paramTemplate.items():
                                args[argKey] = output[i][argValue]
                    localOutput = processor(**args)
                    if localOutput is not None:
                        output[i][key] = localOutput

        for key in self._data_derived_keys:
            if 'CorpusBase' in str(self._data_output[key].__class__):
                for i, index in enumerate(idx_iterable):
                    values = output[i][self._data_output[key]._data_selection_key_parent ]
                    if isinstance(values, str) or isinstance(values, int) or isinstance(values, np.integer):
                        values = [values]
                    output[i][key] = self._data_output[key].get_data_from_key(values)
        return output


    def set_data_output(self,
        data_output: Dict[str,str]
    ) -> None:

        self._data_output = data_output
        if self._data_output is not None:
            self._data_annotation_keys = [key for key in self._data_output.keys() if type(self._data_output[key]).__module__ == "builtins"]
            self._data_derived_keys = [ key for key in self._data_output.keys() if key not in self._data_annotation_keys]

            self._data_annotation_getter = None
            if self._data_annotation_keys:
                self._data_annotation_getter = itemgetter(*self._data_annotation_keys)


    def get_internals(self) -> Tuple:
        """
            Get the internal tables of the corpus.
            This method should be called after loading the corpus tables.
        """
        return self._table_list, self._table_annotation, self._table_index

    def set_internals(self, table_list: object, table_annotation: object, table_index: object) -> None:
        """
            Set the internal tables of the corpus.
        """
        self._table_list = table_list
        self._table_annotation = table_annotation
        self._table_index = table_index

    def setup(self
    ) -> None:
        """
            Setup the corpus after loading the tables.
            This method should be called after loading the corpus tables.
        """

        # TODO
        if self._root:
            subcorpora = self.get_subcorpora()

            nameToIndex = {}
            for k,corpus in enumerate(subcorpora):
                nameToIndex[corpus.get_name()] = k

            for corpus in subcorpora:
                initCorpusFromName = corpus.get_implementation().get_init_from_corpus()
                if initCorpusFromName:
                    if initCorpusFromName not in nameToIndex:
                        raise ValueError(f"Corpus {corpus.get_implementation().get_init_from_corpus()} not found in the list of subcorpora of {self._name}")
                    index = nameToIndex[corpus.get_implementation().get_init_from_corpus()]
                    tableList, tableAnnotation, tableIndex = subcorpora[index].get_implementation().get_internals()
                    corpus.get_implementation().set_internals(tableList, tableAnnotation, tableIndex)

        if self._data_output is not None:
            for key in self._data_output.keys():
                if 'mapper' in str(self._data_output[key].__class__):
                    labels = self._data_output[key].get_labels()
                    if labels is None:
                        param_template = self._data_output[key].get_param_template()
                        if param_template is not None:
                            labels = list(self._table_annotation[param_template['label']].dropna().unique())
                            self._data_output[key].set_labels(labels)

        self.apply_mappers()
        self.apply_filters()
        self.apply_cutters()
        print(self._table_annotation)

    def load(self
    ) -> None:
        """
            Load internal corpus tables from the list and annotation files.

            The list file should contain at least the columns 'annotation_file' and 'id'.
            The annotation files should contain at least the column 'id'.
            All the annotation files should have the same columns.
        """

        if self._list_filename is None:

            folder = Path(self._directory)
            file_paths = [str(p) for p in folder.iterdir() if p.is_file()]

            file_basenames = [Path(p).name for p in file_paths]

            self._table_list = pd.DataFrame({"id": file_basenames})
            self._table_annotation = pd.DataFrame({"id": file_basenames, "path": file_paths})

        else:

            filePath = os.path.join(self._directory,self._list_filename)
            self._table_list = pd.read_csv(filePath,dtype=str)
            if not {"annotation_file","id"}.issubset(self._table_list.columns):
                raise ValueError(f"The list file {filePath} should contain at least 'annotation_file' and 'id' columns")

            annotation_filenames = list(set(self._table_list['annotation_file'].values.tolist()))

            for annotation_filename in annotation_filenames:

                subset = self._table_list[self._table_list['annotation_file'].isin([annotation_filename])]
                ids = subset['id'].values.tolist()

                filePath = os.path.join(self._directory,annotation_filename)
                df = pd.read_csv(filePath,dtype=str)

                if not {"id"}.issubset(self._table_list.columns):
                    raise ValueError(f"The list file {filePath} should contain at least the 'id' column")   
    
                df = df[df['id'].isin(ids)]

                if self._table_annotation is None:
                    self._table_annotation = df
                else:
                    if df.columns.difference(self._table_annotation.columns).empty:
                        self._table_annotation = pd.concat([self._table_annotation,df],ignore_index=True)
                    else:
                        raise ValueError(f"The annotation file {filePath} should contain the same columns as the previous ones")
                

    def save(self,
             path:str
    ) -> None:
        suffix = "_path"
        cols_with_suffix = self._table_annotation.columns[self._table_annotation.columns.str.endswith(suffix)]

        for col_name in cols_with_suffix:
            self._table_annotation[col_name] = self._directory + "/" + self._table_annotation[col_name]
        os.makedirs(path, exist_ok=True)
        self._table_annotation.to_csv(path+"/annotation.csv", index=None)
        print(f"Saved annotation table to {path}/annotation.csv")
        ids = self._table_annotation["id"]
        annotation_files = ["annotation.csv"] * len(ids)
        self._table_list = pd.DataFrame({"id":ids, "annotation_file":annotation_files})
        self._table_list.to_csv(path+"/list.csv", index=None)
        print(f"Saved list table to {path}/list.csv")

class CorpusUnionImplementationPandas(CorpusImplementationPandas):
    """
    Pandas Implementation of the concatenation of several corpora.
    
    Each subcorpus must use a CorpusImplementationPandas. 
    The annotation tables are concatenated. Absolute paths are used in the output annotation table. A column is added with the original corpus id. Ids are modified, adding corpus id as suffix.

    """

    def __init__(self, **kwargs):
        """
        Initialize a union of multiple pandas corpus implementations.

        All keyword arguments are forwarded to the parent
        ``CorpusImplementationPandas.__init__``.

        Args:
            **kwargs: arguments forwarded to
                ``CorpusImplementationPandas.__init__``.
        """
        super().__init__(**kwargs)


    def set_corpus_list(self, corpus_list:List)-> None:
        """
        Set the list of subcorpora.

        Args:
            corpus_list (List): list of corpora (with a pandas implementation)
        """
        self._corpus_list = corpus_list
    
    def set_corpus_names(self, corpus_names:List[str]=None)-> None:
        """
        Set the list of corpus names

        Args:
            corpus_names (List[str]): list of corpus names
        """
        self._corpus_names = corpus_names
        if self._corpus_names is None:
            self._corpus_names = [None for _ in self._corpus_list]
    
    def set_corpus_column_name(self, corpus_column_name:str=None)-> None:
        """
        Set the corpus column name in the output corpus.

        Args:
            corpus_column_name (str): name of the corpus id column.
        """
        self._corpus_column_name = corpus_column_name

    def set_relative_path_columns(self, relative_path_columns:List[str])-> None:
        self._relative_path_columns = relative_path_columns

    def merge_annotation_tables(self):
        """
        Merge the annotation tables. 
        """
        dfs = []
        for corpus_name, corpus in zip(self._corpus_names, self._corpus_list):
            df = corpus._implementation._table_annotation
            directory = corpus._implementation._directory
            columns = list(df.columns)
            for col in columns:
                if col in self._relative_path_columns:
                    df[col] = df[col].apply(lambda path : directory+"/"+path)
            if corpus_name is not None:
                df["id"] = df["id"].apply(lambda id : id+"_"+corpus_name)
            if self._corpus_column_name is not None:
                df[self._corpus_column_name] = corpus_name
            dfs.append(df)

        self._table_annotation = pd.concat(dfs, axis=0, ignore_index=True)

    def setup(self):
        self.merge_annotation_tables()
        super().setup()


    def save(self,
             path:str
    ) -> None:
        suffix = "_path"
        cols_with_suffix = self._table_annotation.columns[self._table_annotation.columns.str.endswith(suffix)]

        for col_name in cols_with_suffix:
            self._table_annotation[col_name] = self._directory + "/" + self._table_annotation[col_name]
        os.makedirs(path, exist_ok=True)
        self._table_annotation.to_csv(path+"/annotation.csv", index=None)
        print(f"Saved annotation table to {path}/annotation.csv")
        ids = self._table_annotation["id"]
        annotation_files = [path+"/annotation.csv"] * len(ids)
        self._table_list = pd.DataFrame({"id":ids, "annotation_file":annotation_files})
        self._table_list.to_csv(path+"/list.csv", index=None)
        print(f"Saved list table to {path}/list.csv")


