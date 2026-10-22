"""
The cutter creates new lines of the corpus instead of each line.
It starts by adding a segments column with the list of tuples (start_time, end_time).
Then it reads the necessary information and produces new lines
"""

from abc import ABC
import numpy as np
import pandas

class Cutter(ABC):
    """
    Abstract class for cutters.

    A cutter is used to define new elements of a corpus.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initialize the cutter.

        Args:
            **kwargs: unused, accepted for API compatibility with concrete
                cutters (e.g. ``BlockCutter``).
        """
        pass

    def set_corpus(self, corpus: object):
        """
        Assign the parent corpus to the cutter.

        The cutter needs a reference to the corpus in order to access
        its annotation table and metadata.

        Args:
            corpus: the corpus to attach to this cutter.
        """
        self.corpus = corpus

    def transform_annotation_table(self, table_annotation: pandas.DataFrame) -> pandas.DataFrame:
      """
      Create the list of cuts

      Args:
        table_annotation (pandas.DataFrame): input annotation table
      
      Returns:
        output annotation table
      """
      return table_annotation

    def __call__(self, table_annotation: pandas.DataFrame) -> object:
        return self.transform_annotation_table(table_annotation=table_annotation)


class BlockCutter(Cutter):
    """
      Cuts  each element of the corpus into fixed duration blocks.
    """

    def __init__(self,
                 block_duration:float,
                 duration_column:str="duration",
                 id_suffix:str="_cut",
                 output_start_column:str="start_time",
                 output_end_column:str="end_time",
                 cut_index_name:str="cut_index"):
        """
        Args:
            block_duration (float): duration of a block, in seconds
            duration_column (str, optional): column name of the duration information in the input annotation table. Defaults to "duration".
            id_suffix (str, optional): suffix for the id of each block, the block_number will be added. Defaults to "_cut".
            output_start_column (str, optional): name of the start time column in the output corpus. Defaults to "start_time".
            output_end_column (str, optional): name of the end time column in the output corpus.Defaults to "end_time".
            cut_index_name (str, optional): name of the cut index column.Defaults to "cut_index".
        """
        super().__init__()
        self.block_duration = block_duration
        self.duration_column = duration_column
        self.id_suffix = id_suffix
        self.output_start_column = output_start_column
        self.output_end_column = output_end_column
        self.cut_index_name= cut_index_name

    def transform_annotation_table(self, table_annotation: pandas.DataFrame)-> pandas.DataFrame:
        """
        Apply the cutter to an annotation table

        Args:
        table_annotation (pandas.DataFrame): input annotation table
        
        Returns:
        output annotation table
        """
        table_annotation[self.duration_column] = table_annotation[self.duration_column].astype(float)
        table_annotation[self.output_start_column] = table_annotation[self.duration_column].apply(lambda d: self.start_time_list(d))
        table_annotation = table_annotation.explode(self.output_start_column)
        print(table_annotation)
        table_annotation[self.cut_index_name] = table_annotation[self.output_start_column].apply(lambda s : s/self.block_duration).astype(int).astype(str)
        table_annotation["id"] = table_annotation["id"]+ self.id_suffix + table_annotation[self.cut_index_name]
        del table_annotation[self.cut_index_name]
        table_annotation[self.output_end_column] = table_annotation[self.output_start_column] + self.block_duration
        table_annotation[self.output_end_column] = table_annotation[[self.output_end_column, self.duration_column]].min(axis=1)
        table_annotation[self.duration_column] = table_annotation[self.output_end_column] - table_annotation[self.output_start_column]
        print(table_annotation)
        return table_annotation
    
    
    def start_time_list(self, total_duration: float) -> list:
        """
        Return the multiples of block duration, inferior to total duration.
        """

        # Nombre de multiples strictement inférieurs à a
        n = int(np.floor(total_duration / self.block_duration))

        # Génère les multiples : b, 2b, ..., n*b
        return (self.block_duration * np.arange(0, n + 1)).tolist()
