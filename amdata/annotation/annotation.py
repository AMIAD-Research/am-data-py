
from abc import ABC, abstractmethod
from typing import Dict

import os

class Annotation(ABC):
    """
    Abstract class to access annotation files.
    """
    def __init__(self,
        file_name:str=None,
        file_prefix:str=None,
        file_suffix:str=None,
        param_template:Dict=None,
        **kwargs
    ):
        """
        Args:
            file_name (str, optional): path of the annotation file. Defaults to None.
            file_prefix (str, optional): prefix of the annotation file (if file_name is None). Defaults to None.
            file_suffix (str, optional): suffix of the annotation file (if file_name is None). Defaults to None.
            param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
        """

        self._file_name = file_name
        self._file_prefix = file_prefix
        self._file_suffix = file_suffix

        self._param_template = param_template

        if not file_name and file_prefix and file_suffix:
            self._file_name = os.path.join(self._file_prefix,self._file_suffix)

        self._data = None

    def get_param_template(self,
    ) -> Dict[str,str]:
        """
        Getter for param_template.
        """
        return self._param_template

    @abstractmethod
    def load(self):
        """
        Read file from the disk.
        """
        pass

    @abstractmethod
    def save(self):
        """
        Write to the disk.
        """

    @abstractmethod
    def get_data(self) -> Dict:
        pass

    @classmethod
    def from_table(cls, row: Dict, args) -> object:
        """
        Create an Annotation object from a table row.
        """
        kwargs = {}
        for key in args.keys():
            if args[key] in row.keys():
                kwargs[key] = row[args[key]]
            else:
                kwargs[key] = args[key]

        return cls(**kwargs)

    def __len__(self):
        """
        Get legnth of the annotation file.
        """
        return len(self._data)

