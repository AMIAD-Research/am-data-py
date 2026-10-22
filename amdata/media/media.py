from abc import abstractmethod
from typing import Dict

from . import FileDescriptor

class Media(FileDescriptor):
    """
    Abstract class to access media.

    Attributes:
        file (str): Path of the file.
    """
    def __init__(self,
        file_name:str=None,
        file_prefix:str=None,
        file_suffix:str=None,
        folder_root: str=None,
        folder_feature: str=None,
        folder_balancing: str=None,
        file_extension: str=None,
        file_basename: str=None,
        param_template = None,
        **kwargs
    ):
        """
        Args:
            file_name (str, optional): path of the media file. Defaults to None.
            file_prefix (str, optional): prefix of the media file (if file_name is None). Defaults to None.
            file_suffix (str, optional): suffix of the media file (if file_name is None). Defaults to None.
            folder_root (str, optional): root folder of the media file (if file_suffix is None). Defaults to None.
            folder_feature (str, optional): feature folder of the media file (if file_suffix is None). Defaults to None.
            folder_balancing (str, optional): balancing folder of the media file (if file_suffix is None). Defaults to None.
            file_extension (str, optional): file extension of the media file (if file_suffix is None). Defaults to None.
            file_basename (str, optional): base name of the media file (if file_suffix is None). Defaults to None.
            param_template (Dict, optional): template used to extract arguments of the __init__ method from information of a corpus. Defaults to None.
        """

        super().__init__(file_name=file_name,file_prefix=file_prefix,file_suffix=file_suffix,
                         folder_root=folder_root,folder_feature=folder_feature,folder_balancing=folder_balancing,
                         file_extension=file_extension,file_basename=file_basename, **kwargs)

        self._param_template = param_template

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
        pass

    @abstractmethod
    def get_metadata(self):
        """
        Get information from the file (signal or metadata).
        """
        pass

    @abstractmethod
    def get_data(self) -> Dict:
        pass
