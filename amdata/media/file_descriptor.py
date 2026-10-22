import os
from abc import ABC

class FileDescriptor(ABC):
    """
    Abstract class to access a file.

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
        """

        self._file_name = file_name
        self._file_prefix = file_prefix
        self._file_suffix = file_suffix

        # Additional components to build file suffix
        self._folder_root = None
        self._folder_feature = None
        self._folder_balancing = None
        self._file_extension = None
        self._file_basename = None

        self.update_path(file_name=file_name,file_prefix=file_prefix,file_suffix=file_suffix,
                         folder_root=folder_root,folder_feature=folder_feature,folder_balancing=folder_balancing,
                         file_extension=file_extension,file_basename=file_basename)

    def update_path(self,
                    file_prefix: str=None,
                    file_suffix: str=None,
                    folder_root: str=None,
                    folder_feature: str=None,
                    folder_balancing: str=None,
                    file_extension: str=None,
                    file_basename: str=None,
                    **kwargs
                    ) -> None:
        """"
        Update file path.

        Args:
            file_name (str, optional): path of the media file. Defaults to None.
            file_prefix (str, optional): prefix of the media file (if file_name is None). Defaults to None.
            file_suffix (str, optional): suffix of the media file (if file_name is None). Defaults to None.
            folder_feature (str, optional): root folder of the media file (if file_suffix is None). Defaults to None.
            folder_balancing (str, optional): sub folder of the media file (if file_suffix is None). Defaults to None.
            file_extension (str, optional): file extension of the media file (if file_suffix is None). Defaults to None.
            file_basename (str, optional): base name of the media file (if file_suffix is None). Defaults to None.
        """

        # Otherwise, update components of the path
        if file_prefix is not None and file_prefix != self._file_prefix:
            self._file_prefix = file_prefix

        if file_suffix is not None and file_suffix != self._file_suffix:
            self._file_suffix = file_suffix

        # If self._file_suffix is not defined, try to build it from sub-components
        if self._file_suffix is None:

            file_suffix_path = None

            if folder_root is not None and folder_root != self._folder_root:
                self._folder_root = folder_root
            if folder_feature is not None and folder_feature != self._folder_feature:
                self._folder_feature = folder_feature
            if folder_balancing is not None and folder_balancing != self._folder_balancing:
                self._folder_balancing = folder_balancing
            if file_basename is not None and file_basename != self._file_basename:
                self._file_basename = file_basename
            if file_extension is not None and file_extension != self._file_extension:
                self._file_extension = file_extension

            if self._folder_root is not None:
                file_suffix_path = self._folder_root

            if self._folder_feature is not None:
                if file_suffix_path is not None:
                    file_suffix_path =  os.path.join(file_suffix_path,self._folder_feature)
                else:
                    file_suffix_path = self._folder_feature

            if self._folder_balancing is not None:
                if file_suffix_path is not None:
                    file_suffix_path =  os.path.join(file_suffix_path,self._folder_balancing)
                else:
                    file_suffix_path = self._folder_balancing

            local_file_suffix = None
            if self._file_basename is not None and self._file_extension is not None:
                if file_suffix_path is not None:
                    local_file_suffix =  os.path.join(file_suffix_path,self._file_basename + '.' + self._file_extension)
                else:
                    local_file_suffix = self._file_basename + '.' + self._file_extension
        else:
            if folder_root is not None and folder_root != self._folder_root:
                self._folder_root = folder_root

            if self._folder_root is not None:
                local_file_suffix = os.path.join(self._folder_root,self._file_suffix)
            else:
                local_file_suffix = self._file_suffix

        # Finally, build file_name from prefix and suffix
        if self._file_prefix is not None and local_file_suffix is not None:
            self._file_name = os.path.join(self._file_prefix, local_file_suffix)
        elif local_file_suffix is not None:
            # If only suffix is given, use it as file_name
            self._file_name = local_file_suffix

    def get_file_name(self) -> str:
        return self._file_name

    def get_file_path(self) -> str:
        path = None
        if self._file_name is not None:
            path,_ = os.path.split(self._file_name)
        return path