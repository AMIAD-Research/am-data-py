
from typing import Dict, Optional, List

import os

import numpy as np
import random
import soundfile as sf

from amdata.media import Media

class Audio(Media):
    """
    Access to an audio file.

    Attributes:
        sample_rate (int): sampling rate.
        n_channel (int): number of channels.
        format (str): format of the audio signal (given by `soundfile`).
        n_sample (int): number of samples.
        format_subtype (str): subtype of the audio signal (given by `soundfile`).
    """

    def __init__(self,
        file_name: Optional[str]=None,
        file_prefix: Optional[str]=None,
        file_suffix: Optional[str]=None,
        folder_root: Optional[str]=None,
        folder_feature: Optional[str]=None,
        folder_balancing: Optional[str]=None,
        file_extension: Optional[str]=None,
        file_basename: Optional[str]=None,
        start_time: Optional[float]=None,
        end_time: Optional[float]=None,
        start_index: Optional[int]=None,
        end_index: int=None,
        param_template: Dict[str,str]=None,
        sample_rate: Optional[int]= None,
        n_channel: Optional[int]= None,
        format: Optional[str]= None,
        n_sample: Optional[int]= None,
        format_subtype: Optional[str]= None,
        data: Optional[object]= None,
        **kwargs
    ) -> None :
        """
        Args:
            file_name (str, optional): path of the audio file. Defaults to None.
            file_prefix (str, optional): prefix of the audio file (if file_name is None). Defaults to None.
            file_suffix (str, optional): suffix of the audio file (if file_name is None). Defaults to None.
            folder_root (str, optional): root folder for file resolution. Defaults to None.
            folder_feature (str, optional): feature folder for file resolution. Defaults to None.
            folder_balancing (str, optional): balancing folder for file resolution. Defaults to None.
            file_extension (str, optional): expected file extension. Defaults to None.
            file_basename (str, optional): base name of the file (without extension). Defaults to None.
            start_time (float, optional): start time (in seconds), used if start_index is None. Defaults to None.
            end_time (float, optional): end time (in seconds), used if end_index is None.  Defaults to None.
            start_index (int, optional): start index (number of samples). Defaults to None.
            end_index (int, optional): end index (number of samples). Defaults to None.
            param_template (Dict, optional): template used to extract arguments of the __init__ method from
                information of a corpus. Defaults to None.
            sample_rate (int, optional): override the sample rate detected from the file. Defaults to None.
            n_channel (int, optional): override the number of channels detected from the file. Defaults to None.
            format (str, optional): override the format detected from the file. Defaults to None.
            n_sample (int, optional): override the number of samples detected from the file. Defaults to None.
            format_subtype (str, optional): override the format subtype detected from the file. Defaults to None.
            data: pre-loaded audio data (useful for testing). Defaults to None.

        Raises:
            FileNotFoundError: if the audio file cannot be found.
        """

        super().__init__(file_name=file_name,file_prefix=file_prefix,file_suffix=file_suffix,folder_root=folder_root,
                         folder_feature=folder_feature,folder_balancing=folder_balancing,file_extension=file_extension,
                         file_basename=file_basename, param_template=param_template,**kwargs)
        

        self._sample_rate = sample_rate
        self._n_channel = n_channel
        self._format = format
        self._n_sample = n_sample
        self._format_subtype = format_subtype

        self._start_time = start_time
        self._end_time = end_time
        self._start_index = start_index
        self._end_index = end_index

        self._param_template = param_template

        self._data = data

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    @property
    def n_channel(self) -> int:
        return self._n_channel
    @property
    def format(self) -> str:
        return self._format
    @property
    def n_sample(self) -> int:
        return self._n_sample
    @property
    def format_subtype(self) -> str:
        return self._format_subtype

    def update_metadata_from_file(self) -> None:
        """
        Read metadata from the file.

        Raises:
            FileNotFoundError: the audio file does not exist.
        """

        if not os.path.exists(self._file_name):
            raise FileNotFoundError(f"File {self._file_name} not found.")

        info = sf.info(self._file_name)

        self._sample_rate = info.samplerate
        self._n_channel = info.channels
        self._format = info.format
        self._n_sample = info.frames
        self._format_subtype = info.subtype

        self._set_limits(self._start_time, self._end_time, self._start_index, self._end_index)

        self._data = None

    def _set_limits(self,
        start_time: float=None,
        end_time: float=None,
        start_index: int=None,
        end_index: int=None
    ) -> None :
        """
        Set time or index limits to read the file. 

        Priority of index over time.

        Args:
            start_time (float, optional): start time (in seconds), used if start_index is None. Defaults to None.
            end_time (float, optional): end time (in seconds), used if end_index is None.  Defaults to None.
            start_index (int, optional): start index (number of samples). Defaults to None.
            end_index (int, optional): end index (number of samples). Defaults to None.
        """

        if start_time is not None:
            self._start_time = start_time
            self._start_index = int(self._start_time * self._sample_rate)
        else:
            self._start_time = 0.0
            self._start_index = 0
        
        if end_time is not None:
            self._end_time = end_time
            self._end_index = int(self._end_time * self._sample_rate)
        else:
            self._end_time = self.duration()
            self._end_index = self._n_sample-1

        if start_index is not None:
            self._start_index = start_index

        if end_index is not None:
            self._end_index = end_index


    def get_metadata(self
    ) -> Dict:
        """
        Get metadata about the audio file.

        Returns:
            Dict: dictionary with the following metadata, `sample_rate`, `n_channel`, 
            `format`, `n_sample`, `format_subtype`.
        """
        output = {}
        output['sample_rate'] = self._sample_rate
        output['n_channel'] = self._n_channel
        output['format'] = self._format
        output['n_sample'] = self._n_sample
        output['format_subtype'] = self._format_subtype
        return output

    def duration(self) -> float:
        """
        File duration (in seconds).

        Returns:
            duration (s).
        """
        return self._n_sample / self._sample_rate

    def load(self) -> None :
        
        if self._file_name is None:
            raise ValueError("File name is not set; cannot load audio.")

        self.update_metadata_from_file()

        self._data, _ = sf.read(self._file_name, dtype='float32')

    def save(self, file_name: Optional[str] = None) -> None :
        
        if file_name is None:
            if self._file_name is None:
                raise ValueError("File name is not set; cannot save audio.")
            file_name = self._file_name

        sf.write(file_name, self._data, self._sample_rate)

    def normalize(self, mode: str=None):
        """
        Normalize the signal.

        Args:
            mode (str, optional): three normalization modes are supported (max, minmax, meanstd). Defaults to None.

        Raises:
            ValueError: unknwon mode.

        """

        if mode is not None:
            if mode == 'max':
                self._data = self._data / np.max(np.abs(self._data))
            elif mode == 'minmax':
                self._data = (self._data - np.min(self._data)) / (np.max(self._data) - np.min(self._data))
            elif mode == 'meanstd':
                self._data = (self._data - np.mean(self._data)) / np.std(self._data)
            else:
                raise ValueError(f'Unknown mode: {mode}')

    def get_data(
        self,
        start_index: int=None,
        end_index: int=None,    
        start_time: float=None,
        end_time: float=None,
        output_keys: List[str]=None,
        **kwargs  
    ) -> Dict :
        """
        Get data from the audio file.

        Args:
            start_time (float, optional): start time (in seconds), used if start_index is None. Defaults to None.
            end_time (float, optional): end time (in seconds), used if end_index is None.  Defaults to None.
            start_index (int, optional): start index (number of samples). Defaults to None.
            end_index (int, optional): end index (number of samples). Defaults to None.
            output_keys (List[str], optional): list of output keys, must be a subset of signal, start_sample, end_sample, sample_rate. Defaults to None.

        Returns:
            Dictionary with the following elements:
                - signal (`np.ndarray`)
                - start_sample (`int`)
                - end_sample (`int`)
                - sample_rate (`int`)
        """
        self._set_limits(start_time, end_time, start_index, end_index)
        if self._data is None:
            with sf.SoundFile(self._file_name) as fileHandle:
                fileHandle.seek(self._start_index)
                n = self._end_index - self._start_index + 1
                self._data = fileHandle.read(n)

        audio_data = self._data
        
        output = {
            "signal": audio_data,
            "start_sample": self._start_index,
            "end_sample": self._end_index ,
            "sample_rate": self._sample_rate,
            "n_sample": self._end_index  - self._start_index,
            "duration": (self._end_index  - self._start_index) / self._sample_rate
        }

        if output_keys is not None:
            output = {key: output[key] for key in output_keys if key in output.keys()}
        return output

    def __call__(self,
        **kwargs
    ) -> Optional[Dict]:
        self.update_path(**kwargs)
        self.update_metadata_from_file()
        self.load()     
        output = self.get_data(**kwargs)
        
        return output



class AudioRandomBlock(Audio):
    """
    Access to a random block of audio from an audio file.

    Unlike Audio (which returns the full file), this class extracts a
    random chunk of configurable duration and returns that instead.

    Attributes:
      sample_rate (int): sampling rate.
      n_channel (int): number of channels.
      format (str): format of the audio signal (given by `soundfile`).
      n_sample (int): number of samples.
      format_subtype (str): subtype of the audio signal (given by `soundfile`).
    """

    def __init__(self,
      duration: float=None,
      duration_min: float=None,
      duration_max: float=None,
      transform: object=None,
      **kwargs
    ) -> None :
        """
        Args:
              duration (float, optional): fixed duration (in seconds) of the
                  block to return. Used as default when both min and max are None.
                  Defaults to None.
              duration_min (float, optional): minimum duration (in seconds) of
                  the random block. Defaults to None.
              duration_max (float, optional): maximum duration (in seconds) of
                  the random block. Defaults to None.
                  If ``duration_min`` is None it defaults to ``duration`` (or 1.0
                  if that is also None). If ``duration_max`` is None it defaults
                  to the file duration.
              transform: transformation (data augmentation) applied in get_data().
              **kwargs: passed to the parent ``Audio`` class. May include
                  ``file_name``, ``file_prefix``, ``file_suffix``,
                  ``param_template``, ``start_time``, ``end_time``,
                  ``start_index``, ``end_index`` and other ``Audio`` parameters.
        """

        super().__init__(**kwargs)

        self._duration = duration
        self._duration_min = duration_min
        self._duration_max = duration_max
        self._transform = transform


    def get_data(
        self,
        start_index: int=None,
        end_index: int=None,    
        start_time: float=None,
        end_time: float=None,
        output_keys: List[str]=None,
        **kwargs  
    ) -> Dict :
        """
        Get data from the audio file.

        Args:
            start_time (float, optional): start time (in seconds), used if start_index is None. Defaults to None.
            end_time (float, optional): end time (in seconds), used if end_index is None.  Defaults to None.
            start_index (int, optional): start index (number of samples). Defaults to None.
            end_index (int, optional): end index (number of samples). Defaults to None.
            output_keys (List[str], optional): list of output keys, must be a subset of signal, start_sample, end_sample, sample_rate. Defaults to None.

        Returns:
            Dictionary with the following elements:
                - signal (`np.ndarray`)
                - start_sample (`int`)
                - end_sample (`int`)
                - sample_rate (`int`)
        """
        self._set_limits(start_time, end_time, start_index, end_index)
        if self._data is None:
            with sf.SoundFile(self._file_name) as fileHandle:
                fileHandle.seek(self._start_index)
                n = self._end_index - self._start_index + 1
                self._data = fileHandle.read(n)
        if self._duration is not None:
            duration_index_unit = int(self._duration * self._sample_rate)
            if duration_index_unit < self._end_index - self._start_index:
                start_index = random.randint(self._start_index, self._end_index - duration_index_unit-1)
                end_index = start_index + duration_index_unit
            else:
                start_index = self._start_index
                end_index = self._end_index
        elif self._duration_min is not None and  self._duration_max is not None:
            duration = random.uniform(self._duration_min, self._duration_max)
            duration_index_unit = int(duration * self._sample_rate)
            if duration_index_unit < self._end_index - self._start_index:
                start_index = random.randint(self._start_index, self._end_index - duration_index_unit-1)
                end_index = start_index + duration_index_unit
            else:
                start_index = self._start_index
                end_index = self._end_index
        else:
            start_index = self._start_index
            end_index = self._end_index

        audio_data = self._data[start_index:end_index]
        
        output = {
            "signal": audio_data,
            "start_sample": start_index,
            "end_sample": end_index,
            "sample_rate": self._sample_rate,
            "n_sample": end_index - start_index,
            "duration": (end_index - start_index) / self._sample_rate
        }

        if self._transform is not None:
            output = self._transform(output)

        if output_keys is not None:
            output = {key: output[key] for key in output_keys if key in output.keys()}

        return output

    def __call__(self, **kwargs) -> Optional[Dict]:
        self.update_path(**kwargs)
        self.update_metadata_from_file()
        self.load()
        output = self.get_data(**kwargs)
        
        return output
