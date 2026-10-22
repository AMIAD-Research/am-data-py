import os
import numpy as np
from typing import Dict, Optional, List

from amdata.media import Media

from PIL import Image as PILImage
import torch
import torchvision

class Image(Media):
    """
    Access to an image file.

    Attributes:
        width (int): width of the image.
        height (int): height of the image.
        n_frame (int): number of frames (for animated images).
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
        param_template: Dict[str,str]=None,
        format: Optional[str]=None,
        mode: Optional[str]=None,
        width: Optional[int]=None,
        dpi_w: Optional[int]=None,
        dpi_h: Optional[int]=None,
        height: Optional[int]=None,
        n_channel: Optional[int]=None,
        is_animated: Optional[bool]=None,
        n_frame: Optional[int]=None,
        data: Optional[object]= None,
        normalize: bool = True,
        **kwargs
    ) -> None:
        """
        Args:
            file_name (str, optional): path of the image file. Defaults to None.
            file_prefix (str, optional): prefix of the image file (if file_name is None). Defaults to None.
            file_suffix (str, optional): suffix of the image file (if file_name is None). Defaults to None.
            folder_root (str, optional): root folder for file resolution. Defaults to None.
            folder_feature (str, optional): feature folder for file resolution. Defaults to None.
            folder_balancing (str, optional): balancing folder for file resolution. Defaults to None.
            file_extension (str, optional): expected file extension. Defaults to None.
            file_basename (str, optional): base name of the file (without extension). Defaults to None.
            param_template (Dict, optional): template used to extract arguments of the __init__ method from
                information of a corpus. Defaults to None.
            format (str, optional): image format (e.g. 'PNG', 'JPEG'). Defaults to None.
            mode (str, optional): image mode (e.g. 'L' for grayscale, 'RGB'). Defaults to None.
            width (int, optional): override the width detected from the file. Defaults to None.
            dpi_w (int, optional): horizontal DPI. Defaults to None.
            dpi_h (int, optional): vertical DPI. Defaults to None.
            height (int, optional): override the height detected from the file. Defaults to None.
            n_channel (int, optional): override the number of channels. Defaults to None.
            is_animated (bool, optional): whether the image is animated (multi-frame). Defaults to None.
            n_frame (int, optional): number of frames for animated images. Defaults to None.
            data: pre-loaded image data (PIL.Image, torch.Tensor, or np.ndarray). Defaults to None.
            normalize (bool, optional): whether to normalize pixel values to [-1, 1]. Defaults to True.
        """

        super().__init__(file_name=file_name,file_prefix=file_prefix,file_suffix=file_suffix,folder_root=folder_root,
                         folder_feature=folder_feature,folder_balancing=folder_balancing,file_extension=file_extension,
                         file_basename=file_basename, param_template=param_template,**kwargs)

        self._format = format
        self._mode = mode

        self._width = width
        self._height = height
        self._n_channel = n_channel

        self._dpi_w = dpi_w
        self._dpi_h = dpi_h
        self._is_animated = is_animated
        self._n_frame = n_frame

        self._param_template = param_template
        self._data = data  # either PIL.Image.Image or torch.Tensor or np.array, depending on pipeline

        self._normalize = normalize


    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def dpi_w(self) -> Optional[int]:
        return self._dpi_w       

    @property
    def dpi_h(self) -> Optional[int]:
        return self._dpi_h

    def update_metadata_from_file(self) -> None:

        if not self._file_name or not os.path.exists(self._file_name):
            raise FileNotFoundError(f"Image file not found: {self._file_name}")

        with PILImage.open(self._file_name) as image:
            image.load()  # ensure decoder runs so size/mode are final
            self._format = image.format
            self._mode = image.mode
            self._width, self._height = image.size
            self._n_channel = len(image.getbands())

            dpi = image.info.get("dpi")
            self._dpi_w, self._dpi_h = (dpi if dpi is not None else (None, None))

            self._is_animated = getattr(image, "is_animated", False)
            self._n_frame = getattr(image, "n_frames", 1)

        self._data = None

    def get_metadata(self) -> Dict:
        return {
            "width": self._width,
            "height": self._height,
            "n_channel": self._n_channel,
            "format": self._format,
            "dpi_w": self._dpi_w,
            "dpi_h": self._dpi_h,
            "is_animated": self._is_animated,
            "n_frame": self._n_frame,
        }

    def load(self, mode: str = "L") -> None:
        """
        Load pixels into memory as a PIL image (optionally converted to `mode`).
        Keeps a PIL.Image in self._data; normalization may later turn it into a tensor.
        """

        if self._file_name is None:
            raise ValueError("File name is not set; cannot load image.")

        self.update_metadata_from_file()

        pil_img = PILImage.open(self._file_name)
        pil_img.load()  # decode now
        if mode is not None:
            pil_img = pil_img.convert(mode)
        self._data = pil_img  # PIL.Image.Image

    def save(self, file_name: Optional[str] = None) -> None:
        """
        Save current data. If self._data is a tensor, convert to PIL first.
        """
        if file_name is None:
            if self._file_name is None:
                raise ValueError("File name is not set; cannot save image.")
            file_name = self._file_name

        if isinstance(self._data, PILImage.Image):
            self._data.save(file_name)
            return

        if torch.is_tensor(self._data):
            # Expect CxHxW or HxW; bring to PIL (uint8 0..255 if needed)
            t = self._data
            if t.dtype.is_floating_point:
                # Assume normalized ~[-1,1] or [0,1]; clamp and scale to [0,255]
                t = t.clamp(0, 1) if t.min() >= 0 else (t.add(1).div(2).clamp(0, 1))
                t = (t * 255.0).round().to(torch.uint8)
            if t.ndim == 2:
                pil_img = PILImage.fromarray(t.cpu().numpy(), mode="L")
            elif t.ndim == 3:
                if t.shape[0] in (1, 3, 4):  # C,H,W
                    pil_img = torchvision.transforms.functional.to_pil_image(t.cpu())
                else:
                    raise ValueError("Tensor must be CxHxW for color images.")
            else:
                raise ValueError("Unsupported tensor shape for saving.")
            print(f"Saving image to {file_name}")
            pil_img.save(file_name)
            return

        raise TypeError(f"Unsupported data type for save(): {type(self._data)}")

    def normalize(self) -> None:
        """
        Make self._data a torch.FloatTensor (1,H,W) normalized to [-1,1].
        Accepts PIL.Image, numpy.ndarray, or torch.Tensor.
        """
        x = self._data

        if isinstance(x, PILImage.Image):
            # ensure grayscale
            if x.mode != "L":
                x = x.convert("L")
            # to tensor in [0,1], shape (1,H,W)
            t = torchvision.transforms.functional.pil_to_tensor(x).to(torch.float32) / 255.0

        # NumPy path (zero-copy into torch when possible)
        elif isinstance(x, np.ndarray):
            arr = x
            if arr.ndim == 2:  # H,W grayscale
                t = torch.from_numpy(arr)
                if t.dtype.is_floating_point:
                    # assume already [0,1] or [0,255]? clamp later
                    pass
                else:
                    t = t.to(torch.float32) / 255.0
                t = t.unsqueeze(0)  # (1,H,W)

            elif arr.ndim == 3:  # H,W,C
                t = torch.from_numpy(arr)
                if t.dtype.is_floating_point:
                    # assume [0,1]; if not, clamp/scale before passing in
                    pass
                else:
                    t = t.to(torch.float32) / 255.0
                # HWC -> CHW
                t = t.permute(2, 0, 1).contiguous()
                # to grayscale if RGB
                if t.shape[0] == 3:
                    t = torchvision.transforms.functional.rgb_to_grayscale(t, num_output_channels=1)
                elif t.shape[0] == 1:
                    pass
                else:
                    raise ValueError(f"Unsupported channel count in ndarray: {t.shape[0]} (expected 1 or 3)")
            else:
                raise TypeError(f"Unsupported ndarray shape: {arr.shape}")

        # Torch path
        elif torch.is_tensor(x):
            t = x
            if t.ndim == 2:
                t = t.unsqueeze(0)  # (1,H,W)
            if t.dtype != torch.float32:
                # assume uint8 0..255 unless already float
                if t.dtype.is_floating_point:
                    t = t.to(torch.float32)
                else:
                    t = t.to(torch.float32) / 255.0
            # if RGB, convert to grayscale
            if t.ndim == 3 and t.shape[0] == 3:
                t = torchvision.transforms.functional.rgb_to_grayscale(t, num_output_channels=1)
        else:
            raise TypeError(f"Unsupported type for normalize(): {type(x)}")

        # Normalize to [-1, 1]
        t = torchvision.transforms.Normalize([0.5], [0.5])(t)
        self._data = t


    def get_data(self,
                 output_keys: List[str] = None,
                 **kwargs) -> Dict:
        """
        Get data from the image file.

        Returns a dict with the following keys:
            - "signal" (np.ndarray): image data as a NumPy array.
                Always returns a NumPy array, even when the underlying
                storage is a ``torch.Tensor``. Shape is (1, H, W) when
                ``normalize=True`` in the constructor, otherwise it
                matches the loaded image shape.
            - "width" (int): image width.
            - "height" (int): image height.

        Args:
            output_keys (List[str], optional): if provided, returns only
                the keys present in this list (and present in the output).
                Example: ``get_data(output_keys=["signal", "width"])``.
                Defaults to None (all keys returned).
            **kwargs: unused but accepted for API compatibility with
                other media classes (e.g. ``Audio``).

        Raises:
            ValueError: if the image data has not been loaded yet.
                Call ``load()`` or pass ``data`` to the constructor first.
            ValueError: if the underlying data is neither a
                ``torch.Tensor`` nor a ``PIL.Image.Image``.
        """
        if self._data is None:
            raise ValueError("Data not loaded; pass data to constructor or call load() before get_data().")

        if self._normalize:
            self.normalize()

        if isinstance(self._data, torch.Tensor):
            signal = self._data.numpy()  # returns np.ndarray
        elif isinstance(self._data, PILImage.Image):
            signal = np.asarray(self._data)  # returns np.ndarray
        else:
            raise ValueError("Unsupported data type")

        output = {"signal": signal, "width": self._width, "height": self._height}

        if output_keys is not None:
            output = {k: output[k] for k in output_keys if k in output}

        return output

    def __call__(self, **kwargs) -> Optional[Dict]:
        self.update_path(**kwargs)
        self.update_metadata_from_file()
        self.load()
        return self.get_data(**kwargs)
