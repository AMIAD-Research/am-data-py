from typing import Dict

from functools import partial
import importlib
import torch

from omegaconf import DictConfig

class DataLoaderBase(torch.utils.data.DataLoader):
    """
    Extended PyTorch DataLoader that wraps a ``torch.utils.data.Dataset``
    and handles configurable ``collate_fn`` resolution.

    The ``collate_fn`` argument can be passed as:
        - a **string** (deprecated): dotted path ``module.ClassName.collate_fn``.
            A ``padding`` dict and extra ``**kwargs`` are bound via ``functools.partial``.
        - a ``DictConfig`` (Hydra): contains a ``"function"`` key (dotted path).
            Remaining keys are bound via ``functools.partial``.
        - a **callable**: used directly.

    Args:
        dataset: the dataset to wrap.
        padding (Dict, optional): padding configuration passed to the collate_fn
            when it is resolved from a string or DictConfig. Defaults to None.
        batch_size (int): size of each batch. Defaults to 1.
        sampler: sampler to define the order of samples. Defaults to None.
        shuffle (bool): whether to shuffle samples. Defaults to False.
        num_workers (int): number of subprocesses for data loading. Defaults to 0.
        pin_memory (bool): whether to pin memory on GPU. Defaults to False.
        persistent_workers (bool): if True, keeps workers alive between epochs.
            Defaults to False.
        drop_last (bool): whether to drop the last incomplete batch. Defaults to False.
        collate_fn: function that merges a list of samples into a batch.
            Can be a callable, a dotted string, or a DictConfig (see class docstring).
            Defaults to None (PyTorch default).
        **kwargs: additional keyword arguments forwarded to the PyTorch DataLoader.
    """

    def __init__(self,
        dataset: torch.utils.data.Dataset,
        padding: Dict = None,
        batch_size: int = 1,
        sampler: torch.utils.data.Sampler = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        drop_last: bool = False,
        collate_fn = None,
        **kwargs
    ) -> None:

        if collate_fn is not None:
            if isinstance(collate_fn, str):
                #TODO
                # TO BE DEPRECATED
                parts = collate_fn.split('.')
                # -2 because we want to get the function from a class
                module = importlib.import_module('.'.join(parts[0:-2]))
                function = getattr(module, parts[-2]).collate_fn
                function = partial(function,padding=padding,**kwargs)
            elif isinstance(collate_fn, DictConfig):
                parts = collate_fn['function'].split('.')
                module = importlib.import_module('.'.join(parts[0:-1]))
                del collate_fn['function']
                function = getattr(module, parts[-1])
                function = partial(function,**collate_fn)
            else:
                function = collate_fn

        self._collate_fn = function
        
        if sampler is not None:
            sampler.setup(corpus=dataset._corpus)
            super().__init__(dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )
        else:
            super().__init__(dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=drop_last,
                collate_fn=self._collate_fn,
                sampler=sampler
            )
