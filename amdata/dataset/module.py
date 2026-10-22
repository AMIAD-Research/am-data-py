import torch
import hydra
from functools import partial
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
import hydra
import importlib
from torch.utils.data import BatchSampler,SequentialSampler
from omegaconf import OmegaConf

from hydra.errors import InstantiationException


def _unwrap_exception(e: Exception) -> Exception:
    return e.__cause__ if isinstance(e, InstantiationException) and e.__cause__ is not None else e


class DataModule(LightningDataModule):
    """
    DataModule for PyTorch Lightning with flexible dataset, dataloader, and sampler configuration.
    
    Supports Hydra configs for datasets, dataloaders, and samplers, and handles
    collate functions with padding for audio tasks.

    Example Hydra config:

    datamodule:
      _target_: amdata.dataset.module.DataModule
      
      train_dataset_cfg: ${train_dataset}
      val_dataset_cfg: ${val_dataset}
      test_dataset_cfg: ${test_dataset}

      train_dataloader_cfg:
        batch_size: 32
        drop_last: True
        num_workers: 8
        pin_memory: true
        persistent_workers: true

      val_dataloader_cfg:
        batch_size: 32
        drop_last: False
        num_workers: 8
        pin_memory: true
        persistent_workers: true

      test_dataloader_cfg:
        batch_size: 32
        drop_last: False
        num_workers: 8
        pin_memory: true
        persistent_workers: true

      train_sampler_cfg:
        _target_: amdata.dataset.sampler.BalancedBatchSampler
        label_column: class_ternary
        batch_size: 9
        drop_last: True
        balance_mode: epoch

      val_sampler_cfg: null
      test_sampler_cfg:
        _target_: amdata.dataset.sampler.SortedAudioDescendingSampler
        drop_last: false

      collate_fn_cfg: lr2ml.model.audio.deepfake.deepfake_ssl_classifier.BinaryClassifierForDeepfakeDetection.collate_fn

      padding:
        audio: 0
    """
    def __init__(self,
                 # Dataset configs
                 train_dataset_cfg: DictConfig | dict | None = None,
                 val_dataset_cfg: DictConfig | dict | None = None,
                 test_dataset_cfg: DictConfig | dict | None = None,
                 # DataLoader configs
                 train_dataloader_cfg: DictConfig | dict | None = None,
                 val_dataloader_cfg: DictConfig | dict | None = None,
                 test_dataloader_cfg: DictConfig | dict | None = None,
                 # Sampler configs
                 train_sampler_cfg: DictConfig | dict | None = None,
                 val_sampler_cfg: DictConfig | dict | None = None,
                 test_sampler_cfg: DictConfig | dict | None = None,
                 # Collate function
                 collate_fn_cfg: str | DictConfig | None = None,
                 padding: dict | None = None):
        """
        Initialize the DataModule with dataset, dataloader, and sampler configs.

        Args:
            train_dataset_cfg (DictConfig | dict | None): Hydra config for training dataset.
            val_dataset_cfg (DictConfig | dict | None): Hydra config for validation dataset.
            test_dataset_cfg (DictConfig | dict | None): Hydra config for test dataset.
            train_dataloader_cfg (DictConfig | dict | None): Dataloader settings for training.
            val_dataloader_cfg (DictConfig | dict | None): Dataloader settings for validation.
            test_dataloader_cfg (DictConfig | dict | None): Dataloader settings for test.
            train_sampler_cfg (DictConfig | dict | callable | None): Sampler config for training.
            val_sampler_cfg (DictConfig | dict | callable | None): Sampler config for validation.
            test_sampler_cfg (DictConfig | dict | callable | None): Sampler config for test.
            collate_fn_cfg (str | DictConfig | callable | None): Collate function or config.
            padding (dict | None): Padding values for collate function (e.g., {'audio': 0}).
        """
        super().__init__()
        
        # Save configs
        self.train_dataset_cfg = train_dataset_cfg
        self.val_dataset_cfg = val_dataset_cfg
        self.test_dataset_cfg = test_dataset_cfg
        
        # Save configs
        self.train_dataloader_cfg = train_dataloader_cfg
        self.val_dataloader_cfg = val_dataloader_cfg
        self.test_dataloader_cfg = test_dataloader_cfg
        
        # Samplers
        self.train_sampler_cfg = train_sampler_cfg
        self.val_sampler_cfg = val_sampler_cfg
        self.test_sampler_cfg = test_sampler_cfg
        
        # Collate function
        self.collate_fn_cfg = collate_fn_cfg

        #TODO
        # TO BE DEPRECATED
        self.padding = padding
        
        # Will hold the actual datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Will hold the actual dataloaders (cached)
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def _setup_collate_fn(self):
        """
        Setup collate function from config.

        Returns:
            callable | None: Collate function with padding applied if specified.
        """
        if self.collate_fn_cfg is None:
            return None

        if isinstance(self.collate_fn_cfg, str):
            #TODO
            # TO BE DEPRECATED
            parts = self.collate_fn_cfg.split('.')
            module = importlib.import_module('.'.join(parts[0:-2]))
            function = getattr(module, parts[-2]).collate_fn
            return partial(function, padding=self.padding)
        elif isinstance(self.collate_fn_cfg, DictConfig):
            parts = self.collate_fn_cfg['function'].split('.')
            module = importlib.import_module('.'.join(parts[0:-1]))
            collate_cfg = dict(self.collate_fn_cfg)
            del collate_cfg['function']
            function = getattr(module, parts[-1])
            return partial(function, **collate_cfg)
        else:
            return self.collate_fn_cfg

    def setup(self, stage: str | None = None):
        """
        Instantiate datasets for a given stage.

        Args:
            stage (str | None): One of 'fit', 'test', 'predict', or None (all stages).
        """
        print(f"=== SETUP stage={stage} ===")
        
        if stage == "fit" or stage is None:
            if self.train_dataset_cfg and self.train_dataset is None:
                print("Instantiate train_dataset...")
                self.train_dataset = hydra.utils.instantiate(self.train_dataset_cfg)
                # Reset dataloader cache when dataset changes
                self._train_dataloader = None
                
            if self.val_dataset_cfg and self.val_dataset is None:
                print("Instantiate val_dataset...")
                if OmegaConf.is_list(self.val_dataset_cfg):
                    self.val_dataset = [ hydra.utils.instantiate(dset_cfg) for dset_cfg in self.val_dataset_cfg ]
                    if not OmegaConf.is_list(self.val_sampler_cfg):
                        self.val_sampler_cfg = [self.val_sampler_cfg for _ in self.val_dataset_cfg]
                else:
                    self.val_dataset = [ hydra.utils.instantiate(self.val_dataset_cfg) ]
                    if not OmegaConf.is_list(self.val_sampler_cfg):
                        self.val_sampler_cfg = [self.val_sampler_cfg for _ in self.val_dataset_cfg]
                # Reset dataloader cache when dataset changes
                self._val_dataloader = None
        
        if stage == "test" or stage is None:
            if self.test_dataset_cfg and self.test_dataset is None:
                print("Instantiate test_dataset...")
                self.test_dataset = hydra.utils.instantiate(self.test_dataset_cfg)
                # Reset dataloader cache when dataset changes
                self._test_dataloader = None
        
        if stage == "predict" or stage is None:
            # For predict, reuse test_dataset
            if self.test_dataset_cfg and self.test_dataset is None:
                print("Instantiate test_dataset pour predict...")
                self.test_dataset = hydra.utils.instantiate(self.test_dataset_cfg)
                # Reset dataloader cache when dataset changes
                self._test_dataloader = None
                
    def _create_dataloader(self, dataset: Dataset, sampler_cfg: DictConfig | dict | None = None, loader_cfg: DictConfig | dict | None = None) -> DataLoader:
        """
        Create a DataLoader for a given dataset.

        Supports both standard samplers and BatchSamplers.

        Args:
            dataset (Dataset): PyTorch Dataset instance.
            sampler_cfg (DictConfig | dict | callable | None): Sampler or BatchSampler config.
            loader_cfg (DictConfig | dict | None): Dataloader configuration (batch_size, num_workers, etc.).

        Returns:
            DataLoader: Configured PyTorch DataLoader.
        """
        # Create sampler (or batch_sampler) if configured
        sampler = None
        if sampler_cfg is not None:
            sampler = self._create_sampler(dataset, sampler_cfg)
            shuffle = False  # Can't shuffle if custom sampler provided

        if loader_cfg is None:
            loader_cfg = {}
        elif not isinstance(loader_cfg, dict):
            loader_cfg = OmegaConf.to_container(loader_cfg, resolve=True)
    
        # Detect if returned object is a BatchSampler (yields full batches)
        is_batch_sampler = False
        if sampler is not None:
            try:
                is_batch_sampler = isinstance(sampler, BatchSampler)
            except Exception:
                is_batch_sampler = False

        # Setup collate function
        collate_fn = self._setup_collate_fn()

        # Override default params avec loader_cfg
        batch_size = loader_cfg.get('batch_size', 16)
        num_workers = loader_cfg.get('num_workers', 4)
        pin_memory = loader_cfg.get('pin_memory', True)
        persistent_workers = loader_cfg.get('persistent_workers', True)
        drop_last = loader_cfg.get('drop_last', False)
        shuffle = False if is_batch_sampler else loader_cfg.get('shuffle', False)

        if is_batch_sampler:
            # When using a BatchSampler -> pass it as batch_sampler and DO NOT pass batch_size/sampler/shuffle/drop_last
            print("Passage BatchSampler")
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                collate_fn=collate_fn,
                shuffle=shuffle
            )
        else:
            # Regular sampler (per-index sampler) or None
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
    
    def _create_sampler(self, dataset: Dataset, sampler_cfg: DictConfig | dict):

        if sampler_cfg is None:
            return None

        if not isinstance(sampler_cfg, DictConfig):
            return sampler_cfg(dataset) if callable(sampler_cfg) else sampler_cfg

        target = sampler_cfg.get("_target_", "")

        if target in {"torch.utils.data.BatchSampler", "torch.utils.data.sampler.BatchSampler"}:
            base = SequentialSampler(dataset)
            cfg = OmegaConf.to_container(sampler_cfg, resolve=True)
            cfg.pop("_target_", None)
            return BatchSampler(base, **cfg)

        # multiple signatures (dataset / data_source / positional)
        attempts = [
            ("dataset_kw",      lambda: hydra.utils.instantiate(sampler_cfg, dataset=dataset)),
            ("data_source_kw",  lambda: hydra.utils.instantiate(sampler_cfg, data_source=dataset)),
            ("positional",      lambda: hydra.utils.instantiate(sampler_cfg, dataset)),
        ]

        errors = []
        for name, fn in attempts:
            try:
                return fn()
            except Exception as e:
                root = _unwrap_exception(e)
                errors.append((name, root))
                if not isinstance(root, TypeError):
                    raise

        msg = "Could not instantiate sampler. Tried:\n" + "\n".join(
            f" - {name}: {type(err).__name__}: {err}" for name, err in errors
        )
        raise TypeError(msg)

    def set_epoch_on_samplers(self, epoch: int):
        """
        Set the current epoch on any batch samplers that support it.

        Args:
            epoch (int): Current epoch number.
        """
        # S'assurer que les dataloaders sont instanciés/cachés
        for loader in (self._train_dataloader, self._val_dataloader, self._test_dataloader):
            if loader is None:
                continue
            # DataLoader expose l'attribut `batch_sampler` si c'est utilisé
            bs = getattr(loader, "batch_sampler", None)
            if bs is not None and hasattr(bs, "set_epoch"):
                try:
                    bs.set_epoch(epoch)
                except Exception as e:
                    # Ne pas crash la boucle d'entraînement si le sampler n'accepte pas l'epoch
                    print(f"[Warning] couldn't call set_epoch on sampler: {e}")
                
                
    def train_dataloader(self):
        """
        Create or return cached training DataLoader.

        Returns:
            DataLoader: Training dataloader instance.
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Did you call setup('fit')?")
        
        # Return cached dataloader if it exists
        if self._train_dataloader is not None:
            return self._train_dataloader
        
        # Create and cache the dataloader
        self._train_dataloader = self._create_dataloader(
            self.train_dataset, 
            sampler_cfg=self.train_sampler_cfg,
            loader_cfg=self.train_dataloader_cfg  
        )
        return self._train_dataloader

    def val_dataloader(self):
        """
        Create or return cached validation DataLoader.

        Returns:
            DataLoader: Validation dataloader instance.
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Did you call setup('fit')?")
        
        # Return cached dataloader if it exists
        if self._val_dataloader is not None:
            return self._val_dataloader
            
        # Create and cache the dataloader
        self._val_dataloader = [ self._create_dataloader(
            val_dset,
            sampler_cfg=self.val_sampler_cfg[idx],
            loader_cfg=self.val_dataloader_cfg  
        ) for idx, val_dset in enumerate(self.val_dataset)]
        return self._val_dataloader

    def test_dataloader(self):
        """
        Create or return cached test DataLoader.

        Returns:
            DataLoader: Test dataloader instance.
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is None. Did you call setup('test')?")
        
        # Return cached dataloader if it exists
        if self._test_dataloader is not None:
            return self._test_dataloader
            
        # Create and cache the dataloader
        self._test_dataloader = self._create_dataloader(
            self.test_dataset,
            sampler_cfg=self.test_sampler_cfg,
            loader_cfg=self.test_dataloader_cfg  
        )
        return self._test_dataloader

    def predict_dataloader(self):
        """
        Create or return cached predict DataLoader (reuses test dataloader).

        Returns:
            DataLoader: Predict dataloader instance.
        """
        return self.test_dataloader()
    
    def refresh_dataloaders(self):
        """
        Force refresh of all cached dataloaders.
        """
        print("Refreshing cached dataloaders...")
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None
    
    def teardown(self, stage: str | None = None) -> None:
        """
        Clean up resources when done with a given stage.

        Args:
            stage (str | None): One of 'fit', 'test', or None (all stages).
        """
        if stage == "fit":
            self._train_dataloader = None
            self._val_dataloader = None
        elif stage == "test":
            self._test_dataloader = None
        elif stage is None:
            # Clean everything
            self.refresh_dataloaders()