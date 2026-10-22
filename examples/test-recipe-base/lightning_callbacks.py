from lightning.pytorch.callbacks import Callback

import time 
from writer import BatchDictCSVWriter
import torch 
import gc

class InferenceTimerCallback(Callback):
    """Callback to time the inference phase using on_predict_start and on_predict_end hooks."""

    def on_predict_start(self, trainer, pl_module):
        """
        Called when the predict loop begins.

        Args:
            trainer (Trainer): The Lightning trainer.
            pl_module (LightningModule): The model being used for prediction.
        """
        self.start_time = time.time()
        print(f"[Timer] Inference started at: {self.start_time}")

    def on_predict_end(self, trainer, pl_module):
        """
        Called when the predict loop ends.

        Args:
            trainer (Trainer): The Lightning trainer.
            pl_module (LightningModule): The model used for prediction.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"[Timer] Inference ended at: {end_time}")
        print(f"[Timer] Total inference time: {elapsed_time:.2f} seconds")
        
    def on_test_start(self, trainer, pl_module):
        """
        Called when the predict loop begins.

        Args:
            trainer (Trainer): The Lightning trainer.
            pl_module (LightningModule): The model being used for prediction.
        """
        self.start_time = time.time()
        print(f"[Timer] Inference started at: {self.start_time}")

    def on_test_end(self, trainer, pl_module):
        """
        Called when the predict loop ends.

        Args:
            trainer (Trainer): The Lightning trainer.
            pl_module (LightningModule): The model used for prediction.
        """
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"[Timer] Inference ended at: {end_time}")
        print(f"[Timer] Total inference time: {elapsed_time:.2f} seconds")
        
class WritePredictionsCSVCallback(Callback):
    def __init__(self, writer: BatchDictCSVWriter, get_embedding = None):
        super().__init__()
        self.writer = writer
        self.get_embedding = get_embedding
        
    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # If we want one file per GPU, write directly
        if getattr(self.writer, "per_gpu_file", False):
            self.writer(outputs)
            if batch_idx == len(trainer.predict_dataloaders) - 1:
                self.writer.finalize()
        else:
            # Synchronize all outputs across all GPUs
            gathered_outputs = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    gathered_outputs[key] = pl_module.all_gather(value)
                else:
                    gathered_outputs[key] = value  # Non-tensor, keep as-is

            # Only rank 0 writes
            if trainer.global_rank == 0:
                for key, value in gathered_outputs.items():
                    if isinstance(value, torch.Tensor):
                        gathered_outputs[key] = torch.cat([value[i] for i in range(value.size(0))], dim=0)
                self.writer(gathered_outputs)
                if batch_idx == len(trainer.predict_dataloaders) - 1:
                    self.writer.finalize()

        # Memory cleanup
        if self.get_embedding is not None:
            trainer.predict_loop._predictions = [[] for _ in range(trainer.predict_loop.num_dataloaders)]
            gc.collect()


    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # If we want one file per GPU, write directly

        if getattr(self.writer, "per_gpu_file", False):
            self.writer(outputs)
            if batch_idx == len(trainer.predict_dataloaders) - 1:
                self.writer.finalize()
        else:
            # Synchronize all outputs across all GPUs
            gathered_outputs = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    if trainer.world_size > 1:
                        gathered_outputs[key] = pl_module.all_gather(value)
                    else:
                        gathered_outputs[key] = value
                else:
                    gathered_outputs[key] = value

            if trainer.global_rank == 0:
                for key, value in gathered_outputs.items():
                    if isinstance(value, torch.Tensor):
                        # If tensor has an extra dimension (world_size > 1), concat:
                        if trainer.world_size > 1:
                            gathered_outputs[key] = torch.cat([value[i] for i in range(value.size(0))], dim=0)
                self.writer(gathered_outputs)


        # Memory cleanup
        if self.get_embedding is not None:
            trainer.predict_loop._predictions = [[] for _ in range(trainer.predict_loop.num_dataloaders)]
            gc.collect()
            
            
    def on_test_end(self, trainer, pl_module):
        if getattr(self.writer, "per_gpu_file", False) or trainer.global_rank == 0:
            self.writer.finalize()
