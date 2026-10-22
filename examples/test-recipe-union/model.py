import torch
from torch import nn, Tensor
import lightning.pytorch as pl
from typing import Dict, List
from collections import Counter




class DummyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Just 2 parameters to optimize
        self.param1 = nn.Parameter(torch.randn(1))
        self.param2 = nn.Parameter(torch.randn(1))

        # History per epoch (reset each epoch)
        self.current_epoch_ids = []
        # Global history (optional, to see overall evolution)
        self.global_id_history = []
        
    def forward(self, x):
        # Example: a very simple linear combination
        return x * self.param1 + self.param2

    def training_step(self, batch, batch_idx):
        inputs = batch["audio"]
        
        #print(batch)
        
        
        ids = batch["id"]   # this should be a list of strings or a tensor of numeric ids

       # Convert to Python list if it's a tensor
        if torch.is_tensor(ids):
            ids = ids.tolist()
        
        # Add to current epoch history
        self.current_epoch_ids.extend(ids)
        # Add to global history (optional)
        self.global_id_history.extend(ids)
        
        
        # Count the number of unique IDs...
        # I want to verify that my sampler is working properly
        
        # Count occurrences
        counts_binary = torch.bincount(batch["class-binary-encoded"].to("cpu").long(), minlength=2).tolist()
        counts_ternary = torch.bincount(batch["class-ternary-encoded"].to("cpu").long(), minlength=3).tolist()
        print(f"TRAIN: [RANK {self.global_rank}][Batch {batch_idx}] Binary: {counts_binary}, Ternary: {counts_ternary}")
                
                
        y_hat = self(inputs)
        loss = nn.functional.mse_loss(y_hat, inputs)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self):
        # Reset epoch history
        self.current_epoch_ids = []
        
        

    def on_train_epoch_end(self):
        rank = self.global_rank
        epoch_counter = Counter(self.current_epoch_ids)
        n_unique = len(epoch_counter)
        total = len(self.current_epoch_ids)

        print("\n" + "="*40)
        print(f"[RANK {rank}] EPOCH {self.current_epoch} — summary")
        print(f"[RANK {rank}] Unique IDs this epoch: {n_unique}")
        print(f"[RANK {rank}] Total samples this epoch: {total}")

        if n_unique > 0:
            freq_counter = Counter(epoch_counter.values())
            print(f"[RANK {rank}] Frequency distribution (per epoch):")
            for freq in sorted(freq_counter.keys()):
                n_ids = freq_counter[freq]
                pct = 100.0 * n_ids / n_unique
                print(f"[RANK {rank}]  {freq}x -> {n_ids} ids ({pct:.1f}%)")

        global_counter = Counter(self.global_id_history)
        n_unique_global = len(global_counter)
        total_global = len(self.global_id_history)
        global_freq_counter = Counter(global_counter.values())

        print(f"[RANK {rank}] --- Global ---")
        print(f"[RANK {rank}] Unique IDs: {n_unique_global}")
        print(f"[RANK {rank}] Total samples: {total_global}")
        for freq in sorted(global_freq_counter.keys()):
            n_ids = global_freq_counter[freq]
            pct = 100.0 * n_ids / n_unique_global
            print(f"[RANK {rank}]  {freq}x -> {n_ids} ids ({pct:.1f}%)")



        print("="*40 + "\n")

    
    def validation_step(self, batch, batch_idx):
        inputs = batch["audio"]
        
        counts_binary = torch.bincount(batch["class-binary-encoded"].to("cpu").long(), minlength=2).tolist()
        counts_ternary = torch.bincount(batch["class-ternary-encoded"].to("cpu").long(), minlength=3).tolist()
        print(f"VALIDATION: [RANK {self.global_rank}][Batch {batch_idx}] Binary: {counts_binary}, Ternary: {counts_ternary}")
        y_hat = self(inputs)
        loss = nn.functional.mse_loss(y_hat, inputs)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
             
        filtered_outputs = {}
        
        print("batch size", len(batch['id']))
        filtered_outputs['id'] = self.encode_utf8_tensor(batch["id"])
        filtered_outputs['audio-duration'] = batch["audio-duration"]
        filtered_outputs["mask_time"] = torch.ones(batch["audio-duration"].shape[0], device=batch["audio-duration"].device) * batch["audio-duration"].min()
        return filtered_outputs
    
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        # Basic optimizer
        return torch.optim.SGD(self.parameters(), lr=0.1)


    def encode_utf8_tensor(self, text_list: List[str], max_length: int = 300) -> Tensor:
        """
        Encode a list of strings into a 2D tensor of shape
        (batch_size, max_length) using UTF-8 encoding.
        Sequences are padded with zeros.

        We assume that the strings contain only ASCII characters
        (letters, digits, hyphens, underscores) encoded as 1 byte each.

        Args:
            text_list (List[str]): List of strings to encode.
            max_length (int): Maximum number of characters allowed per string.
                            Defaults to 300 (which corresponds to 300 bytes in UTF-8 for ASCII characters).

        Returns:
            Tensor: Tensor of shape (batch_size, max_length) of dtype torch.uint8.

        Raises:
            ValueError: If the length of a string (in characters or bytes) exceeds max_length.
        """
        encoded_tensors = []
        for s in text_list:
            # Check character length (for our alphabet, this equals the byte length)
            if len(s) > max_length:
                raise ValueError(
                    f"The filename '{s}' (length {len(s)}) exceeds the maximum allowed number of characters ({max_length})."
                )
            # Encode the string as UTF-8
            encoded = list(s.encode("utf-8"))
            if len(encoded) > max_length:
                raise ValueError(
                    f"The filename '{s}' (encoded length {len(encoded)}) exceeds the maximum allowed size of {max_length} bytes."
                )
            tensor = torch.tensor(encoded, dtype=torch.uint8)
            encoded_tensors.append(tensor)
        
        # Create a tensor of shape (batch_size, L) where L is the max length found among sequences
        padded = torch.nn.utils.rnn.pad_sequence(
            encoded_tensors, batch_first=True, padding_value=0
        )
        # Pad up to max_length if necessary
        if padded.size(1) < max_length:
            pad_size = max_length - padded.size(1)
            pad = torch.zeros((padded.size(0), pad_size), dtype=torch.uint8)
            padded = torch.cat([padded, pad], dim=1)
        return padded


    def decode_utf8_tensor(self, tensor: Tensor) -> List[str]:
        """
        Decode a 2D tensor containing UTF-8 encoded values (with 0 padding)
        into a list of strings.

        Args:
            tensor (Tensor): 2D tensor of dtype torch.uint8.

        Returns:
            List[str]: List of decoded strings.
        """
        
        
        return [
            bytes(row[row != 0].tolist()).decode("utf-8")
            for row in tensor
        ]
    
    
    @staticmethod
    def collate_fn(data,**kwargs):
            """
            """

            #print("Raw batch size:", len(data))
            if 'padding' in kwargs:
                padding = kwargs['padding']
            else:
                padding = None

            inputBatchDict = {key: [i[key] for i in data] for key in data[0]}
            outputBatchDict = {}


            for key in inputBatchDict.keys():
                
                #print("Collate key:", key)
                #print("Batch content:", inputBatchDict[key])
                #print("Types:", [type(x) for x in inputBatchDict[key]])
                #print("Shapes:", [getattr(x, "shape", None) for x in inputBatchDict[key]])

                # print(key, type(inputBatchDict[key][0]))
                if isinstance(inputBatchDict[key][0], torch.Tensor):
                    # Stack tensors along the first dimension
                    if padding is not None and key in padding:
                        tensor = torch.nn.utils.rnn.pad_sequence(inputBatchDict[key], batch_first=True, padding_value=padding[key])
                    else:
                        tensor = torch.stack(inputBatchDict[key], dim=0)
                    outputBatchDict[key] = tensor
                elif isinstance(inputBatchDict[key][0], int):
                    outputBatchDict[key] = torch.tensor(inputBatchDict[key], dtype=torch.long)
                elif isinstance(inputBatchDict[key][0], float):
                    outputBatchDict[key] = torch.tensor(inputBatchDict[key])
                else:
                    outputBatchDict[key] = inputBatchDict[key]

            #print("Raw batch size:", len(data))


            return outputBatchDict
        
        