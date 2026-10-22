import logging
import os
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from typing import List

logger = logging.getLogger(__name__)


class BatchDictCSVWriter(nn.Module):
    """
    A module to write batch data to a CSV file with buffering support.

    This class processes batch data in dictionary format, converts it into a DataFrame,
    and writes it to a CSV file. It supports buffering to reduce the frequency of file
    writes and can append to an existing CSV file.

    Currently, the writer must be embedded within a model class. However, future work
    should aim to move it to a separate module, following Lightning's modular strategy,
    and add it to the trainer with the appropriate callback. The `predict_step` in the
    model class should return an appropriate dictionary.

    Args:
        output_directory (str): Directory where the CSV will be saved.
        file_name (str): Name of the CSV file. Defaults to 'results.csv'.
        buffer_size (int): Number of batch calls to buffer before writing to the CSV. Defaults to 100.
        pad_token (str): Token used to mark padded rows in the 'id' column, which will be excluded. Defaults to "PAD_IGNORE".
        process_corpus (bool): If True, processes the 'id' column to extract CorpusFileName data. Defaults to True.
        append (bool): If True, appends to an existing CSV file instead of overwriting it. Defaults to False.

    Example:
        ```yaml
        # Example Hydra configuration (in model)
        writer:
          object:
            _target_: lr2ml.model.component.writer.BatchDictCSVWriter
            output_directory: inference
            buffer_size: 1000
            file_name: results.csv
            pad_token: PAD_IGNORE
            process_corpus: true
            append: false
        ```

        ```python
        # Example usage in Python
        writer = BatchDictCSVWriter(
            output_directory="path/to/output",
            file_name="results.csv",
            buffer_size=100,
            pad_token="PAD_IGNORE",
            process_corpus=True,
            append=False
        )

        # Simulate forward pass with batch data
        batch_data = {
            "id": ["file1", "file2", "PAD_IGNORE"],
            "feature1": [0.1, 0.2, 0.0],
            "feature2": [0.3, 0.4, 0.0]
        }
        writer.forward(batch_data)
        writer.finalize()
        ```
    """
    def __init__(self, output_directory: str, file_name: str = 'results.csv', buffer_size: int = 100, 
                 pad_token: str = "PAD_IGNORE", process_corpus: bool = True, append: bool = False, **kwargs):
        """
        Initializes the BatchDictCSVWriter with specified parameters.

        Args:
            output_directory (str): Directory where the CSV will be saved.
            file_name (str): Name of the CSV file. Defaults to 'results.csv'.
            buffer_size (int): Number of batch calls to buffer before writing to the CSV. Defaults to 100.
            pad_token (str): Token used to mark padded rows in the 'id' column, which will be excluded. Defaults to "PAD_IGNORE".
            process_corpus (bool): If True, processes the 'id' column to extract CorpusFileName data. Defaults to True.
            append (bool): If True, appends to an existing CSV file instead of overwriting it. Defaults to False.
        """
        super().__init__()
        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)
        self.csv_file_path = os.path.join(self.output_directory, file_name)
        self.buffer_size = buffer_size
        self.buffer = []  # Buffer of DataFrames
        self.header_written = False
        self.pad_token = pad_token
        self.process_corpus = process_corpus
        self.append = append

        # Check if the file already exists and adjust the write flag
        self.header_written = append and os.path.exists(self.csv_file_path)
        
        print(f"[BatchDictCSVWriter] Initialized with config: {vars(self)}\n")
        
    def _process_batch(self, data_dict: dict) -> pd.DataFrame:
        """
        Converts a batch dictionary (containing tensors or lists) into a DataFrame.
        Each key becomes a column and each element (of batch size B) becomes a row.

        Args:
            data_dict (dict): Batch data with keys as column names and values as tensors or lists.

        Returns:
            pd.DataFrame: A DataFrame representing the batch.
        """
        processed = {}
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                if key == "id":  
                    # Decode BEFORE converting to list
                    processed[key] = self.decode_utf8_tensor(value)
                else:
                    processed[key] = value.cpu().tolist()
            elif isinstance(value, list):
                processed[key] = value
            else:
                processed[key] = value  # Handle other types if needed
        return pd.DataFrame(processed)        
        
    def finalize(self):
        """
        Final processing: flush remaining data and update the CSV file.
        """
        self.flush()
        self._post_process_csv()

    def flush(self):
        """
        Concatenates the buffered DataFrames, filters out padded rows based on the pad_token in the 'id' column,
        and writes the result to the CSV file.
        
        Behavior:
        - If not in append mode (append=False):
            * On the first flush, the file is overwritten (mode 'w') even if it exists, with header written.
            * On subsequent flushes, data is appended (mode 'a') without header.
        - If in append mode (append=True):
            * All flushes are done in append mode (mode 'a').
            * On the first flush, a message "Writing in append mode to <file_name>..." is displayed.
        
        After flushing, the internal buffer is cleared.
        """
        if not self.buffer:
            return

        df_all = pd.concat(self.buffer, ignore_index=True)

        # Filter out rows with the pad_token in the 'id' column
        if 'id' in df_all.columns:
            df_all = df_all[df_all['id'] != self.pad_token]

        # Determine mode and whether header needs to be written
        if self.append:
            mode = 'a'
            if not self.header_written:
                print(f"Writing in append mode to {self.csv_file_path}...")
            header = not self.header_written  # Write header only on first flush
        else:
            # In non-append mode, first flush overwrites the file, then appends
            if not self.header_written:
                mode = 'w'
                header = True
            else:
                mode = 'a'
                header = False

        df_all.to_csv(self.csv_file_path, mode=mode, header=header, index=False)
        self.header_written = True  # Header is now written
        self.buffer = []


    def forward(self, data_dict: dict):
        """
        Adds a batch (in the form of a dictionary) to the buffer.
        When the number of buffered batches reaches buffer_size, the buffer is written to the CSV file.

        Args:
            data_dict (dict): Batch data as a dictionary.
        """
        df = self._process_batch(data_dict)
        self.buffer.append(df)
        if len(self.buffer) >= self.buffer_size:
            self.flush()
        
    def _post_process_csv(self):
        """
        Reads the final CSV file, removes padded rows, and adds extracted features.
        """
        if not os.path.exists(self.csv_file_path):
            print(f"CSV file {self.csv_file_path} not found, skipping post-processing.")
            return

        df = pd.read_csv(self.csv_file_path)

        # Remove padded rows
        df = df[df["id"] != self.pad_token].copy()
        df = df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

        # Extract filename features
        #if self.process_corpus:
        #    features_df = df["id"].apply(self._extract_filename_features).apply(pd.Series)
        #    df = pd.concat([df, features_df], axis=1)
            
        df.to_csv(self.csv_file_path, index=False)
        
        
    def encode_utf8_tensor(self, text_list: List[str], max_length: int = 300) -> Tensor:
        """
        Encode a list of strings into a 2D tensor of shape
        (batch_size, max_length) using UTF-8 encoding.
        Sequences are padded with zeros.

        We assume that the strings contain only ASCII characters
        (letters, digits, hyphens, underscores) encoded on 1 byte each.

        Args:
            text_list (List[str]): List of strings to encode.
            max_length (int): Maximum number of characters allowed for each string.
                            Default is 300 (which corresponds to 300 bytes in UTF-8 for ASCII characters).

        Returns:
            Tensor: Tensor of shape (batch_size, max_length) of dtype torch.uint8.

        Raises:
            ValueError: If the length of a string (in characters or bytes) exceeds max_length.
        """
        encoded_tensors = []
        for s in text_list:
            # Check the length in characters (for our alphabet, this is equal to the length in bytes)
            if len(s) > max_length:
                raise ValueError(
                    f"The filename '{s}' (length {len(s)}) exceeds the maximum number of allowed characters ({max_length})."
                )
            # Encode the string in UTF-8
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
    