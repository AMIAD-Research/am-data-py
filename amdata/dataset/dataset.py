from typing import Dict

import importlib
import numpy as np

import torch

from amdata.corpus import Corpus


def flatten_collect(data: Dict, parent_key: str = "", sep: str = "."):
    """
    Takes as input a tree of dictionaries and outputs a flat dictionary 
    with the keys of the different levels merged as a unique str. 

    If a list is in the tree, then the list is reported at the leave level.

    It is a recursive function.

    WARNING: this function does not accept a tree with several lists at different levels.

    Args:
        data (dictionary): input dictionary
        parent_key (str, optional): prefix key coming from the parent dict. Defaults to "".
        sep (str, optional): separator of the keys in the output dictionary. Defaults to "."

    Returns:
        dictionary: output dictionary
    """
    items = {}

    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            sub = flatten_collect(v, new_key, sep)
            for sk, sv in sub.items():
                if sk in items:
                    if isinstance(items[sk], list):
                        items[sk].extend(sv if isinstance(sv, list) else [sv])
                    else:
                        items[sk] = [items[sk]] + (sv if isinstance(sv, list) else [sv])
                else:
                    items[sk] = sv

    elif isinstance(data, list):
        collected = {}
        for element in data:
            sub = flatten_collect(element, parent_key, sep)
            for sk, sv in sub.items():
                collected.setdefault(sk, [])
                if isinstance(sv, list):
                    collected[sk].extend(sv)
                else:
                    collected[sk].append(sv)
        items.update(collected)

    else:
        items[parent_key] = data

    return items


class DatasetBase(torch.utils.data.Dataset):
    """
    """
    def __init__(self,
        corpus: Corpus,
        item_output: Dict[str,str] = None,
        transform: object = None,
    ) -> None:
        """
        Args:
            corpus (Corpus): _description_
            item_output (Dict[str,str], optional): _description_. Defaults to None.
            transform (callable object, optional): takes as input a dictionary with item and transform it.
        """
        self._corpus = corpus
        self._item_output = item_output
        self._data_keys = self._corpus.get_data_keys()
        self.transform = transform

    def __len__(self):
        return self._corpus.get_n()

    def __getitem__(self, idx: int | slice):

        data = self._corpus.get_data_from_idx(idx) # because get_data returns a list of elements
        data = data[0]

        data = flatten_collect(data)

        item = {}
        for key in self._item_output.keys():
            dataKey = self._item_output[key]['data_key']
            try:
                dataValue = data[dataKey]
            except:
                print("Exception")
                print("key = ", key)
                print("data = ", data)

            outputType = self._item_output[key]['type']
            if outputType == 'torch.Tensor':

                if 'dtype' not in self._item_output[key]:
                    raise ValueError(f"Missing dtype for {key}")

                dtype = self._item_output[key]['dtype']
                parts = dtype.split('.')
                module = importlib.import_module('.'.join(parts[0:-1]))
                dtype = getattr(module, parts[-1])
                if isinstance(dataValue, list) and len(dataValue)==1:
                    dataValue = dataValue[0]
                dataValue_tensor = torch.tensor(dataValue, dtype=dtype).detach().clone()
                item[key] = dataValue_tensor
            elif outputType == 'int':
                if isinstance(dataValue, list):
                    dataValue = dataValue[0]
                item[key] = int(dataValue)
            elif outputType == 'float':
                if isinstance(dataValue, list):
                    dataValue = dataValue[0]
                item[key] = float(dataValue)
            elif outputType == 'str':
                if isinstance(dataValue, list):
                    dataValue = dataValue[0]
                item[key] = str(dataValue)
            else:
                item[key] = dataValue
        if self.transform is not None:
            item = self.transform(item)
        return item

    @staticmethod
    def collate_fn(data: list, **kwargs):

        if 'padding' in kwargs:
            padding = kwargs['padding']
        else:
            padding = None

        inputBatchDict = {key: [i[key] for i in data] for key in data[0]}
        outputBatchDict = {}

        for key in inputBatchDict.keys():
            if isinstance(inputBatchDict[key][0], torch.Tensor):
                # Stack tensors along the first dimension
                if padding is not None and key in padding:
                    tensor = torch.nn.utils.rnn.pad_sequence(inputBatchDict[key], batch_first=True, padding_value=padding[key])
                else:
                    tensor = torch.stack(inputBatchDict[key], dim=0)
                outputBatchDict[key] = tensor
            else:
                outputBatchDict[key] = inputBatchDict[key]

        return outputBatchDict
    
