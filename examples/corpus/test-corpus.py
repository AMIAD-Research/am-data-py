import os
import pprint

import hydra
from omegaconf import DictConfig

# import torch

#Make a more complete trace of errors
os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None)
def test_corpus_basic(cfg: DictConfig) -> None:

    print(f"Current working directory : {os.getcwd()}")

    if "corpus_name" in cfg:
        cfg_corpus = cfg[cfg["corpus_name"]]
    else:
        cfg_corpus = cfg.corpus

    corpus = hydra.utils.instantiate(cfg_corpus)

    nElement = corpus.get_n()
    print(f"Number of elements in the corpus : {nElement}")

    print("----------------")
    print("TEST INDEX")
    for index in range(10):
        print(f"Index {index} : {corpus.get_ids()[index]}")
        output = corpus.get_data_from_idx(index)
        pprint.pp(output)


    print("----------------")
    print("TEST SLICE")
    index = slice(0, 8, 1)
    output = corpus.get_data_from_idx(index)
    pprint.pp(output)

    print("----------------")
    print("TEST KEY")
    ids = corpus.get_ids()
    for i in range(8):
        value = ids[i]
        output = corpus.get_data_from_key(value)
        pprint.pp(output)
    if "output_corpus_path" in cfg:
        corpus.save(cfg.output_corpus_path)

if __name__ == "__main__":
    
    test_corpus_basic()
