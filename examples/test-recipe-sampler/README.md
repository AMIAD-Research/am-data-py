# README - Expected Behavior of Samplers

This README describes how to interpret the print outputs from our functional tests to verify that the behavior of samplers is correct, without using an automated testing framework. The idea is to rely solely on batch logs and displayed statistics.

## Running the experiment (works locally or on server):

    create.py ==> creates the dummy corpus (number of files and distribution can be modified in the script)
    run.py ==> runs training and inference on this corpus using the DummyModel class

```Bash
# If needed, set data location to a specific path
export CORPUS_ROOTDIR=/tmp/dummy-data

# Create the corpus
python3 create.py
# Run the library test: training + inference
python3 run.py corpus_directory=$CORPUS_ROOTDIR
```

**Example of Produced Corpus Description**
```
speaker	language	duration (h)	utterance number
speaker_0	language_0	0.0105436	11
speaker_0	language_1	0.0149186	15
speaker_0	language_2	0.0140686	16
speaker_1	language_0	0.0113569	12
speaker_1	language_1	0.012025	16
speaker_1	language_2	0.00754083	8
speaker_2	language_0	0.0133242	16
speaker_2	language_1	0.0149453	15
speaker_2	language_2	0.00453528	6
speaker_3	language_0	0.0152269	18
speaker_3	language_1	0.0145047	15
speaker_3	language_2	0.0086075	9
speaker_4	language_0	0.0142531	17
speaker_4	language_1	0.0133533	14
speaker_4	language_2	0.0127736	13
speaker_5	language_0	0.00812833	10
speaker_5	language_1	0.0119442	14
speaker_5	language_2	0.0114133	12
speaker_6	language_0	0.00931056	10
speaker_6	language_1	0.00735361	11

Rows per page:

1–20 sur 30

Total duration = 0.35 h Total number of utterances = 400 Total number of speaker = 10 Total number of language = 3
```

## Generated Batch Logs

Each generated batch is logged as follows:
```
TRAIN : [RANK 0][Batch 1] Speaker: [0, 6, 0, 6, 6, 6], Language: [9, 3, 12]
speaker=tensor(1) - language=tensor(0) - 3
speaker=tensor(1) - language=tensor(2) - 3
speaker=tensor(3) - language=tensor(0) - 3
speaker=tensor(3) - language=tensor(2) - 3
speaker=tensor(4) - language=tensor(0) - 3
speaker=tensor(4) - language=tensor(2) - 3
speaker=tensor(5) - language=tensor(1) - 3
speaker=tensor(5) - language=tensor(2) - 3
```

## Final Statistics per Epoch

**At the end of each epoch:**
```
========================================
[RANK 0] EPOCH 1 — summary
[RANK 0] Unique IDs this epoch: 244
[RANK 0] Total samples this epoch: 384
[RANK 0] Frequency distribution (per epoch):
[RANK 0]  1x -> 155 ids (63.5%)
[RANK 0]  2x -> 55 ids (22.5%)
[RANK 0]  3x -> 23 ids (9.4%)
[RANK 0]  4x -> 7 ids (2.9%)
[RANK 0]  5x -> 2 ids (0.8%)
[RANK 0]  6x -> 2 ids (0.8%)
[RANK 0] --- Global ---
[RANK 0] Unique IDs: 338
[RANK 0] Total samples: 768
[RANK 0]  1x -> 114 ids (33.7%)
[RANK 0]  2x -> 111 ids (32.8%)
[RANK 0]  3x -> 61 ids (18.0%)
[RANK 0]  4x -> 30 ids (8.9%)
[RANK 0]  5x -> 11 ids (3.3%)
[RANK 0]  6x -> 5 ids (1.5%)
[RANK 0]  7x -> 4 ids (1.2%)
[RANK 0]  8x -> 2 ids (0.6%)
========================================
```
**Interpretation:**
- Unique IDs this epoch: Number of distinct samples seen by the current rank.
- Total samples: Total number of samples presented (may include duplicates if batch is padded).
- Frequency distribution: How many times each sample was seen overall.
  - Should evolve coherently with the minority class (e.g., its indices should appear ~2x by the 2nd epoch) and gradually cover majority classes.
