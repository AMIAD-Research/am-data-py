# README - Expected Behavior

This README explains how to interpret the printed outputs (logs) from our functional tests to verify the expected behavior without using an automated testing framework. The goal is to rely solely on the batch logs and displayed statistics.

Running the Experiment (works locally or on a server):
- create.py: Generates a dummy corpus (the number of files and their distribution can be modified directly in the script).
- run.py: Runs a training and inference on this corpus using the DummyModel class.

```Bash
# To specify a custom path for the data
export CORPUS_ROOTDIR=/tmp/dummy-data

# Create the corpus
python3 create.py
# Run the library test: training + inference
python3 run.py corpus_directory=$CORPUS_ROOTDIR
```

## TRAIN and VALIDATION Logs

**During training:**

```
TRAIN : [RANK 0][Batch 0] Binary: [25, 7], Ternary: [14, 12, 6]
TRAIN : [RANK 0][Batch 1] Binary: [24, 8], Ternary: [15, 13, 4]
TRAIN : [RANK 0][Batch 2] Binary: [19, 13], Ternary: [15, 13, 4]

VALIDATION : [RANK 0][Batch 0] Binary: [25, 7], Ternary: [14, 12, 6]
VALIDATION : [RANK 0][Batch 1] Binary: [24, 8], Ternary: [15, 13, 4]
VALIDATION : [RANK 0][Batch 2] Binary: [19, 13], Ternary: [15, 13, 4]
VALIDATION : [RANK 0][Batch 3] Binary: [3, 1], Ternary: [1, 3, 0]
```

**Interpretation:**
- Binary and Ternary indicate the number of samples per class in the batch.

## Final Statistics per Epoch

**At the end of each epoch:**
```
========================================
[RANK 0] EPOCH 0 — Summary
[RANK 0] Unique IDs this epoch: 96
[RANK 0] Total samples this epoch: 96
[RANK 0] Frequency distribution (per epoch):
[RANK 0]  1x -> 96 ids (100.0%)
[RANK 0] --- Global ---
[RANK 0] Unique IDs: 96
[RANK 0] Total samples: 96
[RANK 0]  1x -> 96 ids (100.0%)
========================================
```

**Interpretation:**
- Unique IDs this epoch: Number of distinct samples seen by the current rank.
- Total samples: Total number of samples presented (may include duplicates if batch padding is used).
- Frequency distribution: How many times each sample was seen in total.
  - Should evolve consistently with the minority class (e.g., its indices should be seen x2 by the 2nd epoch) and the majority classes, which are gradually traversed.
