# README - Expected Behavior

This README describes the tests to run to test the CorpusUnion class.

**Running the experiment (works locally or on a server):**
- create.py ==> creates dummycorpus and dummycorpus2 (number of files and distribution modifiable in the script)
- run.py ==> runs a training and inference on the union of corpora using the DummyModel class and saves the corpus corresponding to the union.
- run_load.py ==> loads the saved union corpus and runs a training and inference with it. It also allows testing filters.

```Bash
# If data needs to be positioned in a specific location
export CORPUS_ROOTDIR=/tmp/dummy-data

# Creating the corpus
python3 create.py
# Running the library test: training + inference
python3 run.py corpus_directory=$CORPUS_ROOTDIR
python3 run_load.py corpus_directory=$CORPUS_ROOTDIR
```

**Example of Produced Corpus Description**
```
                      id                                         audio_path duration class_binary class_ternary class_quaternary  class_10 other-class_binary  corpus other-class_ternary other-class_quaternary other-class_10
6    000007_czudnc_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.716     CLASS_01      CLASS_02         CLASS_04  CLASS_07           CLASS_01  dummy1                 NaN                    NaN            NaN
11   000012_jcrazt_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.117     CLASS_01      CLASS_01         CLASS_02  CLASS_03           CLASS_01  dummy1                 NaN                    NaN            NaN
23   000024_ijxajg_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.759     CLASS_01      CLASS_01         CLASS_02  CLASS_03           CLASS_01  dummy1                 NaN                    NaN            NaN
45   000046_jldkif_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.153     CLASS_01      CLASS_01         CLASS_02  CLASS_03           CLASS_01  dummy1                 NaN                    NaN            NaN
50   000051_fckplc_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.257     CLASS_01      CLASS_02         CLASS_04  CLASS_01           CLASS_01  dummy1                 NaN                    NaN            NaN
51   000052_xmkwaa_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.902     CLASS_01      CLASS_01         CLASS_02  CLASS_06           CLASS_01  dummy1                 NaN                    NaN            NaN
92   000093_qijabc_dummy1  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...     2.53     CLASS_01      CLASS_02         CLASS_01  CLASS_09           CLASS_01  dummy1                 NaN                    NaN            NaN
100  000001_dxjuqr_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.344     CLASS_01      CLASS_02              NaN       NaN           CLASS_01  dummy2            CLASS_02               CLASS_02       CLASS_02
105  000006_xpbtzi_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.712     CLASS_01      CLASS_03              NaN       NaN           CLASS_01  dummy2            CLASS_03               CLASS_03       CLASS_04
113  000014_wmzkch_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.188     CLASS_01      CLASS_02              NaN       NaN           CLASS_01  dummy2            CLASS_02               CLASS_04       CLASS_10
114  000015_nywebq_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.558     CLASS_01      CLASS_02              NaN       NaN           CLASS_01  dummy2            CLASS_02               CLASS_03       CLASS_10
121  000022_xdpwan_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.022     CLASS_01      CLASS_01              NaN       NaN           CLASS_01  dummy2            CLASS_01               CLASS_01       CLASS_08
126  000027_kmgmlj_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.402     CLASS_01      CLASS_02              NaN       NaN           CLASS_01  dummy2            CLASS_02               CLASS_01       CLASS_02
135  000036_lshiue_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.848     CLASS_01      CLASS_02              NaN       NaN           CLASS_01  dummy2            CLASS_02               CLASS_03       CLASS_04
137  000038_miedqo_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.361     CLASS_01      CLASS_03              NaN       NaN           CLASS_01  dummy2            CLASS_03               CLASS_02       CLASS_03
141  000042_qzuqbj_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.189     CLASS_01      CLASS_02              NaN       NaN           CLASS_01  dummy2            CLASS_02               CLASS_02       CLASS_05
143  000044_roshuz_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.815     CLASS_01      CLASS_03              NaN       NaN           CLASS_01  dummy2            CLASS_03               CLASS_03       CLASS_03
146  000047_ndqjwq_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.273     CLASS_01      CLASS_03              NaN       NaN           CLASS_01  dummy2            CLASS_03               CLASS_02       CLASS_01
147  000048_jxzwpc_dummy2  //n/hot/b0008/jbm852/dummy-corpus//dummy-union...    2.681     CLASS_01      CLASS_01              NaN       NaN           CLASS_01  dummy2            CLASS_01               CLASS_04       CLASS_05
```
**Interpretation**
- Column *corpus* contains names of the original corpora of each id. Here *dummy1* and *dummy2*.

## Final Statistics per Epoch

**At the end of each epoch:**
```
======================================== 
[RANK 0] EPOCH 0 — summary 
[RANK 0] Unique IDs this epoch: 128 
[RANK 0] Total samples this epoch: 128 
[RANK 0] Frequency distribution (per epoch): 
[RANK 0] 1x -> 128 ids (100.0%) 
[RANK 0] --- Global --- 
[RANK 0] Unique IDs: 128 
[RANK 0] Total samples: 128 
[RANK 0] 1x -> 128 ids (100.0%)
======================================== 
```
**Interpretation:**
- Unique IDs this epoch: Number of distinct samples seen by the current rank.
- Total samples: Total number of samples presented (may include duplicates if batch is padded).
- Frequency distribution: How many times each sample was seen overall.
  - Should evolve coherently with the minority class (e.g., its indices should appear ~2x by the 2nd epoch) and gradually cover majority classes.
