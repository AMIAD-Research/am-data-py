import os
import csv
import random
import numpy as np
import soundfile as sf
import shutil


# -------------------------
# General parameters
# -------------------------
sr = 16000  # sampling rate

# Distribution commune des classes
distribution = {
    2: {"CLASS_01": 0.7, "CLASS_02": 0.3},
    3: {"CLASS_01": 0.4, "CLASS_02": 0.4, "CLASS_03": 0.2},
    4: {"CLASS_01": 0.25, "CLASS_02": 0.25, "CLASS_03": 0.25, "CLASS_04": 0.25},
    10: {f"CLASS_{i:02d}": 0.1 for i in range(1, 11)},
}

def sample_class(dist_map):
    classes = list(dist_map.keys())
    probs = list(dist_map.values())
    return random.choices(classes, probs)[0]


# -------------------------
# Corpus generation configs
# -------------------------

COL_MAP = {2: "class_binary", 3: "class_ternary", 4: "class_quaternary"}

corpus_configs = [
    {
        "suffix": "dummy-union",
        "n_file": 100,
        "min_dur": 1,
        "max_dur": 5,
        "class_prefix": "",
    },
    {
        "suffix": "dummy-union-2",
        "n_file": 50,
        "min_dur": 0.3,
        "max_dur": 4,
        "class_prefix": "other-",
    },
]


# -------------------------
# Per-corpus generation function
# -------------------------
def generate_corpus(cfg):
    suffix = cfg["suffix"]
    corpus = os.path.join(os.environ["CORPUS_ROOTDIR"], suffix)
    audio_dir = os.path.join(corpus, "audio")
    list_dir = os.path.join(corpus, "list")
    annot_dir = os.path.join(corpus, "annotation")
    list_origin = os.path.join(list_dir, "dummy.lst")
    annotation_file = os.path.join(annot_dir, "dummy.csv")

    n_file = cfg["n_file"]
    min_dur = cfg["min_dur"]
    max_dur = cfg["max_dur"]
    prefix = cfg["class_prefix"]

    # Delete the corpus if it already exists
    if os.path.exists(corpus):
        shutil.rmtree(corpus)

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(list_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    # -------------------------
    # Generate audio files + annotations
    # -------------------------
    print(f"Generating {n_file} dummy audio files for '{suffix}' into {audio_dir}...")
    rows = []

    for i in range(1, n_file + 1):
        random_letters = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
        file_id = f"{i:06d}_{random_letters}"
        fname = f"audio_{file_id}.flac"
        fpath = os.path.join(audio_dir, fname)

        dur = random.uniform(min_dur, max_dur)
        n_samples = int(dur * sr)

        sig = np.random.randn(n_samples).astype(np.float32)
        sf.write(fpath, sig, sr, format="FLAC")

        row = {
            "id": file_id,
            "audio_path": f"audio/{fname}",
            "duration": round(dur, 3),
        }

        for n_classes, dist in distribution.items():
            col_name = COL_MAP.get(n_classes, f"class_{n_classes}")
            row[f"{prefix}{col_name}"] = sample_class(dist)

        rows.append(row)

    # -------------------------
    # Save annotation CSV
    # -------------------------
    fieldnames = list(rows[0].keys())
    with open(annotation_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # -------------------------
    # Save .lst file
    # -------------------------
    with open(list_origin, "w") as f:
        f.write("annotation_file,id\n")
        for r in rows:
            f.write(f"annotation/dummy.csv,{r['id']}\n")

    # -------------------------
    # Statistics
    # -------------------------
    durations = [row["duration"] for row in rows]
    print(f"\nDummy corpus '{suffix}' generated!")
    print(f"📊 Duration statistics:")
    print(f"   Total files: {len(durations)}")
    print(f"   Min duration: {min(durations):.3f}s")
    print(f"   Max duration: {max(durations):.3f}s")
    print(f"   Mean duration: {np.mean(durations):.3f}s")
    print(f"   Std duration: {np.std(durations):.3f}s")
    print(f"📁 Files saved to: {corpus}")
    print(f"📝 Annotation file: {annotation_file}")
    print(f"📋 List file: {list_origin}")


# -------------------------
# Generation loop
# -------------------------
if __name__ == "__main__":
    for i, cfg in enumerate(corpus_configs, start=1):
        print(f"\n{'='*60}")
        print(f"Corpus {i}/{len(corpus_configs)} : {cfg['suffix']}")
        print(f"{'='*60}")
        generate_corpus(cfg)
