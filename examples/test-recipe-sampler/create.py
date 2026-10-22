import os
import csv
import random
import numpy as np
import soundfile as sf
import shutil 


# -------------------------
# General parameters
# -------------------------
corpus = os.path.join(f"{os.environ['CORPUS_ROOTDIR']}", "dummysampler")
audio_dir = os.path.join(corpus, "audio")
list_dir = os.path.join(corpus, "list")
annot_dir = os.path.join(corpus, "annotation")
list_origin = os.path.join(list_dir, "dummy.lst")
annotation_file = os.path.join(annot_dir, "dummy.csv")

n_file = 400
n_speakers = 10
n_languages = 3
sr = 16000  # sampling rate
min_dur, max_dur = 1, 5  # seconds


# Remove DummyCorpus folder if it already exists
if os.path.exists(corpus):
    shutil.rmtree(corpus)


os.makedirs(audio_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)
os.makedirs(annot_dir, exist_ok=True)

# -------------------------
# Class distribution
# -------------------------
# Example: binary with 70% CLASS_01, 30% CLASS_02
distribution = {
    "speaker": dict([("speaker_"+str(n), 1/ n_speakers) for n in range(n_speakers)]),
    "language": dict([("language_"+str(n), 1/ n_speakers) for n in range(n_languages)]),
}

def sample_class(dist_map):
    classes = list(dist_map.keys())
    probs = list(dist_map.values())
    return random.choices(classes, probs)[0]

# -------------------------
# Generate audio files + annotations
# -------------------------
print(f"Generating {n_file} dummy audio files into {audio_dir}...")
rows = []

for i in range(1, n_file + 1):
    random_letters = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
    file_id = f"{i:06d}_{random_letters}"
    fname = f"audio_{file_id}.flac"
    fpath = os.path.join(audio_dir, fname)
    
    # Random duration
    dur = random.uniform(min_dur, max_dur)
    n_samples = int(dur * sr)
    
    # Random white noise signal
    sig = np.random.randn(n_samples).astype(np.float32)
    
    # Save as FLAC
    sf.write(fpath, sig, sr, format="FLAC")
    
    # Generate labels
    row = {
        "id": file_id,
        "audio_path": f"audio/{fname}",
        "duration": round(dur, 3)  # Added duration column
    }
    
    for n_classes, dist in distribution.items():
        row[n_classes] = sample_class(dist)
    
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
    f.write("annotation_file,id\n")  # header
    for r in rows:
        f.write(f"annotation/dummy.csv,{r['id']}\n")

# -------------------------
# Stats on generated durations
# -------------------------
durations = [row["duration"] for row in rows]
print(f"\nDummy dataset generated!")
print(f"📊 Duration statistics:")
print(f"   Total files: {len(durations)}")
print(f"   Min duration: {min(durations):.3f}s")
print(f"   Max duration: {max(durations):.3f}s") 
print(f"   Mean duration: {np.mean(durations):.3f}s")
print(f"   Std duration: {np.std(durations):.3f}s")
print(f"📁 Files saved to: {corpus}")
print(f"📝 Annotation file: {annotation_file}")
print(f"📋 List file: {list_origin}")