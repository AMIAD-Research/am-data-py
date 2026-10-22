# am-data-py

This library provides a standardized way to organize, access, and manipulate corpora, making it easier to integrate with PyTorch-based pipelines.


## 📁 Corpus Structure

The library assumes a **common format** for all corpora. Each corpus must follow this directory structure:

```
corpus_root/
├── list/
│   └── *.csv          # Each file contains two columns: `id` and `annotation_file`
├── annotation/
│   └── *.csv          # Annotation files with an `id` column and additional annotation columns
├── audio/             # Optional: Audio files referenced in annotations (e.g., `audio_path`)
├── image/             # Optional: Image files referenced in annotations (e.g., `image_path`)
├── transcription/     # Optional: Transcription files referenced in annotations (e.g., `transcription_path`)
└── ...               # Other optional media directories
```

### Annotation Files
- Each annotation file is a **CSV** with:
  - An `id` column (unique identifier).
  - Additional columns for annotations.
- Annotations can be:
  - **Primitive Python types** (e.g., `int`, `float`, `str`, `bool`).
  - **File paths** to media (e.g., `audio_path`, `image_path`). Paths must end with `_path`.

---

## 🔧 Core Features

### 1. **Corpus Object**
- The `Corpus` class provides **direct access** to all elements of a corpus:
  - Lists of items.
  - Annotations.
  - Media files (audio, images, transcriptions, etc.).

### 2. **PyTorch Integration**
The library includes the following objects to seamlessly use the corpus format with PyTorch:
- **`Dataset`**: A PyTorch `Dataset` class to load and preprocess corpus data.
- **`Sampler`**: Custom samplers for advanced batching strategies.
- **`DataModule`**: A Lightning `DataModule` for easy integration with PyTorch Lightning.
- **`DataLoader`**: A PyTorch `DataLoader` class.

### 3. **Corpus Manipulation Tools**
- **`Mapper`**: Create new annotations based on existing columns (e.g., derive features from raw data).
- **`Filter`**: Apply filters to subsets of the corpus (e.g., keep only items matching a condition).
- **`Cutter`**: Split the corpus into smaller subsets (e.g., for cross-validation).

### 4. **Media Loaders**
- Dedicated objects to **load and preprocess** different media types:
  - Audio files.
  - Images.
  - Transcriptions.
  - Custom media types.

---

## 🚀 Quick Start

### Installation
```bash
pip install am-data-py
```

### Basic Usage

Basic usage examples are provided in the `examples`directory.


---

## 📜 License

This project is licensed under the **CECILL-C** license.
For more details, see the [LICENSE](LICENSE) file or visit [CECILL-C official page](https://cecill.info/licences/Licence_CECILL-C_V1-en.html).