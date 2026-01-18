# Final-Project-for-Information-Retrieval-Course-of-BUAA
Final Project for Information Retrieval Course of BUAA

## Akkadian-English Translation with ByT5 🏛️

This repository contains the solution for the Kaggle competition [Deep Past Initiative: Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation).

The goal of this project is to translate ancient **Akkadian cuneiform transliterations** into English. The solution utilizes a **Byte-Level T5 (ByT5)** model to handle the unique challenges of ancient languages, such as broken text, rich morphology, and transliteration noise.

**🏆 Result:**
- **Public Leaderboard Score (BLEU-4):** 28.9
- **Rank:** 44 / 369 (Top 12%)

## 📖 Overview

Translating Akkadian is challenging due to:
1.  **Low Resource:** Limited parallel corpus.
2.  **Broken Text:** Input text often contains missing signs (e.g., `[...]`, `x`) due to damaged clay tablets.
3.  **Rich Morphology:** Akkadian is a highly inflected Semitic language.

### Key Approach
Instead of standard subword tokenization (BPE/WordPiece), which struggles with OOV (Out-of-Vocabulary) tokens in noisy ancient text, this solution uses **Google's ByT5 (Byte-Level T5)**. ByT5 processes text byte-by-byte, making it extremely robust to OCR errors, broken text markers, and rare proper nouns.

## 🛠️ Methodology

### 1. Model Architecture
- **Model:** `google/byt5-base`
- **Tokenizer:** None (Byte-level processing)
- **Max Sequence Length:** 512 tokens

### 2. Data Augmentation (Heuristic Alignment)
The raw dataset contains paragraph-level alignments that can be noisy. I implemented a `simple_sentence_aligner` to refine the data:
- Splits source text by newlines and target text by sentence punctuation (`.!?`).
- If the number of lines matches the number of sentences, they are aligned as individual training pairs.
- This strategy expands the dataset and improves local alignment accuracy.

### 3. Training Stability
Training ByT5 on limited hardware can be unstable. The following optimizations were applied:
- **Precision:** FP32 (Full Precision). FP16 was explicitly disabled to prevent gradient overflow (NaN loss).
- **Batching:** `per_device_train_batch_size=2` with `gradient_accumulation_steps=2` to simulate a larger batch size on limited GPU memory.

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/akkadian-translation-byt5.git](https://github.com/YOUR_USERNAME/akkadian-translation-byt5.git)
   cd akkadian-translation-byt5
   ```

2. Install the required dependencies:
   ```bash
   pip install torch transformers pandas numpy datasets evaluate scikit-learn sentence-transformers tqdm
   ```

## 🚀 Usage

### 1. Training
To fine-tune the ByT5 model on the dataset:

```bash
python train.py
```
*Note: Ensure the dataset (`train.csv`, `test.csv`) is placed in the `../data` directory (or update the `INPUT_DIR` path in `train.py`).*

### 2. Inference
To generate translations for the test set:

```bash
python inference.py
```
This script loads the trained model from `./byt5-base-akkadian2`, generates translations, and saves them to `submission2.csv`.

*Note: You may need to update the `MODEL_PATH` and `TEST_DATA_PATH` in `inference.py` to match your local file structure.*

## 📊 Performance

| Metric | Score | Note |
| :--- | :--- | :--- |
| **BLEU-4** | **28.9** | Kaggle Private Leaderboard |
| **chrF** | ~4.54 | Validation Metric during training |

## 📂 Project Structure

```
.
├── train.py           # Training script with data alignment and fine-tuning
├── inference.py       # Inference script for generating submission
├── README.md          # Project documentation
└── requirements.txt   # (Optional) List of dependencies
```

## 📜 Citation & References

- **Competition:** [Kaggle Deep Past Initiative](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)
- **Model:** [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models (Xue et al., 2021)](https://arxiv.org/abs/2105.13626)