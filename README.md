# Transplanting_for_Twi

This repository contains the complete replication code for all experiments reported in the paper titled "Vocabulary Transplanting for Twi: Extending Multilingual Pretrained Models with Language-Specific Byte Pair Encoding Tokenization".


# Repository Structure
├── prepare_data.py          # Step 1: corpus preprocessing + tokenizer training //
├── embedding_init_colab.py  # Step 2: embedding initialization experiments (Colab)
└── README.md

# Data
The Twi-English parallel corpus is publicly available on Mendeley Data:
DOI: https://data.mendeley.com/datasets/x3f8w84s7h/1
Download corpus.csv from that link. The file contains 16,085 raw sentence pairs across five domains (casual, depressed, medical, toxic, agriculture) with three columns: domain, twi, english.


# Replication Instructions
Step 1 — Prepare data and train tokenizers (local machine)
Install the dependency:
pip install sentencepiece pandas

# Run the preprocessing script:
python prepare_data.py --input corpus.csv --output outputs/

Step 2 — Run embedding initialization experiments (Google Colab)
Hardware: Google Colab A100 (recommended) or L4.
Runtime: ~8 min per strategy on A100 (~32 min total).

1. Open Google Colab and select an A100 GPU runtime.
2. Install dependencies:
!pip install -q transformers torch sentencepiece sacrebleu gensim tqdm accelerate pandas

3. Mount Google Drive:
from google.colab import drive
drive.mount('/content/drive')

4. Upload embedding_init_colab.py to Colab and run:
   !python embedding_init_colab.py
5. Upload embedding_init_colab.py to Colab and run:
   !python embedding_init_colab.py

Results are saved automatically to /MyDrive/outputs/embedding_eval/embedding_eval_results.json after each strategy completes. If the session times out, re-running the script will resume from the last completed strategy.
