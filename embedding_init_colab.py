"""
embedding_init_colab.py
=======================
Replication code for:

  "Vocabulary Transplanting for Twi: Extending Multilingual Pretrained
   Models with Language-Specific Byte Pair Encoding Tokenization"

This script reproduces Table 5 (Section 4.3) — Translation Quality by
Embedding Initialization Strategy — by running all four experimental
conditions on the Twi-English parallel corpus:

  1. NLLB-200 Baseline      — zero-shot, no transplanting, no fine-tuning
  2. Random Initialization  — new embeddings sampled from N(0, σ²)
  3. Mean Initialization    — arithmetic mean of constituent subword embeddings
  4. FOCUS Initialization   — fastText-guided cosine-similarity weighted sum

Expected results (Table 5):
  ┌──────────────────────────────────────────┬───────┬───────┐
  │ Configuration                            │  BLEU │  chrF │
  ├──────────────────────────────────────────┼───────┼───────┤
  │ NLLB-200 Baseline (no transplant)        │  5.97 │ 23.49 │
  │ NLLB-200 + Transplant (Random Init)      │  7.18 │ 23.61 │
  │ NLLB-200 + Transplant (FOCUS Init)       │  9.40 │ 26.89 │
  │ NLLB-200 + Transplant (Mean Init)        │ 12.50 │ 31.20 │
  └──────────────────────────────────────────┴───────┴───────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN IN GOOGLE COLAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Install dependencies (run once per session):

  !pip install -q transformers torch sentencepiece sacrebleu \\
               gensim tqdm accelerate pandas

Step 2 — Mount Google Drive:

  from google.colab import drive
  drive.mount('/content/drive')

Step 3 — Upload the required data files to your Google Drive at:

  /MyDrive/outputs/
    corpus.csv            ← parallel Twi-English pairs
                            columns: domain, twi, english  (or: twi, english)
    tok_twi_demo.vocab    ← Twi SentencePiece vocab file

  The corpus and vocab files are available at:
  https://data.mendeley.com/datasets/x3f8w84s7h/1

Step 4 — Upload this script to Colab and run:

  !python embedding_init_colab.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESUME SUPPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The script saves results to Drive after each strategy completes.
If the Colab session times out, simply re-run the script — it
will automatically skip already-completed strategies and resume
from where it left off.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HARDWARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tested on: Google Colab A100 (40 GB) and L4 (22 GB)
Runtime per strategy: ~8 minutes (A100) / ~15 minutes (L4)
If you encounter OOM errors, reduce BATCH_SIZE to 8.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  transformers >= 4.40
  torch        >= 2.0
  sentencepiece
  sacrebleu
  gensim       >= 4.0      (for FOCUS fastText auxiliary model)
  pandas
  tqdm
  accelerate
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import gc
import json
import shutil
import random
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import numpy as np
import torch
import pandas as pd
import sacrebleu
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ── change paths here if your Drive layout differs
# ─────────────────────────────────────────────────────────────────────────────

# Google Drive paths (source data + results output)
DRIVE_DIR     = "/content/drive/MyDrive/outputs"
DRIVE_CORPUS  = f"{DRIVE_DIR}/corpus.csv"
DRIVE_VOCAB   = f"{DRIVE_DIR}/tok_twi_demo.vocab"
DRIVE_OUT_DIR = f"{DRIVE_DIR}/embedding_eval"

# Local SSD paths (all computation happens here for speed)
LOCAL_DIR      = "/content/local_work"
CORPUS_CSV     = f"{LOCAL_DIR}/corpus.csv"
TWI_VOCAB_FILE = f"{LOCAL_DIR}/tok_twi_demo.vocab"
OUTPUT_DIR     = f"{LOCAL_DIR}/embedding_eval"
RESULTS_JSON   = f"{LOCAL_DIR}/embedding_eval_results.json"

# Model
NLLB_HUB = "facebook/nllb-200-distilled-600M"
TWI_LANG  = "twi_Latn"
EN_LANG   = "eng_Latn"

# Hyperparameters (match paper Section 3.5)
SEED         = 42
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
BATCH_SIZE   = 16     # reduce to 8 if OOM on L4
MAX_SRC_LEN  = 64
MAX_TGT_LEN  = 64
NUM_EPOCHS   = 3
WARMUP_STEPS = 200
LR           = 2e-5
NUM_BEAMS    = 4

# FOCUS hyperparameters
FOCUS_TOP_K  = 10
FOCUS_FT_DIM = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Device  : {DEVICE}")
print(f"PyTorch : {torch.__version__}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SETUP
# Copy data files from Drive to local SSD for fast I/O during training.
# Save results back to Drive after each strategy for resume support.
# ─────────────────────────────────────────────────────────────────────────────

def setup():
    """Copy corpus and vocab from Drive to local SSD."""
    os.makedirs(LOCAL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for src, dst in [(DRIVE_CORPUS, CORPUS_CSV),
                     (DRIVE_VOCAB,  TWI_VOCAB_FILE)]:
        if not os.path.exists(dst):
            print(f"  Copying {os.path.basename(src)} to local SSD ...")
            shutil.copy2(src, dst)
        else:
            print(f"  {os.path.basename(dst)} already local — skipping.")


def save_to_drive(results):
    """
    Write results JSON and checkpoint directories back to Drive.
    Called after every strategy so progress is not lost on session timeout.
    """
    os.makedirs(DRIVE_OUT_DIR, exist_ok=True)
    # Save results JSON
    drive_json = os.path.join(DRIVE_OUT_DIR, "embedding_eval_results.json")
    with open(drive_json, "w") as f:
        json.dump(results, f, indent=2)
    # Sync checkpoint directories
    for name in os.listdir(OUTPUT_DIR):
        src = os.path.join(OUTPUT_DIR, name)
        dst = os.path.join(DRIVE_OUT_DIR, name)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    print(f"  Progress saved to Drive: {DRIVE_OUT_DIR}")


def load_prior_results():
    """
    Load previously completed results from Drive for resume support.
    Returns empty dict if no prior results exist.
    """
    drive_json = os.path.join(DRIVE_OUT_DIR, "embedding_eval_results.json")
    if os.path.exists(drive_json):
        with open(drive_json) as f:
            results = json.load(f)
        print(f"  Resumed {len(results)} prior result(s) from Drive.")
        return results
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(path):
    """
    Load the Twi-English parallel corpus from CSV.
    Supports both:
      2-column layout: twi, english
      3-column layout: domain, twi, english
    Applies light English detokenization to remove spacing artifacts.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    if "twi" in df.columns and "english" in df.columns:
        twi_col, en_col = "twi", "english"
    elif len(df.columns) >= 3:
        twi_col, en_col = df.columns[1], df.columns[2]
    else:
        twi_col, en_col = df.columns[0], df.columns[1]

    df = df[[twi_col, en_col]].dropna()
    df[twi_col] = df[twi_col].str.strip()
    df[en_col]  = (df[en_col].str.strip()
                   .str.replace(r"\s([.,!?;:])", r"\1", regex=True)
                   .str.replace(r"\s(n't|'s|'re|'ve|'m|'ll|'d)\b",
                                r"\1", regex=True))
    pairs = list(zip(df[twi_col], df[en_col]))
    print(f"  Loaded {len(pairs):,} sentence pairs.")
    return pairs


def split_corpus(pairs, train_r=0.80, val_r=0.10, seed=42):
    """
    Stratified split with fixed random seed for reproducibility.
    Paper split: 80% train / 10% val / 10% test (seed=42).
    """
    rng  = random.Random(seed)
    data = list(pairs)
    rng.shuffle(data)
    n = len(data)
    t = int(n * train_r)
    v = int(n * val_r)
    return data[:t], data[t:t+v], data[t+v:]


def load_twi_vocab(path):
    """Load token list from SentencePiece .vocab file (tab-separated)."""
    tokens = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split("\t")[0]
            if tok:
                tokens.append(tok)
    return tokens


def get_new_tokens(twi_vocab, base_tokenizer):
    """
    Compute the set-difference between the Twi BPE vocabulary and
    NLLB-200's existing vocabulary. Only tokens absent from NLLB-200
    are transplanted to avoid redundancy.
    """
    nllb_vocab = set(base_tokenizer.get_vocab().keys())
    new_tokens = [t for t in twi_vocab if t not in nllb_vocab]
    print(f"  Tokens already in NLLB-200 : {len(twi_vocab) - len(new_tokens):,}")
    print(f"  New tokens to transplant   : {len(new_tokens):,}")
    return new_tokens


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EMBEDDING INITIALIZATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def _all_emb_layers(model):
    """
    Return all unique embedding layer objects in NLLB-200.
    NLLB-200 uses a shared embedding matrix across encoder, decoder,
    and the language model head. This function returns each unique
    layer object exactly once so writes are not duplicated.
    """
    seen, layers = set(), []
    candidates = [
        model.model.shared,
        getattr(model.model.encoder, "embed_tokens", None),
        getattr(model.model.decoder, "embed_tokens", None),
    ]
    for layer in candidates:
        if layer is not None and id(layer) not in seen:
            seen.add(id(layer))
            layers.append(layer)
    return layers


def _write_emb(model, idx, vec):
    """
    Write embedding vector vec to row idx across all embedding layers
    and the language model head (if weights are not tied).
    """
    for layer in _all_emb_layers(model):
        layer.weight.data[idx] = vec
    lh = getattr(model, "lm_head", None)
    if lh is not None and \
       lh.weight.data_ptr() != model.model.shared.weight.data_ptr():
        lh.weight.data[idx] = vec.clone()


# ── Strategy 1: Random Initialization ────────────────────────────────────────
def init_random(model, new_token_ids):
    """
    Random Initialization (baseline transplant strategy).

    Each new token embedding is sampled independently from:
        N(0, σ²)
    where σ² is the mean row-variance of the existing NLLB-200
    embedding matrix, ensuring the new embeddings are scaled
    consistently with the existing embedding space.

    Reference: standard practice; see Remy et al. (2024) for comparison.
    """
    W   = model.model.shared.weight.data
    std = float(W.var(dim=1).mean().sqrt())
    std = max(std, 1e-6)   # numerical safety floor
    with torch.no_grad():
        for idx in new_token_ids:
            vec = torch.zeros(W.shape[1], device=W.device).normal_(0, std)
            _write_emb(model, idx, vec)
    print("  Random initialization applied.")


# ── Strategy 2: Mean Initialization ──────────────────────────────────────────
def init_mean(model, new_token_ids, new_tokens, base_tokenizer):
    """
    Mean Initialization (recommended for low-resource settings).

    For each new Twi token t_new:
      1. Decompose t_new into constituent subword tokens using the
         original NLLB-200 tokenizer (before vocabulary extension).
      2. Assign the arithmetic mean of their NLLB-200 embedding vectors.
      3. If no constituents are found, fall back to the global mean
         of the entire NLLB-200 embedding matrix.

    This strategy places new token embeddings in a semantically
    coherent region of the existing multilingual embedding space
    without requiring any external data.

    Reference: Remy et al. (2024), recommended for low-resource settings.
    """
    W           = model.model.shared.weight.data
    global_mean = W.mean(dim=0)
    with torch.no_grad():
        for idx, token in zip(new_token_ids, new_tokens):
            surface     = token.replace("▁", " ").strip()
            constituent = base_tokenizer.encode(surface, add_special_tokens=False)
            if constituent:
                vec = W[constituent].mean(dim=0)
            else:
                vec = global_mean.clone()
            _write_emb(model, idx, vec)
    print("  Mean initialization applied.")


# ── Strategy 3: FOCUS-Inspired Initialization ─────────────────────────────────
def init_focus(model, tokenizer, new_tokens, base_tokenizer, corpus_path,
               top_k=FOCUS_TOP_K, ft_dim=FOCUS_FT_DIM):
    """
    FOCUS-Inspired Initialization.

    For each new Twi token t_new:
      1. Train a fastText CBOW model on the Twi corpus as an auxiliary
         embedding space (gensim implementation).
      2. Retrieve the fastText vector for t_new.
      3. Compute cosine similarities between t_new and all existing
         NLLB-200 tokens in the fastText space.
      4. Select the top-k most similar existing tokens.
      5. Assign a softmax-weighted sum of their NLLB-200 embeddings
         as the initialization vector for t_new.

    For tokens that are out-of-vocabulary in fastText (very rare
    surface forms), fall back to Mean initialization using constituent
    NLLB-200 subword embeddings.

    Key implementation detail: the softmax weight tensor is explicitly
    placed on the same device as the NLLB-200 embedding matrix
    (device=W.device) to avoid CPU/GPU device mismatch errors.

    Reference: Inspired by Dobler & de Melo (2023).
    """
    from gensim.models import FastText as GensimFT

    # ── Train fastText auxiliary model ────────────────────────────────────────
    print("  Training fastText auxiliary model on Twi corpus ...")
    df = pd.read_csv(corpus_path)
    df.columns = df.columns.str.strip().str.lower()
    twi_col = "twi" if "twi" in df.columns else df.columns[1]
    sentences = [str(txt).strip().split()
                 for txt in df[twi_col].dropna()]

    ft = GensimFT(
        sentences=sentences,
        vector_size=ft_dim,
        window=5,
        min_count=1,
        epochs=10,
        seed=SEED,
        workers=4,
    )
    print(f"  fastText trained on {len(sentences):,} Twi sentences.")

    # ── Pre-compute fastText matrix for all base NLLB-200 tokens ─────────────
    W          = model.model.shared.weight.data
    base_vocab = base_tokenizer.get_vocab()
    new_vocab  = tokenizer.get_vocab()
    orig_size  = len(base_tokenizer)

    print("  Building fastText similarity matrix for base vocab ...")
    base_tokens = [
        t for t, i in sorted(base_vocab.items(), key=lambda x: x[1])
        if i < orig_size
    ]

    chunk  = 5000
    ft_mat = []
    for i in range(0, len(base_tokens), chunk):
        vecs = np.array([
            ft.wv[t.replace("▁", "").strip()]
            if t.replace("▁", "").strip() in ft.wv
            else np.zeros(ft_dim)
            for t in base_tokens[i:i+chunk]
        ])
        ft_mat.append(vecs)

    ft_mat   = np.vstack(ft_mat)                                    # [V, ft_dim]
    norms    = np.linalg.norm(ft_mat, axis=1, keepdims=True) + 1e-9
    ft_mat_n = ft_mat / norms                                       # unit vectors

    global_mean = W.mean(dim=0)

    # ── Initialize each new token ─────────────────────────────────────────────
    with torch.no_grad():
        for token in tqdm(new_tokens, desc="  FOCUS init"):
            new_idx = new_vocab.get(token)
            if new_idx is None:
                continue

            surface = token.replace("▁", "").strip()

            if surface in ft.wv:
                nv   = ft.wv[surface]
                norm = np.linalg.norm(nv) + 1e-9

                if norm > 1e-6:
                    # Cosine similarity against all base tokens
                    sims     = ft_mat_n @ (nv / norm)            # [V]
                    topk_idx = np.argpartition(sims, -top_k)[-top_k:]
                    ts       = sims[topk_idx]
                    exp_s    = np.exp(ts - ts.max())

                    # IMPORTANT: device=W.device prevents CPU/GPU mismatch
                    weights  = torch.tensor(
                        exp_s / exp_s.sum(),
                        dtype=W.dtype,
                        device=W.device,       # ← must match embedding device
                    ).unsqueeze(1)             # [top_k, 1]

                    topk_nllb_ids = [base_vocab[base_tokens[i]] for i in topk_idx]
                    vec = (weights * W[topk_nllb_ids]).sum(0)
                    _write_emb(model, new_idx, vec)
                    continue

            # Fallback: Mean initialization for fastText OOV tokens
            constituent = base_tokenizer.encode(surface, add_special_tokens=False)
            valid = [i for i in constituent if i < orig_size]
            vec   = W[valid].mean(dim=0) if valid else global_mean.clone()
            _write_emb(model, new_idx, vec)

    del ft, ft_mat, ft_mat_n
    gc.collect()
    print("  FOCUS initialization applied.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TwiEnDataset(Dataset):
    """
    PyTorch Dataset for Twi-to-English sequence-to-sequence training.

    Source (Twi) is encoded with src_lang=twi_Latn.
    Target (English) is encoded by temporarily setting src_lang=eng_Latn,
    which causes the NLLB tokenizer to prepend the correct language token.
    This approach avoids the deprecated as_target_tokenizer() context manager
    and the max_target_length argument, both of which raise errors in
    recent versions of the transformers library.
    """

    def __init__(self, pairs, tokenizer, max_src=64, max_tgt=64):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_src   = max_src
        self.max_tgt   = max_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        # Encode source (Twi)
        self.tokenizer.src_lang = TWI_LANG
        enc = self.tokenizer(
            src,
            max_length=self.max_src,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Encode target (English) — temporarily set src_lang to eng_Latn
        self.tokenizer.src_lang = EN_LANG
        dec = self.tokenizer(
            tgt,
            max_length=self.max_tgt,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.tokenizer.src_lang = TWI_LANG   # restore for next call

        # Replace padding token ids in labels with -100 (ignored by loss)
        labels = dec["input_ids"].squeeze().clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(model, tokenizer, train_pairs, val_pairs, save_path):
    """
    Fine-tune model on the Twi-English training set.

    Optimizer : AdamW (weight_decay=0.01)
    Scheduler : Linear warmup then linear decay
    Epochs    : 3 (NUM_EPOCHS)
    Warmup    : 200 steps (WARMUP_STEPS)
    LR        : 2e-5
    Batch     : 16 (reduce to 8 if OOM)
    Grad clip : 1.0

    Validation loss is computed at the end of each epoch.
    The final checkpoint is saved to save_path.
    """
    train_loader = DataLoader(
        TwiEnDataset(train_pairs, tokenizer, MAX_SRC_LEN, MAX_TGT_LEN),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        TwiEnDataset(val_pairs, tokenizer, MAX_SRC_LEN, MAX_TGT_LEN),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=NUM_EPOCHS * len(train_loader),
    )

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):

        # Training pass
        model.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader,
                          desc=f"  Epoch {epoch}/{NUM_EPOCHS} [train]"):
            out = model(
                input_ids      = batch["input_ids"].to(DEVICE),
                attention_mask = batch["attention_mask"].to(DEVICE),
                labels         = batch["labels"].to(DEVICE),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            tr_loss += out.loss.item()
        tr_loss /= len(train_loader)

        # Validation pass
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"  Epoch {epoch}/{NUM_EPOCHS} [val]  "):
                out = model(
                    input_ids      = batch["input_ids"].to(DEVICE),
                    attention_mask = batch["attention_mask"].to(DEVICE),
                    labels         = batch["labels"].to(DEVICE),
                )
                va_loss += out.loss.item()
        va_loss /= len(val_loader)

        print(f"  Epoch {epoch}  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 4),
            "val_loss":   round(va_loss, 4),
        })

    # Save checkpoint
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  Checkpoint saved -> {save_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def _get_tgt_lang_id(tokenizer):
    """
    Retrieve the NLLB-200 token ID for the English language code (eng_Latn).
    Tries multiple lookup paths for robustness across transformers versions.
    """
    if hasattr(tokenizer, "lang_code_to_id"):
        lid = tokenizer.lang_code_to_id.get(EN_LANG)
        if lid is not None:
            return lid
    lid = tokenizer.convert_tokens_to_ids(EN_LANG)
    if lid != tokenizer.unk_token_id:
        return lid
    return tokenizer.get_vocab().get(EN_LANG)


def run_evaluate(model, tokenizer, test_pairs, batch_size=32):
    """
    Generate translations for the test set using beam search (num_beams=4)
    and compute corpus-level BLEU and chrF scores via SacreBLEU.

    Returns: (bleu: float, chrf: float)
    """
    model.eval()
    tokenizer.src_lang = TWI_LANG
    tgt_lang_id = _get_tgt_lang_id(tokenizer)
    hyps, refs  = [], []

    for i in tqdm(range(0, len(test_pairs), batch_size),
                  desc="  Evaluating"):
        batch = test_pairs[i:i+batch_size]
        inp   = tokenizer(
            [p[0] for p in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SRC_LEN,
        ).to(DEVICE)

        with torch.no_grad():
            out = model.generate(
                **inp,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=MAX_TGT_LEN,
                num_beams=NUM_BEAMS,
            )

        hyps.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
        refs.extend([p[1] for p in batch])

    bleu = round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)
    chrf = round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)
    return bleu, chrf


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results):
    """Print Table 5 from the paper."""
    sep = "─" * 70
    print(f"\n{'='*70}")
    print("  TABLE 5. Translation Quality by Embedding Initialization Strategy")
    print(f"{'='*70}")
    print(f"  {'Configuration':<44} {'BLEU':>6} {'chrF':>7}")
    print(sep)

    order = [
        "NLLB-200 Baseline (no transplant)",
        "NLLB-200 + Transplant (Random Init)",
        "NLLB-200 + Transplant (Mean Init)",
        "NLLB-200 + Transplant (FOCUS Init)",
    ]
    for name in order:
        r = results.get(name, {})
        if not r:
            print(f"  {name:<44} {'—':>6} {'—':>7}  [not yet run]")
        elif "error" in r:
            print(f"  {name:<44} {'ERROR':>6}")
        else:
            print(f"  {name:<44} {r['bleu']:>6} {r['chrf']:>7}")

    print(sep)
    print("  BLEU = SacreBLEU corpus BLEU (tokenize=13a)")
    print("  chrF = Character n-gram F-score (SacreBLEU default)")
    print("  All transplanted models: vocabulary size 260,878 tokens")
    print("  Fine-tuning: 3 epochs, AdamW lr=2e-5, batch=16, seed=42")
    print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── Setup ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SETUP")
    print("="*60)
    setup()

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("DATA")
    print("="*60)
    pairs = load_corpus(CORPUS_CSV)
    train_pairs, val_pairs, test_pairs = split_corpus(
        pairs, TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(f"  Train: {len(train_pairs):,}  "
          f"Val: {len(val_pairs):,}  "
          f"Test: {len(test_pairs):,}")

    # Load Twi vocabulary and compute new tokens
    twi_vocab      = load_twi_vocab(TWI_VOCAB_FILE)
    base_tokenizer = AutoTokenizer.from_pretrained(NLLB_HUB)
    new_tokens     = get_new_tokens(twi_vocab, base_tokenizer)

    # Resume from Drive if prior results exist
    results = load_prior_results()

    # ── Condition 1: Baseline (no transplanting, no fine-tuning) ─────────────
    print("\n" + "="*60)
    print("BASELINE: NLLB-200 (no transplanting, no fine-tuning)")
    print("="*60)
    label_baseline = "NLLB-200 Baseline (no transplant)"

    if label_baseline in results and "error" not in results[label_baseline]:
        print("  Already completed — skipping.")
    else:
        tok0 = AutoTokenizer.from_pretrained(NLLB_HUB)
        mdl0 = AutoModelForSeq2SeqLM.from_pretrained(NLLB_HUB).to(DEVICE)
        b0, c0 = run_evaluate(mdl0, tok0, test_pairs)
        results[label_baseline] = {"bleu": b0, "chrf": c0}
        print(f"  BLEU: {b0}   chrF: {c0}")
        del mdl0, tok0
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        save_to_drive(results)

    # ── Conditions 2–4: Three transplant strategies ───────────────────────────
    strategy_map = {
        "random": "NLLB-200 + Transplant (Random Init)",
        "mean"  : "NLLB-200 + Transplant (Mean Init)",
        "focus" : "NLLB-200 + Transplant (FOCUS Init)",
    }

    for strategy, label in strategy_map.items():
        print(f"\n{'='*60}")
        print(f"STRATEGY: {label}")
        print("="*60)

        # Skip if already successfully completed
        if label in results and "error" not in results[label]:
            print("  Already completed — skipping.")
            continue

        try:
            # Load a fresh model and tokenizer for each strategy
            tokenizer = AutoTokenizer.from_pretrained(NLLB_HUB)
            tokenizer.add_tokens(new_tokens)

            model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_HUB)
            # mean_resizing=False: we handle initialization manually below
            try:
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            except TypeError:
                model.resize_token_embeddings(len(tokenizer))
            model.to(DEVICE)

            new_ids = [tokenizer.convert_tokens_to_ids(t) for t in new_tokens]

            # Apply initialization strategy
            if strategy == "random":
                init_random(model, new_ids)

            elif strategy == "mean":
                init_mean(model, new_ids, new_tokens, base_tokenizer)

            elif strategy == "focus":
                init_focus(
                    model, tokenizer, new_tokens,
                    base_tokenizer, CORPUS_CSV,
                    top_k=FOCUS_TOP_K, ft_dim=FOCUS_FT_DIM,
                )

            # Fine-tune
            ckpt_path = os.path.join(OUTPUT_DIR, f"ckpt_{strategy}")
            history   = fine_tune(
                model, tokenizer, train_pairs, val_pairs, ckpt_path
            )

            # Evaluate on held-out test set
            bleu, chrf = run_evaluate(model, tokenizer, test_pairs)
            results[label] = {"bleu": bleu, "chrf": chrf, "history": history}
            print(f"  BLEU: {bleu}   chrF: {chrf}")

        except Exception as e:
            print(f"  ERROR in {strategy}: {e}")
            results[label] = {"error": str(e)}

        finally:
            try:
                del model, tokenizer
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Save after every strategy (enables resume on timeout)
            save_to_drive(results)

    # ── Save final results locally and print table ────────────────────────────
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal results saved locally -> {RESULTS_JSON}")

    print_results_table(results)
    print("ALL DONE")


if __name__ == "__main__":
    main()
