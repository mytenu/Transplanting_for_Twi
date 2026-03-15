"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Install dependency (once):
  pip install sentencepiece pandas

  # Run (corpus.csv must be in the same directory, or pass --input):
  python prepare_data.py

  # Or specify custom paths:
  python prepare_data.py --input /path/to/corpus.csv --output /path/to/outputs/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  corpus.csv is available at:
  https://data.mendeley.com/datasets/x3f8w84s7h/1

  Expected columns (any order):
    domain   — thematic domain label (casual/depressed/medical/toxic/agriculture)
    twi      — Twi source sentence
    english  — English translation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  sentencepiece >= 0.1.99
  pandas        >= 1.3
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import sentencepiece as spm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (matches paper Section 3.1 and 3.2 exactly)
# ─────────────────────────────────────────────────────────────────────────────

# Tokenizer hyperparameters (Section 3.2)
VOCAB_SIZE        = 8_000      # vocabulary size for both Twi and English tokenizers
CHARACTER_COVERAGE = 0.9995    # retains rare diacritical and tonal Twi characters
SPM_MODEL_TYPE    = "bpe"      # Byte Pair Encoding algorithm (Sennrich et al., 2016)
SPM_PAD_ID        = 0
SPM_UNK_ID        = 1
SPM_BOS_ID        = 2
SPM_EOS_ID        = 3

# Output filenames
TWI_CORPUS_FILE    = "tw_corpus.txt"
EN_CORPUS_FILE     = "en_corpus.txt"
TWI_MODEL_PREFIX   = "tok_twi_demo"
EN_MODEL_PREFIX    = "tok_english_demo"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD CORPUS
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus(csv_path: Path) -> pd.DataFrame:
    """
    Load corpus.csv and normalise column names.

    Accepts both:
      3-column layout: domain, twi, english
      2-column layout: twi, english

    Returns a DataFrame with columns ['domain', 'twi', 'english']
    (domain is set to 'unknown' if not present in the source file).
    """
    print(f"\n{'='*60}")
    print("STEP 1 — Loading corpus")
    print(f"{'='*60}")
    print(f"  Source: {csv_path}")

    if not csv_path.exists():
        print(f"\n  ERROR: File not found: {csv_path}")
        print("  Download corpus.csv from:")
        print("  https://data.mendeley.com/datasets/x3f8w84s7h/1")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    print(f"  Raw rows      : {len(df):,}")
    print(f"  Columns found : {list(df.columns)}")

    # Normalise to standard column names
    if "twi" in df.columns and "english" in df.columns:
        if "domain" not in df.columns:
            df["domain"] = "unknown"
        df = df[["domain", "twi", "english"]]

    elif len(df.columns) >= 3:
        # Assume layout: domain, twi, english
        df.columns = ["domain", "twi", "english"] + list(df.columns[3:])
        df = df[["domain", "twi", "english"]]

    elif len(df.columns) == 2:
        df.columns = ["twi", "english"]
        df["domain"] = "unknown"
        df = df[["domain", "twi", "english"]]

    else:
        print(f"  ERROR: Cannot determine column layout from {list(df.columns)}")
        sys.exit(1)

    # Strip whitespace
    df["twi"]     = df["twi"].astype(str).str.strip()
    df["english"] = df["english"].astype(str).str.strip()
    df["domain"]  = df["domain"].astype(str).str.strip().str.lower()

    print(f"  Loaded        : {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — DEDUPLICATE
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate sentence pairs.

    Deduplication is performed on the (twi, english) pair jointly,
    so a Twi sentence with multiple valid English translations is retained.
    (Paper Section 3.1: "filtering pipeline that removed duplicate entries")

    Expected result: 16,083 pairs (from 16,085 raw pairs — 2 duplicates removed).
    """
    print(f"\n{'='*60}")
    print("STEP 2 — Deduplication")
    print(f"{'='*60}")

    n_before = len(df)
    df = df.drop_duplicates(subset=["twi", "english"]).reset_index(drop=True)
    n_after  = len(df)
    removed  = n_before - n_after

    print(f"  Before : {n_before:,} pairs")
    print(f"  After  : {n_after:,} pairs")
    print(f"  Removed: {removed:,} duplicate(s)")

    if n_after != 16_083:
        print(f"  NOTE: Expected 16,083 pairs after deduplication; "
              f"got {n_after:,}. This may reflect a different corpus version.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — EXTRACT MONOLINGUAL SUB-CORPORA
# ─────────────────────────────────────────────────────────────────────────────

def extract_monolingual(df: pd.DataFrame, output_dir: Path):
    """
    Split the parallel corpus into monolingual Twi and English plain-text files.

    Each file contains one sentence per line with no header.
    These files are the direct inputs to SentencePiece training.

    (Paper Section 3.1: "corpus was split into monolingual Twi and monolingual
    English sub-corpora, each saved as plain-text files with one sentence per line")

    Outputs:
      tw_corpus.txt   — monolingual Twi sentences
      en_corpus.txt   — monolingual English sentences
    """
    print(f"\n{'='*60}")
    print("STEP 3 — Extracting monolingual sub-corpora")
    print(f"{'='*60}")

    twi_path = output_dir / TWI_CORPUS_FILE
    en_path  = output_dir / EN_CORPUS_FILE

    twi_sentences = df["twi"].tolist()
    en_sentences  = df["english"].tolist()

    with open(twi_path, "w", encoding="utf-8") as f:
        f.write("\n".join(twi_sentences) + "\n")

    with open(en_path, "w", encoding="utf-8") as f:
        f.write("\n".join(en_sentences) + "\n")

    print(f"  Twi corpus  -> {twi_path}  ({len(twi_sentences):,} sentences)")
    print(f"  En  corpus  -> {en_path}   ({len(en_sentences):,} sentences)")

    return twi_path, en_path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 & 5 — TRAIN SENTENCEPIECE BPE TOKENIZERS
# ─────────────────────────────────────────────────────────────────────────────

def train_tokenizer(
    corpus_path: Path,
    model_prefix: str,
    output_dir: Path,
    vocab_size: int      = VOCAB_SIZE,
    character_coverage: float = CHARACTER_COVERAGE,
    model_type: str      = SPM_MODEL_TYPE,
):
    """
    Train a SentencePiece BPE tokenizer on a monolingual plain-text corpus.

    Hyperparameters (Section 3.2):
      vocab_size         = 8,000   (balances coverage vs. overfitting)
      character_coverage = 0.9995  (retains rare Twi diacritics and tonal marks)
      model_type         = "bpe"   (Byte Pair Encoding, Sennrich et al. 2016)

    Outputs:
      <model_prefix>.model   — SentencePiece binary model file
      <model_prefix>.vocab   — vocabulary file with log-probability scores

    The .model file is used directly for tokenization.
    The .vocab file is used for:
      - Fertility analysis (Section 3.3)
      - Vocabulary transplanting set-difference computation (Section 3.4)
    """
    model_path = output_dir / f"{model_prefix}.model"
    vocab_path = output_dir / f"{model_prefix}.vocab"

    # SentencePiece writes to the current directory by default;
    # we specify the full prefix path to write directly to output_dir
    full_prefix = str(output_dir / model_prefix)

    print(f"  Training on  : {corpus_path}")
    print(f"  Vocab size   : {vocab_size:,}")
    print(f"  Char coverage: {character_coverage}")
    print(f"  Algorithm    : {model_type.upper()}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=full_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=SPM_PAD_ID,
        unk_id=SPM_UNK_ID,
        bos_id=SPM_BOS_ID,
        eos_id=SPM_EOS_ID,
        # Retain whitespace prefix token (▁) used by NLLB-200 and other
        # SentencePiece models for space-aware subword boundary marking
        add_dummy_prefix=True,
    )

    print(f"  Model saved  -> {model_path}")
    print(f"  Vocab saved  -> {vocab_path}")

    return model_path, vocab_path


def train_both_tokenizers(twi_path: Path, en_path: Path, output_dir: Path):
    """
    Train Twi and English BPE tokenizers under identical hyperparameter
    settings, as described in Section 3.2.

    The English tokenizer serves as a controlled baseline for fertility
    comparison (Table 3 of the paper).
    """
    print(f"\n{'='*60}")
    print("STEP 4 — Training Twi BPE tokenizer")
    print(f"{'='*60}")
    twi_model, twi_vocab = train_tokenizer(
        corpus_path=twi_path,
        model_prefix=TWI_MODEL_PREFIX,
        output_dir=output_dir,
    )

    print(f"\n{'='*60}")
    print("STEP 5 — Training English BPE tokenizer (fertility baseline)")
    print(f"{'='*60}")
    en_model, en_vocab = train_tokenizer(
        corpus_path=en_path,
        model_prefix=EN_MODEL_PREFIX,
        output_dir=output_dir,
    )

    return twi_model, twi_vocab, en_model, en_vocab


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — CORPUS STATISTICS  (reproduces Table 2 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def corpus_statistics(df: pd.DataFrame):
    """
    Compute and print corpus-level statistics matching Table 2 of the paper.

    Statistics reported:
      - Total sentence pairs
      - Total tokens per language (whitespace-delimited)
      - Vocabulary size (unique surface forms) per language
      - Type-Token Ratio (TTR) per language
      - Min/Max/Mean/Std sentence length per language
      - Hapax legomena percentage per language
      - Shared surface-form tokens (Twi ∩ English)
      - Sentence-length Pearson correlation
    """
    print(f"\n{'='*60}")
    print("STEP 6 — Corpus Statistics  (Table 2)")
    print(f"{'='*60}")

    twi_sentences = df["twi"].tolist()
    en_sentences  = df["english"].tolist()

    def tokenize(sentences):
        return [s.split() for s in sentences]

    twi_tokens = tokenize(twi_sentences)
    en_tokens  = tokenize(en_sentences)

    twi_flat  = [t for s in twi_tokens for t in s]
    en_flat   = [t for s in en_tokens  for t in s]
    twi_vocab = set(twi_flat)
    en_vocab  = set(en_flat)

    twi_lengths = [len(s) for s in twi_tokens]
    en_lengths  = [len(s) for s in en_tokens]

    # Pearson correlation of sentence lengths
    import statistics
    n = len(twi_lengths)
    twi_mean = statistics.mean(twi_lengths)
    en_mean  = statistics.mean(en_lengths)
    twi_std  = statistics.stdev(twi_lengths)
    en_std   = statistics.stdev(en_lengths)
    cov      = sum((twi_lengths[i] - twi_mean) * (en_lengths[i] - en_mean)
                   for i in range(n)) / (n - 1)
    pearson_r = cov / (twi_std * en_std)

    # Hapax legomena (words appearing exactly once)
    twi_counts  = Counter(twi_flat)
    en_counts   = Counter(en_flat)
    twi_hapax   = sum(1 for c in twi_counts.values() if c == 1)
    en_hapax    = sum(1 for c in en_counts.values()  if c == 1)
    shared      = len(twi_vocab & en_vocab)

    sep = "─" * 56
    print(f"\n  {'Statistic':<42} {'Twi':>6} {'English':>8}")
    print(f"  {sep}")

    rows = [
        ("Total Sentence Pairs",          f"{len(df):,}",           f"{len(df):,}"),
        ("Total Tokens",                  f"{len(twi_flat):,}",     f"{len(en_flat):,}"),
        ("Vocabulary Size (Unique Forms)", f"{len(twi_vocab):,}",   f"{len(en_vocab):,}"),
        ("Type-Token Ratio (TTR)",
         f"{len(twi_vocab)/len(twi_flat):.4f}",
         f"{len(en_vocab)/len(en_flat):.4f}"),
        ("Min Sentence Length (tokens)",  f"{min(twi_lengths)}",    f"{min(en_lengths)}"),
        ("Max Sentence Length (tokens)",  f"{max(twi_lengths)}",    f"{max(en_lengths)}"),
        ("Mean Sentence Length (tokens)",
         f"{statistics.mean(twi_lengths):.2f}",
         f"{statistics.mean(en_lengths):.2f}"),
        ("Std Dev Sentence Length",
         f"{twi_std:.2f}",
         f"{en_std:.2f}"),
        ("Hapax Legomena (% of vocab)",
         f"{twi_hapax/len(twi_vocab)*100:.1f}%",
         f"{en_hapax/len(en_vocab)*100:.1f}%"),
        ("Shared Surface-Form Tokens",    f"{shared:,}",            f"{shared:,}"),
        ("Sentence-Length Correlation r", f"{pearson_r:.4f}",       f"{pearson_r:.4f}"),
    ]

    for label, twi_val, en_val in rows:
        print(f"  {label:<42} {twi_val:>6} {en_val:>8}")

    print(f"\n  Domain distribution:")
    for domain, count in df["domain"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {domain:<20} {count:>6,}  ({pct:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — VERIFY OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def verify_outputs(
    twi_model_path: Path,
    twi_vocab_path: Path,
    en_model_path: Path,
    en_vocab_path: Path,
    twi_corpus_path: Path,
    en_corpus_path: Path,
):
    """
    Load the trained tokenizers and verify:
      1. Vocabulary sizes match paper (8,000 tokens each)
      2. Both .model files load without error
      3. Sample tokenization of a Twi sentence
      4. Vocabulary file format is correct (tab-separated token + log-prob)
    """
    print(f"\n{'='*60}")
    print("STEP 7 — Verifying outputs")
    print(f"{'='*60}")

    # Load and verify Twi tokenizer
    sp_twi = spm.SentencePieceProcessor()
    sp_twi.Load(str(twi_model_path))
    twi_vocab_size = sp_twi.GetPieceSize()
    print(f"  Twi tokenizer vocab size  : {twi_vocab_size:,}  "
          f"{'✓' if twi_vocab_size == VOCAB_SIZE else '✗ expected 8000'}")

    # Load and verify English tokenizer
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load(str(en_model_path))
    en_vocab_size = sp_en.GetPieceSize()
    print(f"  English tokenizer vocab   : {en_vocab_size:,}  "
          f"{'✓' if en_vocab_size == VOCAB_SIZE else '✗ expected 8000'}")

    # Verify corpus line counts
    with open(twi_corpus_path, encoding="utf-8") as f:
        twi_lines = sum(1 for line in f if line.strip())
    with open(en_corpus_path, encoding="utf-8") as f:
        en_lines = sum(1 for line in f if line.strip())
    print(f"  Twi corpus lines          : {twi_lines:,}  "
          f"{'✓' if twi_lines == 16083 else f'NOTE: expected 16,083'}")
    print(f"  English corpus lines      : {en_lines:,}  "
          f"{'✓' if en_lines == 16083 else f'NOTE: expected 16,083'}")

    # Verify vocab file format
    with open(twi_vocab_path, encoding="utf-8") as f:
        first_line = f.readline().strip()
    parts = first_line.split("\t")
    print(f"  Vocab file format         : "
          f"{'✓ tab-separated (token, log-prob)' if len(parts) == 2 else '✗ unexpected format'}")

    # Sample tokenization
    sample_twi = "Mepa wo kyɛw, me yɛ ɔyarefo"
    tokens = sp_twi.EncodeAsPieces(sample_twi)
    print(f"\n  Sample Twi tokenization:")
    print(f"    Input  : {sample_twi}")
    print(f"    Tokens : {tokens}")
    print(f"    Count  : {len(tokens)} tokens for "
          f"{len(sample_twi.split())} words  "
          f"(fertility = {len(tokens)/len(sample_twi.split()):.2f})")

    # Count new tokens (set-difference with NLLB-200 — requires transformers)
    try:
        from transformers import AutoTokenizer
        print(f"\n  Computing set-difference with NLLB-200 vocabulary ...")
        nllb_tok    = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        nllb_vocab  = set(nllb_tok.get_vocab().keys())

        twi_tokens_list = []
        with open(twi_vocab_path, encoding="utf-8") as f:
            for line in f:
                tok = line.split("\t")[0]
                if tok:
                    twi_tokens_list.append(tok)

        new_tokens  = [t for t in twi_tokens_list if t not in nllb_vocab]
        in_nllb     = len(twi_tokens_list) - len(new_tokens)
        print(f"  Twi vocab size            : {len(twi_tokens_list):,}")
        print(f"  Already in NLLB-200       : {in_nllb:,}  "
              f"{'✓' if in_nllb == 3326 else f'(paper reports 3,326)'}")
        print(f"  New tokens to transplant  : {len(new_tokens):,}  "
              f"{'✓' if len(new_tokens) == 4674 else f'(paper reports 4,674)'}")
    except ImportError:
        print("\n  NOTE: Install transformers to verify NLLB-200 set-difference.")
        print("        pip install transformers")

    print(f"\n  All output files:")
    for path in [twi_model_path, twi_vocab_path,
                 en_model_path,  en_vocab_path,
                 twi_corpus_path, en_corpus_path]:
        size_kb = path.stat().st_size / 1024
        print(f"    {path.name:<35} {size_kb:>8.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare Twi-English corpus and train SentencePiece tokenizers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prepare_data.py
  python prepare_data.py --input data/corpus.csv --output outputs/
  python prepare_data.py --input corpus.csv --output . --skip_en_tokenizer
        """
    )
    parser.add_argument(
        "--input", type=str, default="corpus.csv",
        help="Path to corpus.csv (default: corpus.csv in current directory)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs",
        help="Output directory for all generated files (default: outputs/)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=VOCAB_SIZE,
        help=f"SentencePiece vocabulary size (default: {VOCAB_SIZE})"
    )
    parser.add_argument(
        "--char_coverage", type=float, default=CHARACTER_COVERAGE,
        help=f"SentencePiece character coverage (default: {CHARACTER_COVERAGE})"
    )
    parser.add_argument(
        "--skip_en_tokenizer", action="store_true",
        help="Skip training the English tokenizer (faster, only Twi tokenizer needed)"
    )
    parser.add_argument(
        "--skip_verify", action="store_true",
        help="Skip the output verification step"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Twi Vocabulary Transplanting — Data Preparation")
    print("=" * 60)
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_dir}")

    # Step 1: Load
    df = load_corpus(input_path)

    # Step 2: Deduplicate
    df = deduplicate(df)

    # Step 3: Extract monolingual corpora
    twi_corpus_path, en_corpus_path = extract_monolingual(df, output_dir)

    # Step 4: Train Twi tokenizer
    print(f"\n{'='*60}")
    print("STEP 4 — Training Twi BPE tokenizer")
    print(f"{'='*60}")
    twi_model_path, twi_vocab_path = train_tokenizer(
        corpus_path=twi_corpus_path,
        model_prefix=TWI_MODEL_PREFIX,
        output_dir=output_dir,
        vocab_size=args.vocab_size,
        character_coverage=args.char_coverage,
    )

    # Step 5: Train English tokenizer (fertility baseline)
    if not args.skip_en_tokenizer:
        print(f"\n{'='*60}")
        print("STEP 5 — Training English BPE tokenizer (fertility baseline)")
        print(f"{'='*60}")
        en_model_path, en_vocab_path = train_tokenizer(
            corpus_path=en_corpus_path,
            model_prefix=EN_MODEL_PREFIX,
            output_dir=output_dir,
            vocab_size=args.vocab_size,
            character_coverage=args.char_coverage,
        )
    else:
        print("\nSTEP 5 — Skipped (--skip_en_tokenizer flag set)")
        en_model_path = output_dir / f"{EN_MODEL_PREFIX}.model"
        en_vocab_path = output_dir / f"{EN_MODEL_PREFIX}.vocab"

    # Step 6: Corpus statistics
    corpus_statistics(df)

    # Step 7: Verify
    if not args.skip_verify:
        verify_outputs(
            twi_model_path=twi_model_path,
            twi_vocab_path=twi_vocab_path,
            en_model_path=en_model_path,
            en_vocab_path=en_vocab_path,
            twi_corpus_path=twi_corpus_path,
            en_corpus_path=en_corpus_path,
        )

    # Summary
    print(f"\n{'='*60}")
    print("DONE — All files written to:", output_dir)
    print(f"{'='*60}")
    print("""
  Generated files:
    tw_corpus.txt          monolingual Twi corpus (input to fertility analysis)
    en_corpus.txt          monolingual English corpus (fertility baseline)
    tok_twi_demo.model     Twi BPE tokenizer model
    tok_twi_demo.vocab     Twi BPE vocabulary with log-probabilities
    tok_english_demo.model English BPE tokenizer model
    tok_english_demo.vocab English BPE vocabulary with log-probabilities

  Next step:
    Run embedding_init_colab.py on Google Colab to reproduce Table 5.
    See README.md for full replication instructions.
    """)


if __name__ == "__main__":
    main()
