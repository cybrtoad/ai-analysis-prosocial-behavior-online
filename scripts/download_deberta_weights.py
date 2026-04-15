"""
scripts/download_deberta_weights.py
====================================
Pre-download and cache DeBERTa-v3-large and DeBERTa-v3-base model weights
on ARCC before submitting training jobs.

Run this ONCE on ARCC — either on a login node or via the download_weights.sh
SLURM job — before submitting train_deberta_large.sh / train_deberta_base.sh.

Why this matters
----------------
Compute nodes on HPC clusters often have restricted or no internet access.
If the training job tries to download weights at runtime it will fail or
stall.  Pre-caching here ensures the weights are already on disk when the
training job starts.

Usage
-----
  # Activate your environment, then run:
  python scripts/download_deberta_weights.py

  # On an HPC cluster with SLURM (recommended if login node internet is slow):
  sbatch scripts/download_weights.sh

Cache location
--------------
  Default: ~/.cache/huggingface/hub/  (~2 GB total for both models)
  Override with: HF_HOME=/your/scratch/hf_cache python scripts/download_deberta_weights.py
"""

import sys
from pathlib import Path

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed.  Activate your project environment first.")
    print("  See requirements.txt for dependencies.")
    sys.exit(1)

MODELS = [
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-base",
]


def download_model(model_name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Downloading: {model_name}")
    print(f"{'=' * 60}")

    print("  Tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    print(f"  Tokenizer OK — vocab size: {tok.vocab_size:,}")

    print("  Model weights...", flush=True)
    model = AutoModel.from_pretrained(model_name)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model OK — parameters: {n_params / 1e6:.0f}M")

    # Confirm cache location
    cache_info = Path(
        tok.name_or_path
        if hasattr(tok, "name_or_path") and Path(tok.name_or_path).exists()
        else "~/.cache/huggingface"
    ).expanduser()
    print(f"  Cached at: {cache_info}")


def main() -> None:
    print("DeBERTa Weight Pre-Download")
    print(f"Models to cache: {MODELS}")

    failed = []
    for m in MODELS:
        try:
            download_model(m)
        except Exception as exc:
            print(f"\nERROR downloading {m}: {exc}")
            failed.append(m)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"FAILED: {failed}")
        print("Check internet connectivity and HuggingFace availability.")
        sys.exit(1)
    else:
        print(f"All {len(MODELS)} models cached successfully.")
        print("You can now submit training jobs without internet access on compute nodes.")


if __name__ == "__main__":
    main()
