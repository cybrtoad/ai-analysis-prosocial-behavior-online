"""
One-time migration: add intervener_role column to all existing parquet files
that contain platform and intervention_type.

Derivation rules (matches src/standardize.py):
  reddit                                        -> peer
  stackexchange                                 -> domain_expert
  wikipedia + warning | restriction             -> formal_authority
  wikipedia + reframe | positive_reinforcement  -> peer_editor

Files updated:
  output/processed/unified_interventions.parquet
  output/processed/labeling_sample.parquet
  output/labels/gold_labels_merged.parquet
  output/splits/train.parquet
  output/splits/val.parquet
  output/splits/test.parquet
  output/splits/claude_agreement_holdout.parquet

Usage:
  python scripts/add_intervener_role.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    ROOT / "output" / "processed" / "unified_interventions.parquet",
    ROOT / "output" / "processed" / "labeling_sample.parquet",
    ROOT / "output" / "labels"  / "gold_labels_merged.parquet",
    ROOT / "output" / "splits"  / "train.parquet",
    ROOT / "output" / "splits"  / "val.parquet",
    ROOT / "output" / "splits"  / "test.parquet",
    ROOT / "output" / "splits"  / "claude_agreement_holdout.parquet",
]


def derive_intervener_role(platform: str, intervention_type: str) -> str:
    if platform == "reddit":
        return "peer"
    if platform == "stackexchange":
        return "domain_expert"
    if platform == "wikipedia":
        if intervention_type in ("warning", "restriction"):
            return "formal_authority"
        return "peer_editor"
    return "unknown"


def backfill(path: Path) -> None:
    if not path.exists():
        print(f"  SKIP (not found): {path.relative_to(ROOT)}")
        return

    df = pd.read_parquet(path)

    if "intervener_role" in df.columns:
        print(f"  SKIP (already has column): {path.relative_to(ROOT)}")
        return

    if "platform" not in df.columns or "intervention_type" not in df.columns:
        print(f"  SKIP (missing platform/intervention_type): {path.relative_to(ROOT)}")
        return

    df["intervener_role"] = df.apply(
        lambda r: derive_intervener_role(r["platform"], r["intervention_type"]), axis=1
    )

    df.to_parquet(path, index=False)

    dist = df["intervener_role"].value_counts().to_dict()
    print(f"  OK  {path.relative_to(ROOT)}  ({len(df):,} rows)  {dist}")


def main() -> None:
    print("=== Adding intervener_role to existing parquet files ===")
    for target in TARGETS:
        backfill(target)
    print("=== Done ===")


if __name__ == "__main__":
    main()
