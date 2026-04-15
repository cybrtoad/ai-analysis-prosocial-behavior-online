"""
src/train_deberta.py
====================
Three-stage curriculum fine-tuning of DeBERTa for prosociality scoring.

Run with --model-name microsoft/deberta-v3-large (primary model, ~15h total on L40S)
or --model-name microsoft/deberta-v3-base (comparison baseline, ~4h total on L40S).

Three-stage pipeline
--------------------
  Stage 1  Broad initialization on ~100K Phase 2 Claude-scored records.
           NOTE: requires Phase 2 Claude scoring to be complete first.
           Skip with --skip-stage-1 and go directly to Stage 2 if Phase 2
           data is not yet available.

  Stage 2  Quality refinement on ~3K Phase 1 gold-labeled records.
           Starts from Stage 1 checkpoint (or cold-start if Stage 1 skipped).

  Stage 3  Human alignment on 300–400 human-consensus records.
           Starts from Stage 2 checkpoint.
           Use --predict-holdout to generate the Claude-DeBERTa agreement
           predictions file after Stage 3 completes.

Typical invocation (all three stages in one job, as in the SLURM scripts):
  python src/train_deberta.py \\
      --model-name microsoft/deberta-v3-large \\
      --stage 1 \\
      --train-data output/phase2/phase2_claude_scores.parquet \\
      --val-data output/splits/val.parquet \\
      --output-dir output/models/deberta-v3-large

  python src/train_deberta.py \\
      --model-name microsoft/deberta-v3-large \\
      --stage 2 \\
      --train-data output/splits/train.parquet \\
      --val-data output/splits/val.parquet \\
      --checkpoint output/models/deberta-v3-large/stage1/best \\
      --output-dir output/models/deberta-v3-large

  python src/train_deberta.py \\
      --model-name microsoft/deberta-v3-large \\
      --stage 3 \\
      --train-data output/human_labels/human_train.parquet \\
      --val-data output/human_labels/human_val.parquet \\
      --checkpoint output/models/deberta-v3-large/stage2/best \\
      --output-dir output/models/deberta-v3-large \\
      --predict-holdout

Outputs (under --output-dir/stageN/):
  best/                    Saved model checkpoint (HuggingFace + regressor.pt)
  metrics.json             Loss and per-dimension ICC at each epoch
  val_predictions.csv      Val-set predictions vs. reference scores
  holdout_predictions.csv  Claude-DeBERTa holdout predictions (--predict-holdout)
                           Feed this to evaluate_claude_deberta_agreement.py

Both DeBERTa-v3-large and DeBERTa-v3-base are trained with identical code.
The SLURM scripts in scripts/ submit both as independent jobs.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ROOT         = Path(__file__).resolve().parent.parent
SPLITS_DIR   = ROOT / "output" / "splits"
LOG_DIR      = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

DIMENSIONS = ["empathy", "constructiveness", "respect", "social_cohesion"]

# Stage-specific training defaults.
# All three can be overridden via CLI args.
STAGE_DEFAULTS = {
    1: {"lr": 2e-5, "epochs": 3, "batch_size": 16, "warmup_ratio": 0.06},
    2: {"lr": 1e-5, "epochs": 5, "batch_size": 16, "warmup_ratio": 0.10},
    3: {"lr": 5e-6, "epochs": 10, "batch_size": 8,  "warmup_ratio": 0.20},
}

DEFAULT_MAX_LENGTH  = 512
DEFAULT_HOLDOUT     = str(SPLITS_DIR / "claude_agreement_holdout.parquet")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _collate(batch: list[dict]) -> dict:
    """Custom collate: keeps record_id as a list of strings."""
    record_ids = [item.pop("record_id") for item in batch]
    collated = torch.utils.data.default_collate(batch)
    collated["record_id"] = record_ids
    return collated


class ProsocialityDataset(Dataset):
    """
    Tokenizes intervention triples and loads dimension scores as regression targets.

    Input format fed to DeBERTa:
      text_a = pre_intervention_text
      text_b = intervention_text [SEP] post_intervention_text

    With truncation="longest_first" the tokenizer trims whichever of text_a
    or text_b is longer first, preserving as much of each as possible within
    the max_length token budget.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.records: list[dict] = []

        sep = tokenizer.sep_token or "[SEP]"

        for _, row in df.iterrows():
            pre          = str(row.get("pre_intervention_text",  "") or "").strip()
            intervention = str(row.get("intervention_text",      "") or "").strip()
            post         = str(row.get("post_intervention_text", "") or "").strip()

            scores = [
                float(row[dim]) if pd.notna(row.get(dim)) else float("nan")
                for dim in DIMENSIONS
            ]

            self.records.append({
                "text_a":    pre,
                "text_b":    f"{intervention} {sep} {post}",
                "scores":    scores,
                "record_id": str(row.get("record_id", "")),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text_a"],
            rec["text_b"],
            max_length=self.max_length,
            truncation="longest_first",
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"]    = torch.tensor(rec["scores"], dtype=torch.float32)
        item["record_id"] = rec["record_id"]
        return item


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DeBERTaRegressor(nn.Module):
    """
    DeBERTa encoder with a linear regression head over the [CLS] representation.
    Outputs 4 dimension scores (empathy, constructiveness, respect, social_cohesion),
    clamped to the valid Likert range [1.0, 5.0].
    """

    def __init__(self, model_name: str, n_outputs: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder   = AutoModel.from_pretrained(model_name)
        hidden_size    = self.encoder.config.hidden_size
        self.dropout   = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, n_outputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        out = self.encoder(**kwargs)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return torch.clamp(self.regressor(cls), 1.0, 5.0)

    def save(self, directory: Path) -> None:
        """Save encoder (HuggingFace format) and regressor head weights."""
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory)
        torch.save(self.regressor.state_dict(), directory / "regressor.pt")

    @classmethod
    def load(cls, directory: Path, model_name: str) -> "DeBERTaRegressor":
        """
        Load a previously saved checkpoint.
        model_name is used for the tokenizer config; the encoder weights
        come from the checkpoint directory.
        """
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.encoder   = AutoModel.from_pretrained(directory)
        hidden_size        = instance.encoder.config.hidden_size
        instance.dropout   = nn.Dropout(0.1)
        instance.regressor = nn.Linear(hidden_size, 4)
        instance.regressor.load_state_dict(
            torch.load(directory / "regressor.pt", map_location="cpu")
        )
        return instance


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def icc_2_1(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single rater, absolute agreement."""
    r1 = np.asarray(rater1, dtype=float)
    r2 = np.asarray(rater2, dtype=float)
    mask = ~(np.isnan(r1) | np.isnan(r2))
    r1, r2 = r1[mask], r2[mask]
    n = len(r1)
    if n < 3:
        return float("nan")
    k   = 2
    data = np.column_stack([r1, r2])
    gm   = data.mean()
    rm   = data.mean(axis=1)
    cm   = data.mean(axis=0)
    ss_s = k * np.sum((rm - gm) ** 2)
    ss_r = n * np.sum((cm - gm) ** 2)
    ss_e = np.sum((data - gm) ** 2) - ss_s - ss_r
    ms_s = ss_s / (n - 1)
    ms_r = ss_r / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_s + (k - 1) * ms_e + k * (ms_r - ms_e) / n
    return float("nan") if denom == 0 else float((ms_s - ms_e) / denom)


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: DeBERTaRegressor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns mean batch loss."""
    model.train()
    criterion = nn.MSELoss(reduction="none")
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            preds      = model(input_ids, attention_mask, token_type_ids)
            valid_mask = ~torch.isnan(labels)
            if not valid_mask.any():
                continue
            loss = criterion(preds[valid_mask], labels[valid_mask]).mean()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


@torch.no_grad()
def evaluate(
    model: DeBERTaRegressor,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model; returns loss, per-dimension ICC, and raw predictions."""
    model.eval()
    criterion = nn.MSELoss(reduction="none")

    preds_all = {dim: [] for dim in DIMENSIONS}
    refs_all  = {dim: [] for dim in DIMENSIONS}
    ids_all   = []
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(loader, desc="  eval ", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        preds = model(input_ids, attention_mask, token_type_ids)

        valid_mask = ~torch.isnan(labels)
        if valid_mask.any():
            loss = criterion(preds[valid_mask], labels[valid_mask]).mean()
            total_loss += loss.item()
            n_batches  += 1

        for i, dim in enumerate(DIMENSIONS):
            preds_all[dim].extend(preds[:, i].cpu().numpy().tolist())
            refs_all[dim].extend(labels[:, i].cpu().numpy().tolist())
        ids_all.extend(batch["record_id"])

    icc_per_dim = {
        dim: icc_2_1(refs_all[dim], preds_all[dim])
        for dim in DIMENSIONS
    }
    mean_icc  = float(np.nanmean(list(icc_per_dim.values())))
    avg_loss  = total_loss / n_batches if n_batches > 0 else float("nan")

    return {
        "loss":        avg_loss,
        "mean_icc":    mean_icc,
        "icc_per_dim": icc_per_dim,
        "preds":       preds_all,
        "refs":        refs_all,
        "record_ids":  ids_all,
    }


def save_predictions(results: dict, path: Path) -> None:
    """Write predictions + references to a CSV for downstream analysis."""
    rows = {"record_id": results["record_ids"]}
    for dim in DIMENSIONS:
        rows[dim]             = results["preds"][dim]
        rows[f"ref_{dim}"]    = results["refs"][dim]
    rows["prosociality_composite"] = [
        float(np.nanmean([results["preds"][d][i] for d in DIMENSIONS]))
        for i in range(len(results["record_ids"]))
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-stage DeBERTa curriculum training for prosociality scoring"
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["microsoft/deberta-v3-large", "microsoft/deberta-v3-base"],
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2, 3],
        help="Training stage (1=broad init, 2=quality refinement, 3=human alignment).",
    )
    parser.add_argument("--train-data", required=True,
                        help="Path to training parquet.")
    parser.add_argument("--val-data",   required=True,
                        help="Path to validation parquet.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a previous-stage checkpoint directory to load.")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory. Stage outputs go to output-dir/stageN/.")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--max-length", type=int,   default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--grad-accum", type=int,   default=1,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum).")
    parser.add_argument("--fp16",       action="store_true",
                        help="Use FP16 mixed precision (recommended on L40S).")
    parser.add_argument("--predict-holdout", action="store_true",
                        help="Generate holdout predictions after training (for evaluate_claude_deberta_agreement.py).")
    parser.add_argument("--holdout-data", default=DEFAULT_HOLDOUT,
                        help=f"Path to holdout parquet (default: {DEFAULT_HOLDOUT}).")
    args = parser.parse_args()

    # Stage-specific defaults (overridden by any explicit CLI args)
    defaults  = STAGE_DEFAULTS[args.stage]
    epochs     = args.epochs     or defaults["epochs"]
    batch_size = args.batch_size or defaults["batch_size"]
    lr         = args.lr         or defaults["lr"]
    warmup_ratio = defaults["warmup_ratio"]

    # Derive a short model tag for output paths
    model_tag = args.model_name.split("/")[-1]   # e.g. "deberta-v3-large"

    out_root  = Path(args.output_dir)
    stage_dir = out_root / f"stage{args.stage}"
    best_dir  = stage_dir / "best"
    stage_dir.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"train_{model_tag}_stage{args.stage}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    log = logging.getLogger(__name__)

    log.info(f"=== DeBERTa Training: {model_tag}  Stage {args.stage} ===")
    log.info(f"  train_data  : {args.train_data}")
    log.info(f"  val_data    : {args.val_data}")
    log.info(f"  checkpoint  : {args.checkpoint or 'none (cold start)'}")
    log.info(f"  output_dir  : {stage_dir}")
    log.info(f"  epochs={epochs}  batch={batch_size}  lr={lr}  fp16={args.fp16}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  device      : {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU         : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_path = Path(args.train_data)
    val_path   = Path(args.val_data)

    if not train_path.exists():
        log.error(f"Training data not found: {train_path}")
        sys.exit(1)
    if not val_path.exists():
        log.error(f"Validation data not found: {val_path}")
        sys.exit(1)

    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    log.info(f"  train records: {len(df_train):,}   val records: {len(df_val):,}")

    # Drop invalid triples if the column is present
    for df in [df_train, df_val]:
        if "invalid_triple" in df.columns:
            before = len(df)
            df.drop(df[df["invalid_triple"].fillna(False)].index, inplace=True)
            log.info(f"  Dropped {before - len(df)} invalid triples from split.")

    # ------------------------------------------------------------------
    # Tokenizer and datasets
    # ------------------------------------------------------------------
    tokenizer_src = args.checkpoint if args.checkpoint else args.model_name
    tokenizer     = AutoTokenizer.from_pretrained(tokenizer_src)

    ds_train = ProsocialityDataset(df_train, tokenizer, args.max_length)
    ds_val   = ProsocialityDataset(df_val,   tokenizer, args.max_length)

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=_collate,
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=_collate,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    if args.checkpoint:
        log.info(f"Loading checkpoint from {args.checkpoint}")
        model = DeBERTaRegressor.load(Path(args.checkpoint), args.model_name)
    else:
        log.info(f"Cold-starting from {args.model_name}")
        model = DeBERTaRegressor(args.model_name)

    model = model.to(device)

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    # Use weight decay on non-bias / non-layernorm parameters
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer      = AdamW(param_groups, lr=lr)
    total_steps    = (len(dl_train) // args.grad_accum) * epochs
    warmup_steps   = int(total_steps * warmup_ratio)
    scheduler      = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    scaler         = torch.cuda.amp.GradScaler() if args.fp16 and device.type == "cuda" else None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_icc   = -float("inf")
    best_epoch     = -1
    epoch_metrics  = []

    for epoch in range(1, epochs + 1):
        log.info(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_one_epoch(model, dl_train, optimizer, scheduler, scaler, device)
        val_result = evaluate(model, dl_val, device)

        m = {
            "epoch":       epoch,
            "train_loss":  round(train_loss, 5),
            "val_loss":    round(val_result["loss"], 5),
            "val_mean_icc": round(val_result["mean_icc"], 4),
            "val_icc_per_dim": {
                k: round(v, 4) for k, v in val_result["icc_per_dim"].items()
            },
        }
        epoch_metrics.append(m)
        log.info(
            f"  train_loss={m['train_loss']:.4f}  "
            f"val_loss={m['val_loss']:.4f}  "
            f"val_mean_icc={m['val_mean_icc']:.4f}  "
            + "  ".join(
                f"{d[:4]}={v:.3f}" for d, v in m["val_icc_per_dim"].items()
            )
        )

        if val_result["mean_icc"] > best_val_icc:
            best_val_icc = val_result["mean_icc"]
            best_epoch   = epoch
            model.save(best_dir)
            save_predictions(val_result, stage_dir / "val_predictions.csv")
            log.info(f"  *** New best (val ICC {best_val_icc:.4f}) — saved to {best_dir}")

    log.info(f"\nBest epoch: {best_epoch}  Best val mean ICC: {best_val_icc:.4f}")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "model_name":    args.model_name,
        "stage":         args.stage,
        "train_data":    str(train_path),
        "val_data":      str(val_path),
        "checkpoint_in": args.checkpoint,
        "best_checkpoint": str(best_dir),
        "best_epoch":    best_epoch,
        "best_val_mean_icc": round(best_val_icc, 4),
        "epochs":        epoch_metrics,
    }
    metrics_path = stage_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics → {metrics_path}")

    # ------------------------------------------------------------------
    # Generate holdout predictions (--predict-holdout)
    # ------------------------------------------------------------------
    if args.predict_holdout:
        holdout_path = Path(args.holdout_data)
        if not holdout_path.exists():
            log.error(f"Holdout file not found: {holdout_path}")
            log.error("Run src/split.py first.")
        else:
            log.info(f"\nGenerating holdout predictions from {holdout_path}")
            df_holdout = pd.read_parquet(holdout_path)
            if "invalid_triple" in df_holdout.columns:
                df_holdout = df_holdout[~df_holdout["invalid_triple"].fillna(False)]

            ds_holdout = ProsocialityDataset(df_holdout, tokenizer, args.max_length)
            dl_holdout = DataLoader(
                ds_holdout, batch_size=batch_size * 2, shuffle=False,
                num_workers=4, pin_memory=True, collate_fn=_collate,
            )

            # Load the best checkpoint for inference
            best_model = DeBERTaRegressor.load(best_dir, args.model_name).to(device)
            holdout_result = evaluate(best_model, dl_holdout, device)

            holdout_pred_path = stage_dir / "holdout_predictions.csv"
            save_predictions(holdout_result, holdout_pred_path)
            log.info(f"Saved holdout predictions → {holdout_pred_path}")
            log.info(
                "Run: python src/evaluate_claude_deberta_agreement.py "
                f"--predictions {holdout_pred_path} "
                f"--model-label {model_tag}"
            )

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
