from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .data_loader import make_folds, load_item
from .encoding import encode_sheet_compressor
from .llm import PROMPT_COMPRESSOR_DETECTION, PROMPT_COMPRESSOR_DETECTION_M1M2
from .utils import cell_address, parse_address


# ─────────────────────────────────────────────────────────────────────────────
#  Paper hyperparameters (Appendix G)
# ─────────────────────────────────────────────────────────────────────────────

CUTOFF_TOKENS   = 5800
NUM_EPOCHS      = 15
BATCH_SIZE      = 5
GRAD_ACCUM      = 8
LEARNING_RATE   = 5e-5
MAX_GRAD_NORM   = 1.0
LR_SCHEDULER    = "cosine"
WARMUP_STEPS    = 0
OPTIMIZER       = "adamw_torch"
VAL_SIZE        = 0.0008
EVAL_STEPS            = 50
EVAL_BATCH_SIZE       = 5
EARLY_STOPPING_PATIENCE = 3   # epochs without val_loss improvement before stopping
LORA_RANK       = 32
LORA_ALPHA      = 64
LORA_DROPOUT    = 0.01

# All linear projection layers in Mistral-7B
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Token counter (no model loading)
# ─────────────────────────────────────────────────────────────────────────────

def _count_tokens_heuristic(text: str) -> int:
    """Rough token count without loading a tokenizer (char/4 heuristic)."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────────────────────
#  Ground-truth → completion string
# ─────────────────────────────────────────────────────────────────────────────

def _invert_map(m: dict) -> dict:
    """Invert a compressed→original index map to original→compressed."""
    return {v: k for k, v in m.items()}


def _compress_range(rng: str, inv_row: dict, inv_col: dict) -> Optional[str]:
    """Convert one GT range (original coords) → compressed coords.

    Returns None if either endpoint cannot be mapped (row/col was dropped by
    anchor extraction).  The nearest kept index is used as a fallback so that
    even partially-anchored ranges survive.
    """
    parts = rng.replace(' ', '').split(':')
    if len(parts) not in (1, 2):
        return None

    def nearest(val: int, mapping: dict) -> int:
        if val in mapping:
            return mapping[val]
        return mapping[min(mapping, key=lambda k: abs(k - val))]

    def convert(addr: str):
        parsed = parse_address(addr)
        if parsed is None:
            return None
        r_orig, c_orig = parsed
        if not inv_row or not inv_col:
            return addr.upper()
        r_comp = nearest(r_orig, inv_row)
        c_comp = nearest(c_orig, inv_col)
        return cell_address(r_comp, c_comp)

    if len(parts) == 1:
        a = convert(parts[0])
        return a
    a = convert(parts[0])
    b = convert(parts[1])
    if a is None or b is None:
        return None
    return f"{a}:{b}"


def _gt_to_completion(gt_ranges: List[str],
                      row_map: Optional[dict] = None,
                      col_map: Optional[dict] = None) -> str:
    """Format GT ranges in the paper's output format, e.g.
    ['range': 'A1:F9', 'range': 'A12:F18']

    When row_map/col_map (compressed→original) are provided the GT ranges are
    first projected into compressed coordinates so they match the cell
    addresses visible in the encoded input.
    """
    if row_map is not None and col_map is not None:
        inv_row = _invert_map(row_map)
        inv_col = _invert_map(col_map)
        compressed = [_compress_range(r, inv_row, inv_col) for r in gt_ranges]
        gt_ranges = [c for c in compressed if c is not None]

    parts = [f"'range': '{r}'" for r in gt_ranges]
    return "[" + ", ".join(parts) + "]"


# ─────────────────────────────────────────────────────────────────────────────
#  Per-row worker (module-level so it's picklable by ProcessPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_row(args: Tuple) -> Optional[Tuple[dict, bool]]:
    """Encode one spreadsheet row into a training record.

    Returns (record_dict, is_approx_truncated) or None if the row should be
    skipped (missing file, load error, no GT ranges).
    """
    row_dict, data_dir, k_anchor, use_aggregation = args
    try:
        # Reconstruct a minimal Series-like namespace from the plain dict
        row = pd.Series(row_dict)
        sheet_data, gt_ranges = load_item(row, data_dir)
    except Exception:
        return None

    if not gt_ranges:
        return None

    encoded, row_map, col_map = encode_sheet_compressor(
        sheet_data,
        k=k_anchor,
        use_extraction=True,
        use_translation=True,
        use_aggregation=use_aggregation,
        gt_ranges=gt_ranges,   # training: add GT corners as boundary candidates
    )

    approx_tokens = _count_tokens_heuristic(encoded)
    is_truncated = approx_tokens > CUTOFF_TOKENS

    prompt = (PROMPT_COMPRESSOR_DETECTION if use_aggregation
              else PROMPT_COMPRESSOR_DETECTION_M1M2)
    user_content = prompt + "\nINPUT:\n" + encoded
    # GT must be in compressed coords (matching encoded addresses), not original.
    completion   = _gt_to_completion(gt_ranges, row_map, col_map)

    record = {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": completion},
        ]
    }
    return record, is_truncated


# ─────────────────────────────────────────────────────────────────────────────
#  Build training JSONL for one fold (CPU-only, no model loading)
# ─────────────────────────────────────────────────────────────────────────────

def build_fold_jsonl(
    fold_idx: int,
    train_df: pd.DataFrame,
    data_dir: str,
    output_dir: Path,
    *,
    val_df: Optional[pd.DataFrame] = None,
    k_anchor: int = 4,
    use_aggregation: bool = False,
    num_workers: int = 1,
    verbose: bool = True,
) -> Path:
    """Encode all training-split spreadsheets, write a JSONL file.

    Each line is a JSON object with a 'messages' key in the conversational
    format expected by TRL 1.3+ SFTTrainer with assistant_only_loss=True:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    The encoding uses SheetCompressor without Aggregation and without format
    information, matching the paper's best model configuration.

    num_workers > 1 uses ProcessPoolExecutor to parallelise the CPU-bound
    SheetCompressor encoding across files. Output order is preserved so the
    JSONL is deterministic regardless of the worker count.

    No tokenizer or model is loaded here — token counts use a char/4 heuristic
    for statistics. Exact truncation happens during SFTTrainer collation.
    """
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = fold_dir / "train.jsonl"

    import time
    t_build_start = time.time()

    n_total = len(train_df)
    n_written = n_skipped = n_approx_truncated = 0
    total_approx_tokens = 0

    # Serialise rows to plain dicts so they cross the process boundary cleanly
    row_args = [
        (row.to_dict(), data_dir, k_anchor, use_aggregation)
        for _, row in train_df.iterrows()
    ]

    if verbose:
        workers_str = f"{num_workers} worker{'s' if num_workers > 1 else ''}"
        print(f"  [fold {fold_idx}] encoding {n_total} files ({workers_str}) …")

    def _iter_results():
        if num_workers <= 1:
            for i, args in enumerate(row_args):
                if verbose and i % 100 == 0 and i > 0:
                    elapsed = time.time() - t_build_start
                    print(f"  [fold {fold_idx}] {i}/{n_total} ({elapsed:.0f}s)")
                yield _encode_row(args)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # map preserves order; results arrive as workers finish
                n_done = 0
                for result in executor.map(_encode_row, row_args, chunksize=8):
                    n_done += 1
                    if verbose and n_done % 100 == 0:
                        elapsed = time.time() - t_build_start
                        print(f"  [fold {fold_idx}] {n_done}/{n_total} ({elapsed:.0f}s)")
                    yield result

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for result in _iter_results():
            if result is None:
                n_skipped += 1
                continue
            record, is_truncated = result
            if is_truncated:
                n_approx_truncated += 1
            approx_tokens = _count_tokens_heuristic(record["messages"][0]["content"])
            total_approx_tokens += min(approx_tokens, CUTOFF_TOKENS)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    # ── Validation split ──────────────────────────────────────────────────────
    val_jsonl_path = None
    if val_df is not None and len(val_df) > 0:
        val_jsonl_path = fold_dir / "val.jsonl"
        val_args = [(row.to_dict(), data_dir, k_anchor, use_aggregation) for _, row in val_df.iterrows()]
        n_val_written = 0
        with open(val_jsonl_path, "w", encoding="utf-8") as vf:
            for result in (_encode_row(a) for a in val_args):
                if result is None:
                    continue
                record, _ = result
                vf.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_val_written += 1
        if verbose:
            print(f"  [fold {fold_idx}] val: {n_val_written}/{len(val_df)} examples → {val_jsonl_path}")

    t_build_sec = round(time.time() - t_build_start, 1)

    stats = {
        "fold":                fold_idx,
        "n_total":             n_total,
        "n_written":           n_written,
        "n_skipped":           n_skipped,
        "n_approx_truncated":  n_approx_truncated,
        "total_approx_tokens": total_approx_tokens,
        "avg_approx_tokens":   total_approx_tokens // max(1, n_written),
        "build_sec":           t_build_sec,
        "jsonl_path":          str(jsonl_path),
    }
    stats_path = fold_dir / "build_stats.json"
    with open(stats_path, "w") as sf:
        json.dump(stats, sf, indent=2)

    if verbose:
        print(f"  [fold {fold_idx}] {n_written} examples written "
              f"({n_approx_truncated} approx truncated, {n_skipped} skipped)  "
              f"avg_tokens≈{stats['avg_approx_tokens']}  "
              f"build_time={t_build_sec:.1f}s  → {jsonl_path}")

    return jsonl_path


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA fine-tuning for one fold (GPU required)
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    output_dir: Path,
    base_model_name: str,
    *,
    patience: int = EARLY_STOPPING_PATIENCE,
    verbose: bool = True,
) -> Path:
    """Load the pre-built JSONL for fold_idx, fine-tune the model, save adapter.

    Exact paper hyperparameters (Appendix G):
      cutoff_len=5800, lr=5e-5, epochs=15, batch=5, grad_accum=8,
      cosine LR, AdamW, fp16, LoRA rank=32 alpha=64 dropout=0.01

    Loss is computed on assistant tokens only (TRL 1.3 assistant_only_loss=True).

    The saved adapter is at output_dir/fold_{fold_idx}/adapter/.
    Timing (per-epoch and total) is saved to output_dir/fold_{fold_idx}/train_timing.json.
    """
    import math
    import time
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerState, TrainerControl
    from trl import SFTConfig, SFTTrainer

    jsonl_path = output_dir / f"fold_{fold_idx}" / "train.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Training JSONL not found: {jsonl_path}\n"
            "Run 'build-data' step first."
        )

    # ── Load dataset ──────────────────────────────────────────────────────
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        raise ValueError(f"No training examples found in {jsonl_path}")

    train_dataset = Dataset.from_list(examples)

    # Load val split from val.jsonl written by build_fold_jsonl
    val_jsonl_path = output_dir / f"fold_{fold_idx}" / "val.jsonl"
    eval_dataset = None
    if val_jsonl_path.exists():
        val_examples = []
        with open(val_jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    val_examples.append(json.loads(line))
        eval_dataset = Dataset.from_list(val_examples) if val_examples else None
        if verbose:
            print(f"  [fold {fold_idx}] {len(examples)} train examples, "
                  f"{len(val_examples) if val_examples else 0} val examples")
    else:
        if verbose:
            print(f"  [fold {fold_idx}] {len(examples)} train examples (no val.jsonl found)")

    t_fold_start = time.time()

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required for training with left-pad models

    # TRL 1.3+ requires {% generation %}...{% endgeneration %} markers in the
    # chat template so assistant_only_loss knows which tokens to train on.
    # Mistral-v0.2's default template lacks these; patch it here.
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '[INST] ' + message['content'] + ' [/INST]' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}"
        "{% else %}"
        "{{ raise_exception('Only user and assistant roles are allowed!') }}"
        "{% endif %}"
        "{% endfor %}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"  [fold {fold_idx}] loading {base_model_name} …")

    t_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False      # required for gradient checkpointing
    model.enable_input_require_grads()  # required for PEFT LoRA
    t_load_sec = time.time() - t_load_start
    if verbose:
        print(f"  [fold {fold_idx}] model loaded in {t_load_sec:.1f}s")

    # ── LoRA config (exact paper params) ─────────────────────────────────
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    # ── Training config ───────────────────────────────────────────────────
    adapter_dir = output_dir / f"fold_{fold_idx}" / "adapter"
    has_eval = eval_dataset is not None

    sft_config = SFTConfig(
        output_dir=str(adapter_dir),

        # Exact Appendix G hyperparameters
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,
        optim=OPTIMIZER,
        fp16=True,

        # Sequence length cutoff (cutoff_len=5800 from paper)
        max_length=CUTOFF_TOKENS,

        # Train on assistant responses only
        assistant_only_loss=True,

        # Memory efficiency
        gradient_checkpointing=True,

        # Evaluate once per epoch on the k-fold val split (when available)
        eval_strategy="epoch" if has_eval else "no",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,

        # Save a checkpoint after every epoch; keep best + latest.
        # Primary metric: eval_loss (lower is better, always available).
        # Fallback to eval_mean_token_accuracy handled by _AdaptiveEarlyStopping.
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss" if has_eval else None,
        greater_is_better=False,

        logging_steps=10,
        report_to="none",
        dataloader_num_workers=4,
    )

    # ── Per-epoch timing + val-metrics callback ───────────────────────────
    class _EpochTimer(TrainerCallback):
        def __init__(self):
            self.epoch_times: List[float] = []
            self._epoch_start: Optional[float] = None
            self._best_val_loss: Optional[float] = None

        def on_epoch_begin(self, args, state: TrainerState,
                           control: TrainerControl, **kwargs):
            self._epoch_start = time.time()

        def on_epoch_end(self, args, state: TrainerState,
                         control: TrainerControl, **kwargs):
            if self._epoch_start is not None:
                elapsed = round(time.time() - self._epoch_start, 1)
                self.epoch_times.append(elapsed)

        def on_evaluate(self, args, state: TrainerState,
                        control: TrainerControl, metrics: dict = None, **kwargs):
            if not verbose:
                return
            epoch_no = len(self.epoch_times)
            elapsed = self.epoch_times[-1] if self.epoch_times else 0
            train_entries = [
                e for e in state.log_history
                if abs(e.get('epoch', -1) - epoch_no) < 0.5 and 'loss' in e
                and 'eval_loss' not in e
            ]
            train_loss_str = (f"  train_loss={train_entries[-1]['loss']:.4f}"
                              if train_entries else "")
            val_loss = (metrics or {}).get('eval_loss')
            val_acc  = (metrics or {}).get('eval_mean_token_accuracy')
            val_str  = (f"  val_loss={val_loss:.4f}" if val_loss is not None else "")
            val_str += (f"  val_acc={val_acc:.4f}"   if val_acc  is not None else "")
            if val_loss is not None and not math.isinf(val_loss):
                if self._best_val_loss is None or val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
            best_str = (f"  [best={self._best_val_loss:.4f}]"
                        if self._best_val_loss is not None else "")
            print(f"  [fold {fold_idx}] epoch {epoch_no:2d}/{NUM_EPOCHS}"
                  f"  {elapsed:.0f}s{train_loss_str}{val_str}{best_str}")

    epoch_timer = _EpochTimer()

    # ── Early stopping: eval_loss primary, mean_token_accuracy fallback ────
    # Uses eval_loss (lower is better) when it is finite.
    # Falls back to eval_mean_token_accuracy (higher is better) when eval_loss
    # is +inf (e.g. degenerate checkpoints early in training).
    class _AdaptiveEarlyStopping(TrainerCallback):
        def __init__(self, pat: int):
            self._patience  = pat
            self._no_improve = 0
            self._best: Optional[float] = None
            self._higher_better = False   # tracks which metric is active

        def on_evaluate(self, args, state: TrainerState,
                        control: TrainerControl, metrics: dict = None, **kwargs):
            m = metrics or {}
            loss = m.get("eval_loss", float("inf"))
            acc  = m.get("eval_mean_token_accuracy")

            if not math.isinf(loss):
                current, higher_better = loss, False
            elif acc is not None:
                current, higher_better = acc, True
            else:
                return   # nothing to track

            improved = (
                self._best is None
                or (higher_better and current > self._best)
                or (not higher_better and current < self._best)
            )
            if improved:
                self._best = current
                self._higher_better = higher_better
                self._no_improve = 0
            else:
                self._no_improve += 1

            if self._patience > 0 and self._no_improve >= self._patience:
                control.should_training_stop = True
                if verbose:
                    metric_name = "val_acc" if higher_better else "val_loss"
                    print(f"  [fold {fold_idx}] early stopping: "
                          f"no improvement for {self._patience} evals "
                          f"({metric_name}={current:.4f}, best={self._best:.4f})")

    callbacks = [epoch_timer]
    if has_eval and patience > 0:
        callbacks.append(_AdaptiveEarlyStopping(pat=patience))
        if verbose:
            print(f"  [fold {fold_idx}] early stopping: patience={patience} epochs "
                  f"(primary=eval_loss, fallback=eval_mean_token_accuracy)")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if verbose:
        print(f"  [fold {fold_idx}] starting training "
              f"({NUM_EPOCHS} epochs, effective_batch={BATCH_SIZE * GRAD_ACCUM}, "
              f"steps_per_epoch≈{len(examples) // BATCH_SIZE // GRAD_ACCUM}) …")

    t_train_start = time.time()
    trainer.train()
    t_train_sec = round(time.time() - t_train_start, 1)

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    t_total_sec = round(time.time() - t_fold_start, 1)

    # ── Save timing ───────────────────────────────────────────────────────
    timing = {
        "fold":           fold_idx,
        "model_load_sec": round(t_load_sec, 1),
        "train_sec":      t_train_sec,
        "total_sec":      t_total_sec,
        "epoch_times_sec": epoch_timer.epoch_times,
        "avg_epoch_sec":  round(sum(epoch_timer.epoch_times) /
                                max(1, len(epoch_timer.epoch_times)), 1),
        "n_epochs":       NUM_EPOCHS,
        "n_train":        len(examples),
        "adapter_path":   str(adapter_dir),
    }
    timing_path = output_dir / f"fold_{fold_idx}" / "train_timing.json"
    with open(timing_path, "w") as tf:
        json.dump(timing, tf, indent=2)

    if verbose:
        h, m = divmod(int(t_train_sec), 3600)
        m, s = divmod(m, 60)
        time_str = (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")
        print(f"  [fold {fold_idx}] training done in {time_str}  "
              f"(load={t_load_sec:.0f}s, total={t_total_sec:.0f}s)")
        print(f"  [fold {fold_idx}] adapter saved → {adapter_dir}")
        print(f"  [fold {fold_idx}] timing saved  → {timing_path}")

    return adapter_dir


# ─────────────────────────────────────────────────────────────────────────────
#  Cost / stats summary
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(output_dir: Path, fold_indices: List[int]) -> None:
    total_tokens = 0
    for fi in fold_indices:
        stats_path = output_dir / f"fold_{fi}" / "build_stats.json"
        if not stats_path.exists():
            continue
        s = json.loads(stats_path.read_text())
        total_tokens += s.get("total_approx_tokens", 0)
        print(f"  fold {fi}: {s['n_written']} examples  "
              f"avg≈{s['avg_approx_tokens']} tok  "
              f"({s['n_approx_truncated']} truncated)")
    print(f"  total approx input tokens across {len(fold_indices)} folds: "
          f"{total_tokens:,}")
    print(f"  (×{NUM_EPOCHS} epochs = {total_tokens*NUM_EPOCHS:,} training tokens)")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spreadsheet_llm.finetune",
        description="Fine-tune Mistral-7B-Instruct-v0.2 for SpreadsheetLLM table detection.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── shared ────────────────────────────────────────────────────
    def _add_common(sp):
        sp.add_argument("--output", required=True,
                        help="Root output directory (JSONL + adapter checkpoints).")
        sp.add_argument("--k", type=int, default=5)
        sp.add_argument("--random-state", type=int, default=2112)
        sp.add_argument("--fold", type=int, default=None,
                        help="Single fold index (0-based). Default: all folds.")
        sp.add_argument("--quiet", action="store_true")

    # ── build-data ────────────────────────────────────────────────
    sp_build = sub.add_parser(
        "build-data",
        help="Encode training splits and write JSONL files. CPU-only, no model loading.",
    )
    _add_common(sp_build)
    sp_build.add_argument("--manifest", required=True)
    sp_build.add_argument("--data-dir", default=".")
    sp_build.add_argument("--k-anchor", type=int, default=4)
    sp_build.add_argument("--use-aggregation", action="store_true",
                          help="Enable Module 3 (data-format aggregation). "
                               "Off by default: paper Table 2 shows M1+M2 best.")
    sp_build.add_argument("--num-workers", type=int,
                          default=os.cpu_count() or 1,
                          help="Parallel worker processes for encoding "
                               "(default: all CPU cores).")

    # ── train ─────────────────────────────────────────────────────
    sp_train = sub.add_parser(
        "train",
        help="Fine-tune one or all folds from pre-built JSONL. GPU required.",
    )
    _add_common(sp_train)
    sp_train.add_argument("--base-model",
                          default="mistralai/Mistral-7B-Instruct-v0.2")
    sp_train.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                          help="Early-stopping patience in epochs (0 = disabled).")

    # ── run (all-in-one) ─────────────────────────────────────────
    sp_run = sub.add_parser(
        "run",
        help="build-data + train in one shot. GPU required.",
    )
    _add_common(sp_run)
    sp_run.add_argument("--manifest", required=True)
    sp_run.add_argument("--data-dir", default=".")
    sp_run.add_argument("--k-anchor", type=int, default=4)
    sp_run.add_argument("--use-aggregation", action="store_true",
                        help="Enable Module 3 (data-format aggregation). "
                             "Off by default: paper Table 2 shows M1+M2 best.")
    sp_run.add_argument("--base-model",
                        default="mistralai/Mistral-7B-Instruct-v0.2")
    sp_run.add_argument("--num-workers", type=int,
                        default=os.cpu_count() or 1,
                        help="Parallel worker processes for encoding "
                             "(default: all CPU cores).")

    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    verbose = not args.quiet
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine fold indices
    fold_indices = list(range(args.k)) if args.fold is None else [args.fold]

    # ── build-data step ───────────────────────────────────────────
    if args.cmd in ("build-data", "run"):
        folds = make_folds(args.manifest, k=args.k,
                           random_state=args.random_state)
        if verbose:
            print(f"[build-data] {len(fold_indices)} fold(s)  "
                  f"manifest={args.manifest}")
        num_workers = getattr(args, 'num_workers', 1)
        use_agg = getattr(args, 'use_aggregation', False)
        for fi in fold_indices:
            train_df, val_df = folds[fi]
            build_fold_jsonl(
                fi, train_df, args.data_dir, output_dir,
                val_df=val_df,
                k_anchor=args.k_anchor,
                use_aggregation=use_agg,
                num_workers=num_workers,
                verbose=verbose,
            )
        if verbose:
            print("\n[stats]")
            print_stats(output_dir, fold_indices)

    # ── train step ────────────────────────────────────────────────
    if args.cmd in ("train", "run"):
        base_model = args.base_model
        if verbose:
            print(f"\n[train] base_model={base_model}")
        for fi in fold_indices:
            if verbose:
                print(f"\n[train] fold {fi}")
            try:
                adapter_dir = train_fold(
                    fi, output_dir, base_model,
                    patience=getattr(args, 'patience', EARLY_STOPPING_PATIENCE),
                    verbose=verbose,
                )
                if verbose:
                    print(f"[train] fold {fi} done → {adapter_dir}")
            except Exception as e:
                print(f"[train] fold {fi} FAILED: {e}", file=sys.stderr)
                traceback.print_exc()
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
