"""
Train mT5 on ThaiSum with keyword-guided inputs.

Key features:
- Optional keyword prefixing (overlap-based for train, unsupervised for val)
- Validation-time ROUGE (greedy decode) via HF evaluate
- Size-controlled training subset for speed/fair comparisons
- Hardware-aware precision (bf16/fp16) and data loader tuning
- Saves structured JSON + readable TXT metrics
"""

import os, sys, argparse
from datetime import datetime
import warnings
import json
import re
import numpy as np
import evaluate
import torch
from datasets import DatasetDict

warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ===== Make ../src importable (for preprocess.py) =====
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    GenerationConfig,
)
from preprocess import load_thaisum, preprocess_dataset, MODEL_NAME


# ---------------------------------------------------------------------
# Path resolver: ./model/<name> with timestamp if exists (no --overwrite)
# ---------------------------------------------------------------------
def _resolve_output_dir(name: str, overwrite: bool) -> str:
    """
    Resolve output directory under ./model/<name>.
    If the directory already exists and --overwrite is False, append a timestamp.
    """
    base = os.path.join(".", "model", name)
    if os.path.isdir(base) and not overwrite:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        print(f"⚠️  Output dir exists: {base} → will use timestamped dir instead.")
        base = f"{base}-{ts}"
    return base


# --------------------------- Keyword helpers ---------------------------
# Thai Unicode block and basic ASCII token pattern (words/numbers)
_TH_RE = r"[\u0E00-\u0E7F]+"
_EN_RE = r"[A-Za-z0-9]+"
_TOKEN_RE = re.compile(f"{_TH_RE}|{_EN_RE}")


def _simple_word_tokenize(text: str):
    """Simple token extractor: contiguous Thai sequences or contiguous [A-Za-z0-9] sequences."""
    if not text:
        return []
    return _TOKEN_RE.findall(text)


def _topk_from_counts(counts: dict, k: int):
    """
    Return top-k tokens by frequency (desc). If ties, prefer slightly longer tokens.
    """
    return [
        w
        for w, _ in sorted(
            counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True
        )[:k]
    ]


def _extract_keywords_overlap(body: str, summary: str, topk: int = 10, minlen: int = 2):
    """
    Overlap-based keywords: tokens that appear in BOTH body and gold summary.
    Used for TRAIN to simulate keyword hints derived from references.
    """
    bw = [w for w in _simple_word_tokenize(body)]
    sw = set(_simple_word_tokenize(summary))
    counts = {}
    for w in bw:
        if w in sw and len(w) >= minlen:
            counts[w] = counts.get(w, 0) + 1
    return _topk_from_counts(counts, topk)


def _extract_keywords_unsupervised(body: str, topk: int = 10, minlen: int = 2):
    """
    Unsupervised keywords: frequency-based tokens from body only (no gold summary).
    Used for VALIDATION to keep a fair setting.
    """
    bw = [w for w in _simple_word_tokenize(body)]
    counts = {}
    for w in bw:
        if len(w) >= minlen:
            counts[w] = counts.get(w, 0) + 1
    return _topk_from_counts(counts, topk)


def _prefix_with_keywords(body: str, keywords, sep: str = ", "):
    """
    Construct input prefix: "Keywords: k1, k2, ... | Article: <body>"
    Fallback to "Article: <body>" if keywords are empty.
    """
    kw_str = sep.join(keywords) if keywords else ""
    if kw_str:
        return f"Keywords: {kw_str} | Article: {body}"
    else:
        return f"Article: {body}"


# --------------------------- Metrics helpers ---------------------------
def build_compute_metrics(tokenizer):
    """
    Build a compute_metrics callback for Seq2SeqTrainer that returns ROUGE-1/2/L.
    Handles label -100 -> pad_token_id and sanitizes invalid token ids.
    """
    rouge = evaluate.load("rouge")

    def _decode(ids):
        # Accept tuple from Trainer, coerce to ndarray, and sanitize ids
        if isinstance(ids, tuple):
            ids = ids[0]
        ids = np.asarray(ids, dtype=np.int64)
        pad_id = tokenizer.pad_token_id or 0
        vocab_len = len(tokenizer)
        ids[(ids < 0) | (ids >= vocab_len)] = pad_id
        return tokenizer.batch_decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

    def compute(eval_pred):
        preds, labels = eval_pred
        labels = labels.copy()
        labels[labels == -100] = tokenizer.pad_token_id
        pred_texts = _decode(preds)
        ref_texts = _decode(labels)
        r = rouge.compute(
            predictions=pred_texts, references=ref_texts, use_stemmer=False
        )
        return {"rouge1": r["rouge1"], "rouge2": r["rouge2"], "rougeL": r["rougeL"]}

    return compute


def count_params(model):
    """Return (total_params, trainable_params) for the given model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    """
    CLI entrypoint: trains mT5 with keyword-guided inputs.
    Follows train.py's training/eval strategies for fair comparison.
    """
    parser = argparse.ArgumentParser(
        description="Train mT5 on ThaiSum with keyword-guided input (fair comparison to full/LoRA)."
    )
    # common to train.py
    parser.add_argument(
        "--size", type=float, default=1.0, help="สัดส่วนของ train set ที่ใช้ (0–1]"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Keyword-mT5-ThaiSum",
        help="ชื่อโฟลเดอร์โมเดล ./model/<name>",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="ทับโฟลเดอร์เดิมหากมีอยู่แล้ว"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="จำลอง batch ใหญ่ขึ้น"
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5
    )  # เหมือน full fine-tune
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    # speed-up validation during training (เหมือน train.py)
    parser.add_argument(
        "--eval_max_samples", type=int, default=1200, help="จำกัดจำนวนตัวอย่าง validation"
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=64,
        help="ความยาวสรุประหว่างเทรน (greedy)",
    )

    # keyword options
    parser.add_argument(
        "--keyword_topk", type=int, default=10, help="จำนวนคีย์เวิร์ดสูงสุดต่อบทความ"
    )
    parser.add_argument(
        "--keyword_minlen", type=int, default=2, help="ความยาวขั้นต่ำของคีย์เวิร์ด (อักขระ)"
    )
    parser.add_argument(
        "--keyword_sep", type=str, default=", ", help="ตัวคั่นคีย์เวิร์ดเวลา prefix"
    )
    parser.add_argument(
        "--keyword_mode_train",
        choices=["overlap", "unsupervised"],
        default="overlap",
        help="โหมดสร้างคีย์เวิร์ดสำหรับ train (default: overlap กับสรุปทอง)",
    )
    parser.add_argument(
        "--keyword_mode_val",
        choices=["unsupervised", "overlap"],
        default="unsupervised",
        help="โหมดสร้างคีย์เวิร์ดสำหรับ validation (default: unsupervised จาก body เท่านั้น)",
    )

    args = parser.parse_args()

    # ===== Resolve output_dir =====
    output_dir = _resolve_output_dir(args.name, args.overwrite)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output dir: {output_dir}")

    # ===== Load dataset + tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False, use_fast=False)
    dataset = load_thaisum()  # expected splits: train / validation / test

    # ----- Choose train subset by --size -----
    total_train = len(dataset["train"])
    if 0 < args.size < 1.0:
        subset_size_est = int(total_train * args.size)
        dataset["train"] = dataset["train"].train_test_split(
            test_size=(1 - args.size), seed=args.seed
        )["train"]
        actual_train = len(dataset["train"])
        print(
            f"⚙️  Using {args.size*100:.0f}% of train → ~{subset_size_est} (actual {actual_train}) / {total_train}"
        )
    elif args.size == 1.0:
        print(f"✅ Using full training dataset → {total_train} samples")
    else:
        print("⚠️  --size should be in (0,1]; fallback to full dataset.")
        args.size = 1.0
        print(f"✅ Using full training dataset → {total_train} samples")

    # ===== Build keyword-guided inputs BEFORE tokenization =====
    def add_keywords_train(batch):
        """
        TRAIN: build inputs with keyword prefix.
        Mode 'overlap' uses gold summary overlap; 'unsupervised' uses body-only frequency.
        """
        bodies = batch["body"]
        sums = batch["summary"]
        out = []
        for b, s in zip(bodies, sums):
            if args.keyword_mode_train == "overlap":
                kws = _extract_keywords_overlap(
                    b, s, topk=args.keyword_topk, minlen=args.keyword_minlen
                )
            else:
                kws = _extract_keywords_unsupervised(
                    b, topk=args.keyword_topk, minlen=args.keyword_minlen
                )
            out.append(_prefix_with_keywords(b, kws, sep=args.keyword_sep))
        batch["body"] = out
        return batch

    def add_keywords_val(batch):
        """
        VALIDATION: default to 'unsupervised' keywords for fairness (no gold access).
        'overlap' is intentionally disallowed to avoid leaking reference info.
        """
        bodies = batch["body"]
        # default: unsupervised (ไม่ใช้สรุปทอง)
        out = []
        for b in bodies:
            if args.keyword_mode_val == "overlap":
                # If you want to experiment with overlap at val time, wire it manually.
                # Disallowed here by design for fair evaluation.
                raise ValueError(
                    "keyword_mode_val=overlap ไม่แนะนำและไม่ได้รองรับในสคริปต์นี้เพื่อความแฟร์"
                )
            kws = _extract_keywords_unsupervised(
                b, topk=args.keyword_topk, minlen=args.keyword_minlen
            )
            out.append(_prefix_with_keywords(b, kws, sep=args.keyword_sep))
        batch["body"] = out
        return batch

    dataset["train"] = dataset["train"].map(add_keywords_train, batched=True)
    if "validation" not in dataset:
        raise ValueError(
            "ไม่พบ validation split ใน dataset — กรุณาตรวจ preprocess/load_thaisum()"
        )
    dataset["validation"] = dataset["validation"].map(add_keywords_val, batched=True)

    # ---- Slice validation for faster evaluation ----
    eval_full = dataset["validation"]
    if args.eval_max_samples and args.eval_max_samples < len(eval_full):
        eval_ds_raw = eval_full.select(range(args.eval_max_samples))
        print(
            f"🧪 Validation subset for eval: {len(eval_ds_raw)} / {len(eval_full)} samples"
        )
    else:
        eval_ds_raw = eval_full
        print(f"🧪 Validation subset for eval: using full {len(eval_full)} samples")

    # ===== Preprocess =====
    tokenized = preprocess_dataset(dataset, tokenizer)
    train_ds = tokenized["train"]
    # Re-tokenize only the (possibly sliced) validation set to keep mapping consistent
    eval_ds = preprocess_dataset(DatasetDict({"validation": eval_ds_raw}), tokenizer)[
        "validation"
    ]

    # ===== Load model =====
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ----- Generation config for validation-time eval (FAST, GREEDY) -----
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = args.eval_max_new_tokens
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)
    # Trainer uses model.generation_config when predict_with_generate=True
    model.generation_config = gen_cfg

    # ===== Data collator =====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ===== Hardware & precision detection =====
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    # Prefer bf16 on Ampere+ (or MPS path), otherwise fp16 on CUDA if available
    use_bf16 = (use_cuda and torch.cuda.get_device_capability(0)[0] >= 8) or use_mps

    num_workers = 0 if use_mps else 4
    pin_memory = False if use_mps else True

    extra_args = dict(
        bf16=use_bf16,
        fp16=(use_cuda and not use_bf16),
    )

    device_info = "MPS" if use_mps else ("CUDA" if use_cuda else "CPU")
    print(
        f"🔧 Device: {device_info}, Workers: {num_workers}, BF16: {use_bf16}, FP16: {extra_args.get('fp16', False)}"
    )

    # ===== Training arguments =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,  # match full fine-tune (not LoRA's 1e-3)
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        save_total_limit=2,
        **extra_args,
    )

    # ===== Trainer =====
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=tokenizer,
    )

    # ===== Train =====
    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # ===== Extract best eval ROUGE + eval_loss from log history =====
    best = None
    best_loss = None
    last_eval = None
    for rec in trainer.state.log_history:
        # Track best ROUGE-L
        if "eval_rougeL" in rec:
            last_eval = rec
            if (best is None) or (rec["eval_rougeL"] > best["eval_rougeL"]):
                best = rec
        # Track best (lowest) eval_loss
        if "eval_loss" in rec:
            if (best_loss is None) or (rec["eval_loss"] < best_loss["eval_loss"]):
                best_loss = rec

    # Fallback guards when no eval logs were produced
    if best is None:
        best = {
            "eval_rouge1": None,
            "eval_rouge2": None,
            "eval_rougeL": None,
            "step": None,
        }
    if best_loss is None:
        best_loss = {"eval_loss": None, "step": None}
    if last_eval is None:
        last_eval = {"eval_loss": None, "eval_rougeL": None, "step": None}

    # ===== Save training-time metrics + hyperparams =====
    result_dir = os.path.join(output_dir, "result")
    os.makedirs(result_dir, exist_ok=True)

    tr_metrics = getattr(train_result, "metrics", {}) or {}
    payload = {
        "method": "keyword-guided",
        "model_base": MODEL_NAME,
        "output_dir": output_dir,
        "train_examples_used": len(train_ds),
        "eval_examples_used": len(eval_ds),
        "best_validation": {
            "rouge1": best["eval_rouge1"],
            "rouge2": best["eval_rouge2"],
            "rougeL": best["eval_rougeL"],
            "global_step": best.get("step"),
        },
        "loss_summary": {
            "best_eval_loss": best_loss["eval_loss"],
            "best_eval_loss_step": best_loss.get("step"),
            "last_eval_loss": last_eval.get("eval_loss"),
            "last_eval_rougeL": last_eval.get("eval_rougeL"),
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler_type": "linear",
            "seed": args.seed,
            "size_fraction": args.size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        },
        "generation_used_for_validation": {
            "decoding": "greedy (num_beams=1)",
            "max_new_tokens": args.eval_max_new_tokens,
        },
        "keyword_settings": {
            "topk": args.keyword_topk,
            "minlen": args.keyword_minlen,
            "sep": args.keyword_sep,
            "mode_train": args.keyword_mode_train,
            "mode_val": args.keyword_mode_val,
        },
        "hardware": {
            "device": device_info,
            "bf16": bool(extra_args.get("bf16", False)),
            "fp16": bool(extra_args.get("fp16", False)),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
        "training_time": {
            "train_runtime_seconds": tr_metrics.get("train_runtime", None),
            "train_loss": tr_metrics.get("train_loss", None),
            "train_samples_per_second": tr_metrics.get(
                "train_samples_per_second", None
            ),
            "train_steps_per_second": tr_metrics.get("train_steps_per_second", None),
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    # Persist JSON + readable TXT summaries
    with open(
        os.path.join(result_dir, "metrics_train.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    def pct(x):
        return "—" if x is None else f"{round(100*x, 2)}"

    with open(
        os.path.join(result_dir, "metrics_train_readable.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("🎯 Validation ROUGE during training (best by ROUGE-L)\n")
        f.write(f"- ROUGE-1 (F1): {pct(best['eval_rouge1'])}\n")
        f.write(f"- ROUGE-2 (F1): {pct(best['eval_rouge2'])}\n")
        f.write(f"- ROUGE-L (F1): {pct(best['eval_rougeL'])}\n\n")
        f.write("📉 Validation Loss\n")
        f.write(
            f"- Best eval_loss: {best_loss['eval_loss']} (step {best_loss.get('step')})\n"
        )
        f.write(
            f"- Last  eval_loss: {last_eval.get('eval_loss')} (step {last_eval.get('step')})\n\n"
        )
        f.write("⚙️ Hyperparameters\n")
        f.write(
            f"- epochs: {args.epochs}\n- batch_size: {args.batch_size}\n- learning_rate: {args.learning_rate}\n"
        )
        f.write(
            f"- warmup_ratio: {args.warmup_ratio}\n- lr_scheduler_type: linear\n- seed: {args.seed}\n"
        )
        f.write(
            f"- size_fraction: {args.size}\n- gradient_accumulation_steps: {args.gradient_accumulation_steps}\n"
        )
        f.write("\n🧪 Generation for validation\n")
        f.write(
            f"- decoding: greedy (num_beams=1)\n- max_new_tokens: {args.eval_max_new_tokens}\n"
        )
        f.write("\n🧷 Keyword Settings\n")
        f.write(
            f"- topk: {args.keyword_topk}\n- minlen: {args.keyword_minlen}\n- mode_train: {args.keyword_mode_train}\n- mode_val: {args.keyword_mode_val}\n"
        )
        f.write("\n⏱️ Training time\n")
        f.write(f"- runtime (s): {payload['training_time']['train_runtime_seconds']}\n")
        f.write(f"- train_loss: {payload['training_time']['train_loss']}\n")
        f.write(
            f"- samples/s: {payload['training_time']['train_samples_per_second']}\n"
        )
        f.write(f"- steps/s: {payload['training_time']['train_steps_per_second']}\n")

    print("✅ Training finished and model saved.")
    print(
        f"📊 Saved training-time metrics to: {os.path.join(result_dir, 'metrics_train.json')}"
    )
    print(
        f"📝 Readable metrics: {os.path.join(result_dir, 'metrics_train_readable.txt')}"
    )


if __name__ == "__main__":
    main()
