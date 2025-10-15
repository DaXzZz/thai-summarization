"""
Evaluate a seq2seq summarization model on ThaiSum with ROUGE and BERTScore.

Highlights:
- Supports validation/test split with optional sample cap for quick runs
- Optional unsupervised keyword prefixing to match keyword-trained models
- Beam-search decoding (num_beams=4) with length penalty
- Saves predictions, references, inputs, and machine-readable metrics
"""

import os, sys, argparse, datetime, json
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import torch
import numpy as np
import evaluate
from collections import Counter
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)

# ===== Make ../src importable (for preprocess.py) =====
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))
from preprocess import load_thaisum, preprocess_dataset  # noqa: E402


# --------------------------- Helpers ---------------------------
def _decode(tokenizer, ids):
    """
    Safe batch decode helper: handles tuple inputs, coerces to ndarray,
    clamps invalid token ids to pad, and strips special tokens.
    """
    if isinstance(ids, tuple):
        ids = ids[0]
    ids = np.asarray(ids, dtype=np.int64)
    vocab_len = len(tokenizer)
    pad_id = tokenizer.pad_token_id or 0
    ids[(ids < 0) | (ids >= vocab_len)] = pad_id
    return tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


# --- very light-weight unsupervised keyword extractor (source-only) ---
# keep Thai block + word chars + spaces; drop other punctuation
_TH_PUNCT_RE = re.compile(r"[^\w\s\u0E00-\u0E7F]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _simple_keywords_from_text(text, topk=10, minlen=2):
    """
    Minimal keyword extractor (frequency-based):
      1) strip non-Thai/word punctuation, normalize spaces
      2) whitespace tokenization
      3) filter too-short tokens and pure digits
      4) top-k by frequency
    """
    # 1) normalize
    t = _TH_PUNCT_RE.sub(" ", text)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()

    # 2) crude tokenization
    toks = [tok for tok in t.split(" ") if tok]

    # 3) filter
    def ok(tok):
        if len(tok) < minlen:
            return False
        return not tok.isdigit()

    toks = [tok for tok in toks if ok(tok)]

    # 4) top-k by frequency
    cnt = Counter(toks)
    kws = [w for w, _ in cnt.most_common(topk)]
    return kws


def _build_prefixed_inputs(ds, topk, minlen):
    """
    Map over dataset to prepend unsupervised keywords:
      "Keywords: k1, k2, ... | Article: <body>"
    This must mirror how the model was trained if --use_keywords is set.
    """

    def add_kw_prefix(batch):
        bodies = batch["body"]
        new_bodies = []
        for body in bodies:
            kws = _simple_keywords_from_text(body, topk=topk, minlen=minlen)
            kw_str = ", ".join(kws) if kws else ""
            new_bodies.append(f"Keywords: {kw_str} | Article: {body}")
        batch["body"] = new_bodies
        return batch

    return ds.map(add_kw_prefix, batched=True)


# ----------------------------- Main ----------------------------
def main():
    """
    CLI entrypoint: loads model + ThaiSum split, optionally prefixes keywords,
    generates summaries with beam search, computes ROUGE/BERTScore, and saves artifacts.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate model on ThaiSum (ROUGE & BERTScore) — LITE"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path or HF id (e.g., ./model/FineTuned-100 or google/mt5-small)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["validation", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit #samples for faster eval (optional)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device eval batch size (default: 8)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Results",
        help="Folder under ./data/ to save outputs",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite ./data/{name} if exists",
    )
    parser.add_argument(
        "--use_keywords",
        action="store_true",
        help="Prepend unsupervised keywords from the article: "
        "'Keywords: ... | Article: <body>' (must match how the model was trained).",
    )
    parser.add_argument(
        "--keyword_topk",
        type=int,
        default=10,
        help="How many keywords to prepend when --use_keywords (default: 10).",
    )
    parser.add_argument(
        "--keyword_minlen",
        type=int,
        default=2,
        help="Minimum token length for keywords when --use_keywords (default: 2).",
    )

    args = parser.parse_args()

    # ===== Device =====
    # Prefer CUDA, otherwise Apple MPS, otherwise CPU
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    print(f"🔧 Device: {device}")

    # ===== Load model/tokenizer =====
    # legacy=False + use_fast=False to keep tokenizer behavior stable across runs
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    # ===== Load dataset =====
    raw = load_thaisum()
    ds = raw[args.split]
    total_len = len(ds)
    if args.max_samples:
        used_len = min(args.max_samples, total_len)
        ds = ds.select(range(used_len))
        pct = round((used_len / total_len) * 100, 2)
        print(f"📦 Loaded {args.split}: {used_len} samples ({pct}% of {total_len})")
    else:
        used_len = total_len
        print(f"📦 Loaded {args.split}: {total_len} samples (full)")

    # ===== OPTIONAL: prepend keywords to inputs (to match keyword-trained model) =====
    if args.use_keywords:
        print(
            f"🧩 Using unsupervised keywords (topk={args.keyword_topk}, minlen={args.keyword_minlen})"
        )
        ds = _build_prefixed_inputs(ds, args.keyword_topk, args.keyword_minlen)

    # ===== Preprocess =====
    tokenized = preprocess_dataset(DatasetDict({args.split: ds}), tokenizer)[args.split]

    # ===== Generation Config (match our usual eval defaults) =====
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = 4
    gen_cfg.max_new_tokens = 128
    gen_cfg.length_penalty = 0.8
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)
    model.generation_config = gen_cfg

    # ===== Prepare run folders =====
    base_dir = "./data"
    run_dir = os.path.join(base_dir, args.name)
    if os.path.exists(run_dir) and not args.overwrite_output_dir:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"⚠️  Folder exists → creating '{args.name}_{ts}' instead")
        run_dir = os.path.join(base_dir, f"{args.name}_{ts}")
    score_dir = os.path.join(run_dir, "score")
    os.makedirs(score_dir, exist_ok=True)

    # ===== Trainer (predict_with_generate) =====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    eval_args = Seq2SeqTrainingArguments(
        output_dir=score_dir,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        dataloader_pin_memory=False if device == "mps" else True,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=eval_args, data_collator=data_collator, eval_dataset=tokenized
    )

    # ===== Generate =====
    print(f"🚀 Generating summaries (beams=4, max_new_tokens=128) ...")
    preds = trainer.predict(tokenized)

    # ===== Decode safely =====
    pred_ids = preds.predictions
    label_ids = preds.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_texts = _decode(tokenizer, pred_ids)
    ref_texts = _decode(tokenizer, label_ids)

    # ===== Metrics =====
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(
        predictions=pred_texts, references=ref_texts, use_stemmer=False
    )

    bertscore = evaluate.load("bertscore")
    bs = bertscore.compute(
        predictions=pred_texts,
        references=ref_texts,
        model_type="xlm-roberta-large",
        device=device,
        batch_size=args.batch_size,
    )
    bs_p = float(np.mean(bs["precision"]))
    bs_r = float(np.mean(bs["recall"]))
    bs_f1 = float(np.mean(bs["f1"]))

    # ===== Print =====
    pct = lambda x: round(100 * x, 2)
    print("\n🎯 Results")
    print(f"- ROUGE-1 (F1): {pct(rouge_res['rouge1'])}")
    print(f"- ROUGE-2 (F1): {pct(rouge_res['rouge2'])}")
    print(f"- ROUGE-L (F1): {pct(rouge_res['rougeL'])}")
    print(f"- BERTScore P/R/F1: {pct(bs_p)} / {pct(bs_r)} / {pct(bs_f1)}")

    # ===== Save outputs =====
    split = args.split
    pred_path = os.path.join(run_dir, f"predictions_{split}.txt")
    ref_path = os.path.join(run_dir, f"references_{split}.txt")
    inp_path = os.path.join(run_dir, f"inputs_{split}.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.writelines(t.strip() + "\n" for t in pred_texts)
    with open(ref_path, "w", encoding="utf-8") as f:
        f.writelines(t.strip() + "\n" for t in ref_texts)
    # Save the actual inputs used (post-prefix if --use_keywords)
    with open(inp_path, "w", encoding="utf-8") as f:
        raw_inputs = ds["body"]
        f.writelines(t.strip() + "\n" for t in raw_inputs)

    metrics = {
        "model": args.model,
        "split": args.split,
        "num_samples_used": used_len,
        "generation_defaults": {
            "num_beams": 4,
            "max_new_tokens": 128,
            "length_penalty": 0.8,
        },
        "used_keywords": bool(args.use_keywords),
        "keyword_settings": (
            {
                "topk": args.keyword_topk,
                "minlen": args.keyword_minlen,
            }
            if args.use_keywords
            else None
        ),
        "rouge": {
            "rouge1_f1": float(rouge_res["rouge1"]),
            "rouge2_f1": float(rouge_res["rouge2"]),
            "rougeL_f1": float(rouge_res["rougeL"]),
        },
        "bertscore": {
            "precision": bs_p,
            "recall": bs_r,
            "f1": bs_f1,
            "model": "xlm-roberta-large",
        },
        "batch_size": args.batch_size,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(score_dir, "metrics.json"), "w", encoding="utf-8") as jf:
        json.dump(metrics, jf, ensure_ascii=False, indent=2)
    with open(
        os.path.join(score_dir, "metrics_readable.txt"), "w", encoding="utf-8"
    ) as tf:
        tf.write("🎯 Results\n")
        tf.write(f"- ROUGE-1 (F1): {pct(rouge_res['rouge1'])}\n")
        tf.write(f"- ROUGE-2 (F1): {pct(rouge_res['rouge2'])}\n")
        tf.write(f"- ROUGE-L (F1): {pct(rouge_res['rougeL'])}\n")
        tf.write(f"- BERTScore P/R/F1: {pct(bs_p)} / {pct(bs_r)} / {pct(bs_f1)}\n")

    print(f"\n📝 Saved predictions to: {pred_path}")
    print(f"📝 Saved references  to: {ref_path}")
    print(f"📝 Saved inputs      to: {inp_path}")
    print(
        f"📊 Saved metrics to: {os.path.join(score_dir, 'metrics.json')} (and metrics_readable.txt)"
    )
    print(f"📁 Run folder: {run_dir}")


if __name__ == "__main__":
    main()
