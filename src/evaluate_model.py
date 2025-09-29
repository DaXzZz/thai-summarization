# (optional) hide HF symlink warning on Windows
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys, argparse, datetime, json
import torch
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
)
import evaluate
import warnings

# ===== Quiet some noisy warnings =====
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ===== Import project-local preprocess helpers =====
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))
from preprocess import load_thaisum, preprocess_dataset  # noqa: E402


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def decode_text(tokenizer, ids):
    """Decode token ids → text."""
    return tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def sanitize_ids(ids, tokenizer):
    """Clamp invalid token ids to prevent SentencePiece 'id out of range'."""
    if isinstance(ids, tuple):  # some HF versions return (array, None, ...)
        ids = ids[0]
    ids = np.asarray(ids, dtype=np.int64)
    vocab_len = len(tokenizer)  # safer than tokenizer.vocab_size for SP
    pad_id = tokenizer.pad_token_id or 0
    ids[(ids < 0) | (ids >= vocab_len)] = pad_id
    return ids


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # ===== Args =====
    parser = argparse.ArgumentParser(
        description="Evaluate Seq2Seq model on ThaiSum (ROUGE & BERTScore)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF repo id (e.g., ../model/FineTuned-mT5-ThaiSum)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples for faster eval",
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Beam size for generation"
    )
    parser.add_argument(
        "--gen_max_len", type=int, default=128, help="Max generated length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="xlm-roberta-large",
        help="External model for BERTScore",
    )
    parser.add_argument(
        "--input_prefix",
        type=str,
        default="summarize: ",
        help="Prefix before input (e.g., 'summarize: ' for zero-shot)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Results",
        help="Folder name under ./data/ to save outputs",
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite existing result folder if exists",
    )
    args = parser.parse_args()

    # ===== Device pick =====
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    print(f"🔧 Device: {device}")
    if args.input_prefix:
        print(f"🧩 Input prefix: {repr(args.input_prefix)}")

    # ===== Load model/tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    # ===== GenerationConfig (future-proof, no deprecation) =====
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = args.num_beams
    gen_cfg.max_length = args.gen_max_len
    gen_cfg.early_stopping = True
    gen_cfg.do_sample = False
    gen_cfg.no_repeat_ngram_size = 3  # basic anti-repetition
    gen_cfg.length_penalty = 1.0
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)

    # Block <extra_id_*> sentinel tokens (common leak in zero-shot mT5)
    bad_tokens = []
    for i in range(100):
        tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        if tid not in (None, tokenizer.unk_token_id):
            bad_tokens.append([tid])
    if bad_tokens:
        gen_cfg.bad_words_ids = bad_tokens
    model.generation_config = gen_cfg

    # ===== Load dataset split =====
    raw = load_thaisum()
    ds = raw[args.split]
    total_len = len(ds)
    if args.max_samples:
        used_len = min(args.max_samples, total_len)
        ds = ds.select(range(used_len))  # take the first N for speed/debug
        pct = round((used_len / total_len) * 100, 2)
        print(
            f"📦 Loaded {args.split} split: {used_len} samples ({pct}% of total {total_len})"
        )
    else:
        used_len = total_len
        print(f"📦 Loaded {args.split} split: {total_len} samples (full dataset)")

    # ===== Optional input prefix for zero-shot T5/mT5 =====
    if args.input_prefix:

        def add_prefix(batch):
            batch["body"] = [args.input_prefix + x for x in batch["body"]]
            return batch

        ds = ds.map(add_prefix, batched=True)

    # ===== Preprocess (same as training) =====
    tokenized = preprocess_dataset(DatasetDict({args.split: ds}), tokenizer)[args.split]

    # ===== Resolve output dirs (timestamp only if name exists and no overwrite) =====
    base_dir = "./data"
    run_dir = os.path.join(base_dir, args.name)
    if os.path.exists(run_dir) and not args.overwrite_output_dir:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"⚠️  Folder '{args.name}' exists → creating '{args.name}_{ts}' instead")
        run_dir = os.path.join(base_dir, f"{args.name}_{ts}")
    score_dir = os.path.join(run_dir, "score")
    os.makedirs(score_dir, exist_ok=True)

    # ===== Trainer (predict_with_generate) =====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    eval_args = Seq2SeqTrainingArguments(
        output_dir=score_dir,  # keep temp/metrics under the run folder
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.gen_max_len,
        generation_num_beams=args.num_beams,
        dataloader_pin_memory=False if device == "mps" else True,
        report_to="none",
        save_strategy="no",
        logging_strategy="no",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=eval_args, data_collator=data_collator, eval_dataset=tokenized
    )

    # ===== Generate summaries =====
    print(
        f"🚀 Generating summaries (beams={args.num_beams}, max_len={args.gen_max_len}) ..."
    )
    preds = trainer.predict(tokenized)

    # ===== Decode safely (sanitize ids and -100 labels) =====
    pred_ids = sanitize_ids(preds.predictions, tokenizer)
    label_ids = np.asarray(preds.label_ids, dtype=np.int64)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids = sanitize_ids(label_ids, tokenizer)
    pred_texts = decode_text(tokenizer, pred_ids)
    ref_texts = decode_text(tokenizer, label_ids)

    # ===== Metrics: ROUGE & BERTScore =====
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(
        predictions=pred_texts, references=ref_texts, use_stemmer=False
    )

    bertscore = evaluate.load("bertscore")
    bs = bertscore.compute(
        predictions=pred_texts,
        references=ref_texts,
        model_type=args.bertscore_model,  # default: xlm-roberta-large
        device=device,  # use available accelerator
        batch_size=args.batch_size,
    )
    bs_p, bs_r, bs_f1 = map(
        lambda arr: float(np.mean(arr)), (bs["precision"], bs["recall"], bs["f1"])
    )

    # ===== Pretty print to console =====
    pct = lambda x: round(100 * x, 2)
    print("\n🎯 Results")
    print(f"- ROUGE-1 (F1): {pct(rouge_res['rouge1'])}")
    print(f"- ROUGE-2 (F1): {pct(rouge_res['rouge2'])}")
    print(f"- ROUGE-L (F1): {pct(rouge_res['rougeL'])}")
    print(f"- BERTScore P/R/F1: {pct(bs_p)} / {pct(bs_r)} / {pct(bs_f1)}")

    # ===== Save predictions/references =====
    split = args.split
    pred_path = os.path.join(run_dir, f"predictions_{split}.txt")
    ref_path = os.path.join(run_dir, f"references_{split}.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.writelines(t.strip() + "\n" for t in pred_texts)
    with open(ref_path, "w", encoding="utf-8") as f:
        f.writelines(t.strip() + "\n" for t in ref_texts)

    # ===== Save metrics (JSON + readable text) =====
    metrics = {
        "model": args.model,
        "split": args.split,
        "num_samples_used": used_len,
        "generation": {
            "num_beams": args.num_beams,
            "max_length": args.gen_max_len,
            "no_repeat_ngram_size": gen_cfg.no_repeat_ngram_size,
            "length_penalty": gen_cfg.length_penalty,
            "input_prefix": args.input_prefix,
        },
        "rouge": {
            "rouge1_f1": float(rouge_res["rouge1"]),
            "rouge2_f1": float(rouge_res["rouge2"]),
            "rougeL_f1": float(rouge_res["rougeL"]),
        },
        "bertscore": {
            "precision": bs_p,
            "recall": bs_r,
            "f1": bs_f1,
            "model": args.bertscore_model,
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
    print(
        f"📊 Saved metrics (json/text) to: {os.path.join(score_dir, 'metrics.json')} / metrics_readable.txt"
    )
    print(f"📁 Run folder: {run_dir}")


if __name__ == "__main__":
    main()
