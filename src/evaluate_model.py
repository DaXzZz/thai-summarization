# (optional) hide HF symlink warning on Windows
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys, argparse, datetime
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

# quiet down tokenizer-related warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# import project-local preprocess helpers
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))
from preprocess import load_thaisum, preprocess_dataset  # noqa: E402


# ---- helper functions ----
def decode_text(tokenizer, ids):
    """Decode token ids → text."""
    return tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def sanitize_ids(ids, tokenizer):
    """Clamp invalid token ids to prevent SentencePiece 'id out of range'."""
    if isinstance(ids, tuple):
        ids = ids[0]
    ids = np.asarray(ids, dtype=np.int64)
    vocab_len = len(tokenizer)
    pad_id = tokenizer.pad_token_id or 0
    ids[(ids < 0) | (ids >= vocab_len)] = pad_id
    return ids


# ---- main ----
def main():
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
        default="",
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

    # ---- device ----
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    print(f"🔧 Device: {device}")
    if args.input_prefix:
        print(f"🧩 Input prefix: {repr(args.input_prefix)}")

    # ---- load model/tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    # generation config (cleaner than overriding model.config)
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = args.num_beams
    gen_cfg.max_length = args.gen_max_len
    gen_cfg.early_stopping = True
    gen_cfg.do_sample = False
    gen_cfg.no_repeat_ngram_size = 3
    gen_cfg.length_penalty = 1.0
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)

    # block <extra_id_*> tokens (common in zero-shot mT5)
    bad_tokens = []
    for i in range(100):
        tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        if tid not in (None, tokenizer.unk_token_id):
            bad_tokens.append([tid])
    if bad_tokens:
        gen_cfg.bad_words_ids = bad_tokens
    model.generation_config = gen_cfg

    # ---- load dataset ----
    raw = load_thaisum()
    ds = raw[args.split]
    total_len = len(ds)

    if args.max_samples:
        used_len = min(args.max_samples, total_len)
        ds = ds.select(range(used_len))
        pct = round((used_len / total_len) * 100, 2)
        print(
            f"📦 Loaded {args.split} split: {used_len} samples ({pct}% of total {total_len})"
        )
    else:
        print(f"📦 Loaded {args.split} split: {total_len} samples (full dataset)")

    # add prefix if specified
    if args.input_prefix:

        def add_prefix(batch):
            batch["body"] = [args.input_prefix + x for x in batch["body"]]
            return batch

        ds = ds.map(add_prefix, batched=True)

    # ---- preprocess ----
    tokenized = preprocess_dataset(DatasetDict({args.split: ds}), tokenizer)[args.split]

    # ---- Trainer ----
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    eval_args = Seq2SeqTrainingArguments(
        output_dir="./data/eval_tmp",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=args.gen_max_len,
        generation_num_beams=args.num_beams,
        dataloader_pin_memory=False if device == "mps" else True,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=eval_args, data_collator=data_collator, eval_dataset=tokenized
    )

    # ---- generate ----
    print(
        f"🚀 Generating summaries (beams={args.num_beams}, max_len={args.gen_max_len}) ..."
    )
    preds = trainer.predict(tokenized)

    # ---- decode safely ----
    pred_ids = sanitize_ids(preds.predictions, tokenizer)
    label_ids = np.asarray(preds.label_ids, dtype=np.int64)
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids = sanitize_ids(label_ids, tokenizer)

    pred_texts = decode_text(tokenizer, pred_ids)
    ref_texts = decode_text(tokenizer, label_ids)

    # ---- compute metrics ----
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(
        predictions=pred_texts, references=ref_texts, use_stemmer=False
    )
    bertscore = evaluate.load("bertscore")
    bs = bertscore.compute(
        predictions=pred_texts,
        references=ref_texts,
        model_type=args.bertscore_model,
        device=device,
        batch_size=args.batch_size,
    )
    bs_p, bs_r, bs_f1 = map(
        lambda arr: float(np.mean(arr)), (bs["precision"], bs["recall"], bs["f1"])
    )

    # ---- results ----
    pct = lambda x: round(100 * x, 2)
    print("\n🎯 Results")
    print(f"- ROUGE-1 (F1): {pct(rouge_res['rouge1'])}")
    print(f"- ROUGE-2 (F1): {pct(rouge_res['rouge2'])}")
    print(f"- ROUGE-L (F1): {pct(rouge_res['rougeL'])}")
    print(f"- BERTScore P/R/F1: {pct(bs_p)} / {pct(bs_r)} / {pct(bs_f1)}")

    # ---- save ----
    base_dir = "./data"
    if args.overwrite_output_dir:
        out_dir = os.path.join(base_dir, args.name)
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = os.path.join(base_dir, f"{args.name}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    split = args.split
    with open(os.path.join(out_dir, f"pred_{split}.txt"), "w", encoding="utf-8") as f:
        f.writelines(t.strip() + "\n" for t in pred_texts)
    with open(os.path.join(out_dir, f"ref_{split}.txt"), "w", encoding="utf-8") as f:
        f.writelines(t.strip() + "\n" for t in ref_texts)

    print(f"\n📝 Saved predictions to: {out_dir}/pred_{split}.txt")
    print(f"📝 Saved references  to: {out_dir}/ref_{split}.txt")


if __name__ == "__main__":
    main()
