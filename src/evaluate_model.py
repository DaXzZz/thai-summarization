# (optional) hide HF symlink warning on Windows; put this before other imports
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys, argparse
import torch
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,  # ← use GenerationConfig to avoid deprecated warning
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


# ---- helpers ----
def decode_text(tokenizer, ids):
    """Decode token ids to text (ids should be sanitized already)."""
    return tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def sanitize_ids(ids, tokenizer):
    """
    Prevent SentencePiece 'piece id is out of range' by clamping invalid ids.
    Some HF versions return predictions as a tuple; also ensure int dtype.
    """
    if isinstance(ids, tuple):
        ids = ids[0]
    ids = np.asarray(ids, dtype=np.int64)
    vocab_len = len(tokenizer)  # safer than tokenizer.vocab_size for SP models
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bad = (ids < 0) | (ids >= vocab_len)
    if np.any(bad):
        ids[bad] = pad_id
    return ids


def main():
    # ---- args ----
    parser = argparse.ArgumentParser(
        description="Evaluate Seq2Seq model on ThaiSum (ROUGE & BERTScore)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path or HF repo id (e.g., /Model/FineTuned-mT5-ThaiSum or google/mt5-small)",
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
        help="Limit number of samples for speed",
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Beam size for generation"
    )
    parser.add_argument(
        "--gen_max_len", type=int, default=128, help="Max generated length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Per-device eval batch size"
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
        help="Prefix before input (e.g., 'summarize: ' for T5/mT5 zero-shot)",
    )
    args = parser.parse_args()

    # ---- device pick (CUDA → MPS → CPU) ----
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    print(f"Device: {device}")
    if args.input_prefix:
        print(f"Input prefix: {repr(args.input_prefix)}")

    # ---- load model & tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    # use GenerationConfig (future-proof, no deprecation warning)
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = args.num_beams
    gen_cfg.max_length = args.gen_max_len
    gen_cfg.early_stopping = True
    gen_cfg.do_sample = False
    gen_cfg.no_repeat_ngram_size = 3  # basic anti-repetition
    gen_cfg.length_penalty = 1.0
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)
    # discourage T5 span tokens like <extra_id_0..99> that often leak in zero-shot
    bad_tokens = []
    for i in range(100):
        tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        if tid is not None and tid != tokenizer.unk_token_id:
            bad_tokens.append([tid])
    if bad_tokens:
        gen_cfg.bad_words_ids = bad_tokens
    model.generation_config = gen_cfg  # ← attach to model

    # ---- load dataset & (optional) prefix ----
    raw = load_thaisum()
    ds = raw[args.split]
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    if args.input_prefix:
        # prepend prefix to news body so mT5 understands "summarize" in zero-shot
        def add_prefix(batch):
            batch["body"] = [args.input_prefix + x for x in batch["body"]]
            return batch

        ds = ds.map(add_prefix, batched=True)

    # ---- preprocess (same as training) ----
    tokenized = preprocess_dataset(DatasetDict({args.split: ds}), tokenizer)[args.split]

    # ---- trainer for generation-based prediction ----
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    eval_args = Seq2SeqTrainingArguments(
        output_dir="../data/eval_tmp",  # scratch dir; metrics & cache
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,  # generate summaries
        generation_max_length=args.gen_max_len,
        generation_num_beams=args.num_beams,
        dataloader_pin_memory=False if device == "mps" else True,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        eval_dataset=tokenized,
        processing_class=tokenizer,  # ok for transformers versions that accept this
    )

    # ---- generate ----
    print(
        f"Generating summaries (beams={args.num_beams}, max_len={args.gen_max_len}) ..."
    )
    preds = trainer.predict(tokenized)

    # ---- sanitize & decode ----
    pred_ids = sanitize_ids(preds.predictions, tokenizer)
    label_ids = np.asarray(preds.label_ids, dtype=np.int64)
    label_ids[label_ids == -100] = tokenizer.pad_token_id  # align ignore index
    label_ids = sanitize_ids(label_ids, tokenizer)

    pred_texts = decode_text(tokenizer, pred_ids)
    ref_texts = decode_text(tokenizer, label_ids)

    # ---- metrics ----
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(
        predictions=pred_texts, references=ref_texts, use_stemmer=False
    )

    print(f"Using BERTScore model: {args.bertscore_model}")
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

    # ---- report ----
    pct = lambda x: round(100 * x, 2)
    print("\nResults")
    print(f"- ROUGE-1 (F1): {pct(rouge_res['rouge1'])}")
    print(f"- ROUGE-2 (F1): {pct(rouge_res['rouge2'])}")
    print(f"- ROUGE-L (F1): {pct(rouge_res['rougeL'])}")
    print(f"- BERTScore P/R/F1: {pct(bs_p)} / {pct(bs_r)} / {pct(bs_f1)}")

    # ---- save outputs ----
    out_dir = (
        os.path.join(args.model, "eval_outputs")
        if os.path.isdir(args.model)
        else "./eval_outputs"
    )
    os.makedirs(out_dir, exist_ok=True)
    split = args.split
    with open(os.path.join(out_dir, f"pred_{split}.txt"), "w", encoding="utf-8") as f:
        for t in pred_texts:
            f.write(t.strip() + "\n")
    with open(os.path.join(out_dir, f"ref_{split}.txt"), "w", encoding="utf-8") as f:
        for t in ref_texts:
            f.write(t.strip() + "\n")
    print(f"\nSaved predictions to: {out_dir}/pred_{split}.txt")
    print(f"Saved references  to: {out_dir}/ref_{split}.txt")


if __name__ == "__main__":
    main()