# summarize.py — Generate summary from custom input text using any trained model

import os, sys, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import warnings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))


def main():
    parser = argparse.ArgumentParser(description="Summarize custom input text")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path or HF id (e.g. ./model/FineTuned-100 or google/mt5-small)",
    )
    parser.add_argument(
        "--text", required=True, type=str, help="Input text to summarize"
    )
    parser.add_argument(
        "--input_prefix",
        default="",
        type=str,
        help="Prefix for zero-shot T5/mT5, e.g. 'summarize: '",
    )
    # --- new knobs ---
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size")
    parser.add_argument("--max_length", type=int, default=128, help="Max output tokens")
    parser.add_argument("--min_length", type=int, default=30, help="Min output tokens")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.9,
        help="<1.0 favors shorter, >1.0 favors longer",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1, help=">1.0 discourages copying"
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="Truncate input to this many tokens (match training)",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print(f"🔧 Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    # Generation config (stronger push to summarize)
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = args.num_beams
    gen_cfg.max_length = args.max_length
    gen_cfg.min_length = args.min_length
    gen_cfg.length_penalty = args.length_penalty
    gen_cfg.repetition_penalty = args.repetition_penalty
    gen_cfg.early_stopping = True
    gen_cfg.do_sample = False
    gen_cfg.no_repeat_ngram_size = 3
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)

    # block T5 sentinel tokens
    bad = []
    for i in range(100):
        tok_id = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        if tok_id not in (None, tokenizer.unk_token_id):
            bad.append([tok_id])
    if bad:
        gen_cfg.bad_words_ids = bad
    model.generation_config = gen_cfg

    # Prepare input (truncate to match training)
    text = (args.input_prefix or "") + args.text.strip()
    enc = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_source_length,  # <-- สำคัญ: กำหนดความยาวอินพุต
        padding=True,
    ).to(device)

    print("🚀 Generating summary ...")
    with torch.no_grad():
        out_ids = model.generate(**enc)
    summary = tokenizer.decode(
        out_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    print("\n📰 Input:")
    print(args.text.strip())
    print("\n🧠 Summary:")
    print(summary.strip())


if __name__ == "__main__":
    main()
