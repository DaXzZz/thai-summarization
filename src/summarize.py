import os, sys, argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import warnings

# ===== Silence some noisy warnings =====
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ===== Make ../src importable (for preprocess.py if needed) =====
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # ===== Args =====
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

    # ===== Generation hyperparameters =====
    parser.add_argument("--num_beams", type=int, default=4, help="Beam search width")
    parser.add_argument("--max_length", type=int, default=128, help="Max output tokens")
    parser.add_argument("--min_length", type=int, default=30, help="Min output tokens")
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.9,
        help="<1 favors shorter, >1 favors longer summaries",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help=">1 discourages repetition and copying",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="Truncate input text to this many tokens (same as training)",
    )
    args = parser.parse_args()

    # ===== Device detection =====
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    print(f"🔧 Device: {device}")

    # ===== Load model & tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)

    # ===== Configure generation settings =====
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

    # ----- Block T5 sentinel tokens (<extra_id_*>) -----
    bad = []
    for i in range(100):
        tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
        if tid not in (None, tokenizer.unk_token_id):
            bad.append([tid])
    if bad:
        gen_cfg.bad_words_ids = bad
    model.generation_config = gen_cfg

    # ===== Prepare input text =====
    # Combine prefix + text (for zero-shot T5/mT5)
    text = (args.input_prefix or "") + args.text.strip()
    enc = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=args.max_source_length,  # Match training input length
        padding=True,
    ).to(device)

    # ===== Generate summary =====
    print("🚀 Generating summary ...")
    with torch.no_grad():
        out_ids = model.generate(**enc)
    summary = tokenizer.decode(
        out_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    # ===== Display results =====
    print("\n📰 Input:")
    print(args.text.strip())
    print("\n🧠 Summary:")
    print(summary.strip())


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
