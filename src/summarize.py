"""
Summarize a custom input string with a Seq2Seq model (e.g., mT5 fine-tuned on ThaiSum).

Features:
- Auto device selection (CUDA / MPS / CPU)
- Clean, safe tokenization (truncates to 512)
- Beam search decoding with length penalty
- Prints both input and generated summary
"""

import os, sys, argparse, warnings
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

# ===== Silence some noisy warnings =====
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ===== Make ../src importable if needed =====
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))


def main():
    # ===== Args =====
    parser = argparse.ArgumentParser(description="Summarize custom input text")
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path or HF id (e.g., ./Model/FineTuned-ThaiSum or google/mt5-small)",
    )
    parser.add_argument(
        "--text", required=True, type=str, help="Input text to summarize"
    )
    args = parser.parse_args()

    # ===== Device selection =====
    # Prefer CUDA, otherwise Apple MPS, otherwise CPU
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    device = "cuda" if torch.cuda.is_available() else ("mps" if use_mps else "cpu")
    print(f"🔧 Device: {device}")

    # ===== Load model & tokenizer =====
    # Note: legacy=False + use_fast=False to avoid slow-tokenizer warnings and ensure stability
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()

    # ===== Encode input =====
    # Truncate long inputs to 512 tokens (encoder side) to avoid OOM / excessive latency
    enc = tokenizer(
        [args.text.strip()],
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    # ===== Generation config (aligned with evaluation script style) =====
    # Beam search for higher-quality summaries; limit length and bias toward brevity
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.num_beams = 4
    gen_cfg.max_new_tokens = 128
    gen_cfg.length_penalty = 0.8  # <1 encourages shorter outputs
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)
    model.generation_config = gen_cfg

    # ===== Generate =====
    print(
        f"🚀 Generating summary ... (beams=4, max_new_tokens=128, length_penalty=0.8)"
    )
    with torch.no_grad():
        out_ids = model.generate(**enc)
    summary = tokenizer.decode(
        out_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).strip()

    # ===== Output =====
    print("\n📰 Input:")
    print(args.text.strip())
    print("\n🧠 Summary:")
    print(summary)


if __name__ == "__main__":
    main()
