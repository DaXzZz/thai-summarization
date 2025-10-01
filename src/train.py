import torch
import sys, os, argparse
from datetime import datetime
import warnings
import json
import numpy as np
import evaluate

# ===== Silence some noisy warnings =====
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

# (Optional) PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


# -----------------------------------------------------------------------------
# Path resolver: ./model/<name> with "timestamp only if exists (and no --overwrite)"
# -----------------------------------------------------------------------------
def _resolve_output_dir(name: str, overwrite: bool) -> str:
    base = os.path.join(".", "model", name)
    if os.path.isdir(base) and not overwrite:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        print(f"⚠️  Output dir exists: {base} → will use timestamped dir instead.")
        base = f"{base}-{ts}"
    return base


def build_compute_metrics(tokenizer):
    rouge = evaluate.load("rouge")

    def _decode(ids):
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
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # ===== Args =====
    parser = argparse.ArgumentParser(
        description="Train mT5 on ThaiSum dataset (with validation ROUGE, optional LoRA, and result export)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=1.0,
        help="สัดส่วนของ train set ที่ใช้ (0–1), เช่น 0.4 = 40% (default=1.0 ใช้ทั้งหมด)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="FineTuned-mT5-ThaiSum",
        help="ตั้งชื่อโฟลเดอร์โมเดล (จะเซฟที่ ./model/<name>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ทับโฟลเดอร์โมเดลเดิมถ้ามีอยู่แล้ว (ระวังข้อมูลหาย)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="จำนวน step ที่จะสะสม gradient ก่อนอัปเดต (ช่วยจำลอง batch ใหญ่ขึ้น)",
    )

    # train hyperparams
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    # speed-up validation during training
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=1200,
        help="จำกัดจำนวนตัวอย่าง validation ที่ใช้คำนวณ ROUGE ระหว่างเทรน (default=1200)",
    )
    parser.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=64,
        help="ความยาวสรุประหว่างเทรน (default=64) — greedy decoding",
    )

    # ===== LoRA switches =====
    parser.add_argument(
        "--lora", action="store_true", help="เปิดโหมด Low-Rank Adaptation (PEFT)"
    )
    parser.add_argument(
        "--lora_r", type=int, default=8, help="rank ของ LoRA (default=8)"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="alpha ของ LoRA (default=16)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="dropout ใน LoRA (default=0.05)",
    )
    parser.add_argument(
        "--lora_target",
        type=str,
        default="q,k,v,o,wi_0,wi_1,wo",
        help="คอมมาแยกรายชื่อโมดูลเป้าหมาย (T5/mT5 แนะนำ: q,k,v,o,wi_0,wi_1,wo)",
    )

    args = parser.parse_args()

    # ===== Resolve output_dir =====
    output_dir = _resolve_output_dir(args.name, args.overwrite)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output dir: {output_dir}")

    # ===== Load dataset + tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False, use_fast=False)
    dataset = load_thaisum()  # ควรมี train / validation / test

    # ----- Choose train subset by --size (random, reproducible) -----
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

    # ===== Preprocess =====
    tokenized = preprocess_dataset(dataset, tokenizer)
    train_ds = tokenized["train"]
    eval_full = tokenized.get("validation", None)
    if eval_full is None:
        raise ValueError(
            "ไม่พบ validation split ใน dataset — กรุณาตรวจ preprocess/load_thaisum()"
        )

    # ---- Slice validation for faster evaluation ----
    if args.eval_max_samples and args.eval_max_samples < len(eval_full):
        eval_ds = eval_full.select(range(args.eval_max_samples))
        print(
            f"🧪 Validation subset for eval: {len(eval_ds)} / {len(eval_full)} samples"
        )
    else:
        eval_ds = eval_full
        print(f"🧪 Validation subset for eval: using full {len(eval_ds)} samples")

    # ===== Load base model =====
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ----- Optionally wrap with LoRA -----
    lora_info = None
    if args.lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("ต้องติดตั้ง peft ก่อนใช้งาน LoRA: pip install peft")
        target_modules = [s.strip() for s in args.lora_target.split(",") if s.strip()]
        lcfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
        lora_info = {
            "enabled": True,
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        }
    else:
        lora_info = {"enabled": False}

    # ----- Generation config for validation-time eval (FAST, GREEDY) -----
    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.max_new_tokens = args.eval_max_new_tokens
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = getattr(tokenizer, "eos_token_id", gen_cfg.eos_token_id)
    model.generation_config = gen_cfg  # ให้ Trainer อ่านค่านี้ตอน predict_with_generate=True

    # ===== Data collator =====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ===== Hardware & precision detection =====
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
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

    # ===== Training arguments (with validation ROUGE) =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",  # รองรับเวอร์ชัน transformers เก่า
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
    # Save model/adapters
    trainer.save_model(training_args.output_dir)  # PEFT จะเซฟ adapter ถ้าเป็น LoRA
    tokenizer.save_pretrained(training_args.output_dir)

    # ===== Extract best eval ROUGE + eval_loss from log history =====
    best = None
    best_loss = None
    last_eval = None
    for rec in trainer.state.log_history:
        if "eval_rougeL" in rec:
            last_eval = rec
            if (best is None) or (rec["eval_rougeL"] > best["eval_rougeL"]):
                best = rec
        if "eval_loss" in rec:
            if (best_loss is None) or (rec["eval_loss"] < best_loss["eval_loss"]):
                best_loss = rec

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

    # training runtime info from trainer/train_result
    tr_metrics = getattr(train_result, "metrics", {}) or {}
    train_runtime_sec = tr_metrics.get("train_runtime", None)
    train_loss = tr_metrics.get("train_loss", None)
    train_sps = tr_metrics.get("train_samples_per_second", None)
    train_stepsps = tr_metrics.get("train_steps_per_second", None)

    total_params, trainable_params = count_params(model)

    payload = {
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
        },
        "generation_used_for_validation": {
            "decoding": "greedy (num_beams=1)",
            "max_new_tokens": args.eval_max_new_tokens,
        },
        "lora": {
            **lora_info,
            "trainable_params": int(trainable_params),
            "total_params": int(total_params),
            "trainable_ratio_pct": (
                round(100.0 * trainable_params / total_params, 4)
                if total_params
                else None
            ),
        },
        "hardware": {
            "device": device_info,
            "bf16": bool(extra_args.get("bf16", False)),
            "fp16": bool(extra_args.get("fp16", False)),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        },
        "training_time": {
            "train_runtime_seconds": train_runtime_sec,
            "train_loss": train_loss,
            "train_samples_per_second": train_sps,
            "train_steps_per_second": train_stepsps,
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    with open(
        os.path.join(result_dir, "metrics_train.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(
        os.path.join(result_dir, "metrics_train_readable.txt"), "w", encoding="utf-8"
    ) as f:

        def pct(x):
            return "—" if x is None else f"{round(100*x, 2)}"

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
        f.write(f"- size_fraction: {args.size}\n")
        f.write("\n🧪 Generation for validation\n")
        f.write(
            f"- decoding: greedy (num_beams=1)\n- max_new_tokens: {args.eval_max_new_tokens}\n"
        )
        f.write("\n🧩 LoRA\n")
        if lora_info["enabled"]:
            f.write(
                f"- enabled: True\n- r: {lora_info['r']}\n- alpha: {lora_info['alpha']}\n- dropout: {lora_info['dropout']}\n"
            )
            f.write(f"- target_modules: {', '.join(lora_info['target_modules'])}\n")
        else:
            f.write("- enabled: False\n")
        f.write(
            f"- trainable params: {int(trainable_params)} / total: {int(total_params)} "
            f"({round(100.0 * trainable_params / total_params, 4) if total_params else '—'}%)\n"
        )
        f.write("\n⏱️ Training time\n")
        f.write(
            f"- runtime (s): {payload['training_time']['train_runtime_seconds']}\n- train_loss: {payload['training_time']['train_loss']}\n"
        )
        f.write(
            f"- samples/s: {payload['training_time']['train_samples_per_second']}\n- steps/s: {payload['training_time']['train_steps_per_second']}\n"
        )

    print("✅ Training finished and model saved.")
    print(
        f"📊 Saved training-time metrics to: {os.path.join(result_dir, 'metrics_train.json')}"
    )
    print(
        f"📝 Readable metrics: {os.path.join(result_dir, 'metrics_train_readable.txt')}"
    )


if __name__ == "__main__":
    main()
