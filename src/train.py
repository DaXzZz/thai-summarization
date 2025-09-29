import torch
import sys, os, argparse
from datetime import datetime
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

# ให้ import preprocess.py ได้ (อยู่ใน ../src)
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from preprocess import load_thaisum, preprocess_dataset, MODEL_NAME


def _resolve_output_dir(name: str, overwrite: bool) -> str:
    """สร้าง path ./model/<name>, ถ้ามีแล้วจะเติม timestamp กันทับ (เว้นแต่ใช้ --overwrite)"""
    base = os.path.join(".", "model", name)
    if os.path.isdir(base) and not overwrite:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M")
        print(f"⚠️  Output dir exists: {base} → will use timestamped dir instead.")
        base = f"{base}-{ts}"
    return base


def main():
    # ===== Argument parser =====
    parser = argparse.ArgumentParser(description="Train mT5 on ThaiSum dataset")
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
        help="ตั้งชื่อโฟลเดอร์โมเดล (จะเซฟที่ ../model/<name>)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="ทับโฟลเดอร์โมเดลเดิมถ้ามีอยู่แล้ว (ระวังข้อมูลหาย)",
    )
    args = parser.parse_args()

    # ===== Resolve output_dir =====
    output_dir = _resolve_output_dir(args.name, args.overwrite)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output dir: {output_dir}")

    # ===== Load dataset + tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False, use_fast=False)
    dataset = load_thaisum()

    # แสดงจำนวนข้อมูลทั้งหมดใน train split
    total_train = len(dataset["train"])

    # ใช้เฉพาะบางส่วนของ train set ถ้าระบุ --size < 1.0
    if 0 < args.size < 1.0:
        subset_size = int(total_train * args.size)
        print(
            f"⚙️  Using {args.size*100:.0f}% of the training set → {subset_size}/{total_train} samples"
        )
        dataset["train"] = dataset["train"].train_test_split(
            test_size=(1 - args.size), seed=42
        )["train"]
    elif args.size == 1.0:
        print(f"✅ Using full training dataset → {total_train} samples")
    else:
        print("⚠️  --size should be in (0,1]; fallback to full dataset.")
        args.size = 1.0

    # ===== Preprocess =====
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)

    # ===== Load model =====
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # ===== Data collator =====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # ===== Detect hardware & precision =====
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

    # ===== Training arguments =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        load_best_model_at_end=False,
        dataloader_pin_memory=pin_memory,
        report_to="none",
        seed=42,
        dataloader_num_workers=num_workers,
        **extra_args,
    )

    # ===== Trainer =====
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ===== Train =====
    trainer.train()

    # ===== Save final model =====
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("✅ Training finished and model saved.")


if __name__ == "__main__":
    main()
