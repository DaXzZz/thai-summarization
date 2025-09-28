import torch
import sys, os, argparse
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="transformers.convert_slow_tokenizer"
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))

from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
)
from preprocess import load_thaisum, preprocess_dataset, MODEL_NAME


def main():
    # ===== Argument parser =====
    parser = argparse.ArgumentParser(description="Train mT5 on ThaiSum dataset")
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="ระบุสัดส่วนของ train set ที่ต้องการใช้ เช่น 0.4 = 40%% (default = 1.0 ใช้ทั้งหมด)",
    )
    args = parser.parse_args()

    # ===== Load dataset + tokenizer =====
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False, use_fast=False)
    dataset = load_thaisum()

    if args.fraction < 1.0:
        ratio = round(args.fraction * 100)
        print(f"⚙️  Using {ratio}% of the training set (random subset)...")
        dataset["train"] = dataset["train"].train_test_split(
            test_size=(1 - args.fraction), seed=42
        )["train"]
    else:
        print("✅ Using full training dataset")

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

    # MPS doesn't support multiprocessing tensor sharing, so use 0 workers
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
        output_dir="../model/FineTuned-mT5-ThaiSum",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-5,  # 0.00005
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

    # ===== Train (รวดเดียว) =====
    trainer.train()

    # ===== Save final model =====
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
