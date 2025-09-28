import os, sys, argparse
import torch
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

# ===== Import preprocess.py เพื่อใช้ฟังก์ชันโหลดและ preprocess dataset =====
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "src"))
from preprocess import load_thaisum, preprocess_dataset, MODEL_NAME  # noqa: E402


# ===== Helper: แปลง token ids กลับเป็นข้อความ =====
def decode_text(tokenizer, ids):
    return tokenizer.batch_decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )


def main():
    # ===== Argument Parser =====
    # รับพารามิเตอร์จาก command line เช่น --model, --split, --num_beams ฯลฯ
    parser = argparse.ArgumentParser(
        description="Evaluate (any) Seq2Seq model on ThaiSum with ROUGE & BERTScore"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="พาธหรือชื่อโมเดล เช่น /Model/FineTuned-mT5-ThaiSum หรือ google/mt5-small",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="เลือกชุดข้อมูลที่ใช้ประเมิน (default: test)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="จำกัดจำนวนข้อมูลเพื่อประเมินเร็วขึ้น"
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="จำนวน beam ตอน generate"
    )
    parser.add_argument(
        "--gen_max_len", type=int, default=128, help="ความยาวสูงสุดของสรุปที่ generate"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size ตอน evaluate"
    )
    parser.add_argument(
        "--bertscore_model",
        type=str,
        default="xlm-roberta-large",
        help="โมเดลภายนอกที่ใช้คำนวณ BERTScore",
    )
    parser.add_argument(
        "--input_prefix",
        type=str,
        default="",
        help="เติม prefix ก่อนข้อความ input (เช่น 'summarize: ' สำหรับ T5/mT5 zero-shot)",
    )
    args = parser.parse_args()

    # ===== ตรวจสอบอุปกรณ์ (Device) =====
    use_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
    print(f"🔧 Device: {device}")
    if args.input_prefix:
        print(f"🧩 Input prefix: {repr(args.input_prefix)}")

    # ===== โหลดโมเดลและโทเคไนเซอร์ =====
    # โหลดทั้ง tokenizer และ model จากชื่อหรือ path ที่ระบุใน --model
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)

    # ===== โหลดชุดข้อมูล ThaiSum =====
    raw = load_thaisum()
    assert args.split in raw, f"Split '{args.split}' not found in dataset"
    ds = raw[args.split]

    # จำกัดจำนวนข้อมูลถ้าระบุ --max_samples
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # ===== เติม prefix (ใช้ใน zero-shot เช่น summarize:) =====
    if args.input_prefix:

        def add_prefix(batch):
            batch["body"] = [args.input_prefix + x for x in batch["body"]]
            return batch

        ds = ds.map(add_prefix, batched=True)

    # ===== Tokenize dataset (เหมือนตอน train) =====
    tokenized_dd = preprocess_dataset(DatasetDict({args.split: ds}), tokenizer)
    tokenized = tokenized_dd[args.split]

    # ===== เตรียม Trainer สำหรับ evaluate =====
    # ใช้ Seq2SeqTrainer เพื่อ generate summary อัตโนมัติ
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    eval_args = Seq2SeqTrainingArguments(
        output_dir="../data/eval_tmp",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,  # ให้ generate ข้อความแทนการ predict logits
        generation_max_length=args.gen_max_len,
        generation_num_beams=args.num_beams,
        dataloader_pin_memory=(False if use_mps else True),
        report_to="none",  # ปิดการ log ไปยัง wandb / tensorboard
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        eval_dataset=tokenized,
        processing_class=tokenizer,
    )

    # ===== ให้โมเดล generate สรุป =====
    print(
        f"🚀 Generating summaries (beams={args.num_beams}, max_len={args.gen_max_len}) ..."
    )
    preds = trainer.predict(tokenized)

    # แปลง token id → text (ทั้ง pred และ reference)
    pred_ids = preds.predictions
    label_ids = preds.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_texts = decode_text(tokenizer, pred_ids)
    ref_texts = decode_text(tokenizer, label_ids)

    # ===== คำนวณคะแนน ROUGE =====
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(
        predictions=pred_texts, references=ref_texts, use_stemmer=False
    )

    # ===== คำนวณ BERTScore =====
    print(f"📏 Using BERTScore model: {args.bertscore_model}")
    bertscore = evaluate.load("bertscore")
    bs_res = bertscore.compute(
        predictions=pred_texts,
        references=ref_texts,
        model_type=args.bertscore_model,
        device=device,
        batch_size=args.batch_size,
    )
    bs_p = float(np.mean(bs_res["precision"]))
    bs_r = float(np.mean(bs_res["recall"]))
    bs_f1 = float(np.mean(bs_res["f1"]))

    # ===== แสดงผลลัพธ์ =====
    def pct(x):
        return round(100 * x, 2)

    print("\n🎯 Results")
    print(f"- ROUGE-1 (F1): {pct(rouge_res['rouge1'])}")
    print(f"- ROUGE-2 (F1): {pct(rouge_res['rouge2'])}")
    print(f"- ROUGE-L (F1): {pct(rouge_res['rougeL'])}")
    print(f"- BERTScore P/R/F1: {pct(bs_p)} / {pct(bs_r)} / {pct(bs_f1)}")

    # ===== บันทึกผลลัพธ์ลงไฟล์ =====
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
    print(f"\n📝 Saved predictions to: {out_dir}/pred_{split}.txt")
    print(f"📝 Saved references  to: {out_dir}/ref_{split}.txt")


if __name__ == "__main__":
    main()
