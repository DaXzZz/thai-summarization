from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

MODEL_NAME = "google/mt5-small"  # เปลี่ยนเป็น mt5-base ได้ถ้า resource ไหว

# แก้ warning legacy tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)


def load_thaisum():
    # โหลด ThaiSum dataset สำหรับสรุปข้อความไทย
    dataset = load_dataset("pythainlp/thaisum")
    if "valid" in dataset:  # ลบ split ที่ซ้ำ (เหลือแค่ train/validation/test)
        dataset.pop("valid")
    return dataset


def sanity_check(dataset):
    # แสดงจำนวนข้อมูลในแต่ละ split
    for split in dataset.keys():
        print(f"{split}: {len(dataset[split])} rows")
    # ตัวอย่าง 1 แถว
    print(dataset["train"][0])


def preprocess_dataset(dataset, tokenizer, max_input=512, max_target=128):
    def tokenize_fn(batch):
        # แปลง "body" (ข้อความยาว) → input tokens
        model_inputs = tokenizer(batch["body"], max_length=max_input, truncation=True)
        # แปลง "summary" (สรุป) → target labels
        labels = tokenizer(batch["summary"], max_length=max_target, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # ประมวลผลข้อมูลและลบ columns ที่ไม่ต้องการ
    return dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["title", "body", "summary", "tags", "url", "type"],
    )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_thaisum()
    sanity_check(dataset)
    tokenized_dataset = preprocess_dataset(dataset, tokenizer)
    print(tokenized_dataset)
