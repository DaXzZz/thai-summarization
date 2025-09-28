# 🧠 Thai Summarization (mT5 + ThaiSum Dataset)

This project implements **Thai text summarization** using **mT5** fine-tuned on the **ThaiSum** dataset. It supports both **fine-tuned** and **zero-shot** evaluation with **ROUGE** and **BERTScore** metrics.

---

## 📁 Project Structure

```
thai-summarization/
│
├── data/                        # Temporary results and evaluation logs
├── Model/                       # Trained models
│   └── FineTuned-mT5-ThaiSum/   # Fine-tuned model
│       └── eval_outputs/        # Evaluation outputs (pred/ref)
├── notebooks/                   # Jupyter notebooks for analysis or visualization
├── src/                         # Core source code
│   ├── preprocess.py            # Dataset loading and preprocessing
│   ├── train.py                 # Model fine-tuning
│   ├── evaluate.py              # Evaluation (ROUGE, BERTScore)
│   └── summarize.py             # Summarization inference script
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Environment Setup

### 🪟 Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 🍎 macOS / 🐧 Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 🔥 Install PyTorch

#### 🖥️ Windows / Linux (NVIDIA GPU)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### 🍏 macOS (Apple M-series)
```bash
pip3 install torch torchvision
```

---

## 🧩 Data Preprocessing

Prepare the ThaiSum dataset for mT5 training and evaluation.

### Steps:
1. **Load the ThaiSum dataset**
   - Keep only `train`, `validation`, `test` splits
   - Remove duplicate `valid` split if present

2. **Sanity check**
   - Print dataset size for each split
   - Show one example row

3. **Tokenization**
   - Convert text into token IDs using `mT5 tokenizer`
   - `body` → `input_ids`, `attention_mask`
   - `summary` → `labels`

4. **Truncation**
   - Input (body) ≤ 512 tokens
   - Target (summary) ≤ 128 tokens

5. **Remove unnecessary columns**
   - Drop `title`, `body`, `summary`, `tags`, `url`, `type`
   - Keep only `input_ids`, `attention_mask`, `labels`

### 📘 Run preprocessing:
```bash
python src/preprocess.py
```

---

## 🚀 Training (Fine-tuning mT5)

Fine-tune the `mT5` model on ThaiSum dataset. Supports partial dataset usage via `--fraction` (0–1).

```bash
# Example: train with 30% of the training set
python -u "src/train.py" --fraction 0.3
```

📍 **Output:** The trained model will be saved to `Model/FineTuned-mT5-ThaiSum/`

---

## 📊 Evaluation (ROUGE + BERTScore)

Evaluate either a fine-tuned model or a zero-shot pre-trained mT5 model.

### ✅ Fine-tuned Model
```bash
python -u src/evaluate.py \
  --model /Model/FineTuned-mT5-ThaiSum \
  --split test
```

### 🌏 Zero-shot mT5
```bash
python -u src/evaluate.py \
  --model google/mt5-small \
  --input_prefix "summarize: " \
  --split test
```

### 📁 Output Files
```
Model/FineTuned-mT5-ThaiSum/eval_outputs/
├── pred_test.txt   # Model-generated summaries
└── ref_test.txt    # Reference summaries (ground truth)
```

---

## 🔄 Evaluation Process

The evaluation pipeline uses **ROUGE-1/2/L (F1)** and **BERTScore** metrics to assess model quality.

### Process Steps:
1. **Load model and tokenizer** (path or Hugging Face ID via `--model`)
2. **Load ThaiSum dataset split** (`validation` / `test`)
3. **Add input prefix** (optional, e.g., `"summarize: "` for zero-shot)
4. **Preprocess and tokenize dataset**
5. **Run generation** using beam search
6. **Decode predictions** → `pred_texts` & `ref_texts`
7. **Compute metrics** (ROUGE & BERTScore)
8. **Report and save results**

---

## ⚙️ Evaluation Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--model` | Model path or Hugging Face ID | `--model /Model/FineTuned-mT5-ThaiSum`<br>`--model google/mt5-small` |
| `--split` | Dataset split (`validation` or `test`) | `--split test` |
| `--max_samples` | Limit samples for faster evaluation | `--max_samples 500` |
| `--num_beams` | Beam search size | `--num_beams 4` |
| `--gen_max_len` | Maximum generation length (tokens) | `--gen_max_len 128` |
| `--batch_size` | Evaluation batch size | `--batch_size 8` |
| `--bertscore_model` | Model for BERTScore computation | `--bertscore_model xlm-roberta-large` |
| `--input_prefix` | Prefix for zero-shot summarization | `--input_prefix "summarize: "` |

---

## 🧹 Additional Commands

```bash
# Clear pip cache
pip cache purge

# Run preprocessing (dataset check)
python src/preprocess.py
```

---

## 💡 Performance Tips

- **Training:** Use `--fraction` to fine-tune on a percentage of the dataset
- **Inference:** Larger `--num_beams` → higher quality but slower inference
- **BERTScore:** `xlm-roberta-large` performs best for Thai text
- **Hardware:** On macOS M-series, `mps` backend runs much faster than CPU
- **Storage:** Temporary logs in `data/eval_tmp/` can be safely deleted

---

## 📖 References & Credits

- **Dataset:** [ThaiSum (PyThaiNLP)](https://huggingface.co/datasets/pythainlp/thaisum)
- **Base Model:** [google/mt5-small](https://huggingface.co/google/mt5-small)
- **Metrics:** [Hugging Face Evaluate (ROUGE, BERTScore)](https://huggingface.co/docs/evaluate)

---

<div align="center">

**Author:** Nontapat Chucharnchai  
**Environment:** Python 3.10+, PyTorch, Hugging Face Transformers  
**License:** Research / Academic use only

</div>