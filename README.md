# 🧠 Thai Summarization — mT5 + ThaiSum Dataset

This project implements **Thai text summarization** using the **mT5 model** fine-tuned on the **ThaiSum dataset**.  
It supports **three training modes** — full fine-tuning, LoRA fine-tuning (via flag), and keyword-based fine-tuning — and includes evaluation using **ROUGE** and **BERTScore** metrics.

> ⚠️ **Important:** All command examples below use relative paths (e.g., `src/train.py`). Please adjust the paths according to your actual project location before running any commands.

---

## 📂 Project Structure

```
thai-summarization/
│
├── data/                        # Evaluation results and logs
├── model/                       # Saved models
├── notebooks/                   # Optional analysis notebooks
├── src/                         # Source code
│   ├── train.py                 # Training script (supports --lora flag)
│   ├── train_keyword.py         # Keyword-based fine-tuning script
│   ├── evaluate_model.py        # Model evaluation (ROUGE + BERTScore)
│   ├── summarize.py             # Custom text summarization
│   └── preprocess.py            # Dataset loading and preprocessing
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
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

> ⚠️ Install PyTorch **after** installing requirements.txt

#### 🖥️ Windows / Linux (NVIDIA GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### 🍏 macOS (Apple Silicon)
```bash
pip install torch torchvision
```

---

## 🚀 Training Modes

### 🧩 1. Full Fine-tuning

Fine-tune all model parameters on ThaiSum.

```bash
python src/train.py --size 1.0 --epochs 2 --batch_size 8 --learning_rate 5e-5 --name FineTune-100
```

---

### 🧩 2. LoRA Fine-tuning (Parameter-efficient)

Enable LoRA mode with the `--lora` flag (only ~0.4% parameters are trainable).  
Recommended for limited VRAM setups.

```bash
python src/train.py --lora --size 1.0 --epochs 2 --batch_size 8 --learning_rate 1e-3 --name LoRA-100
```

---

### 🧩 3. Keyword-based Fine-tuning

Train with keyword-enhanced input for better focus on key concepts.

```bash
python src/train_keyword.py --size 1.0 --epochs 2 --batch_size 8 --learning_rate 5e-5 --topk 10 --minlen 2 --name Keyword-100
```

---

## ⚙️ Training Arguments (All Modes)

| Argument | Description | Typical Range / Example |
|----------|-------------|-------------------------|
| `--size` | Fraction of training data used (0–1) | `--size 1.0` |
| `--epochs` | Number of training epochs | `1–3` |
| `--batch_size` | Samples per training step | `4–16` |
| `--learning_rate` | Optimizer learning rate | `5e-5` (full), `1e-3` (LoRA) |
| `--warmup_ratio` | Ratio of steps for warmup | `0.03` |
| `--lr_scheduler_type` | Scheduler type | `linear` |
| `--gradient_accumulation_steps` | Accumulate grads before update | `1–4` |
| `--name` | Save folder name for model | `FineTune-100`, `LoRA-100`, etc. |

### Extra for LoRA mode (`--lora`):

| Argument | Description | Default |
|----------|-------------|---------|
| `--lora_r` | Rank of LoRA matrices | `8` |
| `--lora_alpha` | Scaling factor | `16` |
| `--lora_dropout` | Dropout for LoRA layers | `0.05` |

### Extra for Keyword-based training:

| Argument | Description | Default |
|----------|-------------|---------|
| `--topk` | Max extracted keywords | `10` |
| `--minlen` | Minimum keyword length | `2` |
| `--mode_train` | Keyword extraction rule for train | `overlap` |
| `--mode_val` | Keyword extraction rule for validation | `unsupervised` |

---

## 🧮 Evaluation

Evaluate any trained or pre-trained model using ROUGE + BERTScore.

```bash
python src/evaluate_model.py --model ./model/FineTune-100 --split test
```

### Optional Flags

| Argument | Description | Example |
|----------|-------------|---------|
| `--split` | Dataset split (`validation` / `test`) | `--split test` |
| `--max_samples` | Limit samples for quick test | `--max_samples 500` |
| `--batch_size` | Evaluation batch size | `--batch_size 8` |
| `--name` | Output folder under `./data/` | `--name EvalRun` |
| `--overwrite_output_dir` | Overwrite existing output | Optional |
| `--use_keywords` | (Keyword model only) rebuild keyword-based input during eval | `--use_keywords` |

### 📁 Output files

```
./data/{name}/
├── predictions_test.txt
├── references_test.txt
├── inputs_test.txt
└── score/
    ├── metrics.json
    └── metrics_readable.txt
```

---

## 💬 Summarize Custom Text

Generate summaries manually using any trained model.

```bash
python src/summarize.py --model ./model/FineTune-100 --text "ข้อความที่ต้องการสรุป..."
```

### Optional (Keyword-based model)

```bash
python src/summarize.py --model ./model/Keyword-100 --text "Keywords: ฝุ่น, PM2.5, กรุงเทพมหานคร | Article: <เนื้อหา>"
```

### Notes
- Works even if no keywords are given (model still summarizes).
- Keep input format consistent with training format.
- You can modify generation settings inside the code.

---

## ⚙️ Generation Config (Inside summarize.py & evaluate_model.py)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_beams` | Number of beams for beam search | `4` |
| `max_new_tokens` | Max generated tokens | `128` |
| `length_penalty` | Adjust summary length (<1: shorter, >1: longer) | `0.8` |
| `pad_token_id` | Padding token ID | auto |
| `eos_token_id` | End-of-sequence token ID | auto |

---

## 💡 Practical Tips

### 🎯 Choosing Batch Size
- **High VRAM (16GB+):** Use `--batch_size 16` for faster training
- **Medium VRAM (8-12GB):** Use `--batch_size 8` (default)
- **Low VRAM (<8GB):** Use `--batch_size 4` or enable LoRA mode

### ⚡ LoRA vs Full Fine-tuning
- **LoRA:** Faster (~7 hrs), uses less memory, achieves similar performance
- **Full Fine-tuning:** Slightly better results (~12 hrs), requires more VRAM
- **Recommendation:** Start with LoRA for experiments, use full fine-tuning for production

### 📊 Learning Rate Guidelines
- **Full Fine-tuning:** `5e-5` (default, stable)
- **LoRA:** `1e-3` (higher because fewer parameters are trained)
- **Keyword-based:** `5e-5` (same as full fine-tuning)

### 🔧 Generation Quality
- **Shorter summaries:** Set `length_penalty=0.6` in code
- **Longer summaries:** Set `length_penalty=1.2` in code
- **More diverse outputs:** Increase `num_beams=8`

### 🚀 Quick Testing Workflow
1. Train with `--size 0.2` (20% data) for fast iteration
2. Evaluate with `--max_samples 100` to check quickly
3. Once satisfied, train with `--size 1.0` (full dataset)

---

## 📖 References

- **Dataset:** [ThaiSum (PyThaiNLP)](https://huggingface.co/datasets/pythainlp/thaisum)
- **Base Model:** [google/mt5-small](https://huggingface.co/google/mt5-small)
- **Metrics:** [Hugging Face Evaluate (ROUGE, BERTScore)](https://huggingface.co/docs/evaluate)
- **LoRA Reference:** Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
- **Keyword-based Approach:** Adapted from *Automatic Thai Text Summarization Using Keyword-Based Abstractive Method* (2022)

---

<div align="center">

**Author:** Nontapat Chucharnchai  
**Environment:** Python 3.10+, PyTorch, Hugging Face Transformers  

</div>