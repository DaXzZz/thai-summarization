# 🧠 Thai Summarization (mT5 + ThaiSum Dataset)

This project implements **Thai text summarization** using **mT5** fine-tuned on the **ThaiSum** dataset. It supports both **fine-tuned** and **zero-shot** evaluation with **ROUGE** and **BERTScore** metrics.

---

## 📁 Project Structure

```
thai-summarization/
│
├── data/                        # Evaluation results and logs
├── model/                       # Trained models
├── notebooks/                   # Jupyter notebooks for analysis
├── src/                         # Core source code
│   ├── evaluate_model.py        # Evaluation (ROUGE, BERTScore)
│   ├── preprocess.py            # Dataset loading and preprocessing
│   ├── summarize.py             # Summarization inference script
│   └── train.py                 # Model fine-tuning
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
> ⚠️ **Important:** Install PyTorch **after** installing requirements.txt

#### 🖥️ Windows / Linux (NVIDIA GPU)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### 🍏 macOS (Apple M-series)
```bash
pip3 install torch torchvision
```

---

## 🚀 Training (Fine-tuning mT5)

Fine-tune the `mT5` model on ThaiSum dataset. Supports partial dataset usage via `--size` (number of samples).

```bash
# Train with 1000 samples
python src/train.py --size 1000

# Train with full dataset (no size limit)
python src/train.py
```

📍 **Output:** The trained model will be saved to `./model/FineTuned-{steps}/`

---

## 📊 Evaluation (ROUGE + BERTScore)

Evaluate either a fine-tuned model or a zero-shot pre-trained mT5 model.

### ✅ Fine-tuned Model
```bash
# Basic evaluation
python src/evaluate_model.py --model ./model/FineTuned-100 --split test

# With sample limit for faster testing
python src/evaluate_model.py --model ./model/FineTuned-100 --split test --max_samples 500

# Custom output folder name
python src/evaluate_model.py --model ./model/FineTuned-100 --split test --name MyEvalRun
```

### 🌏 Zero-shot mT5
```bash
# Basic zero-shot evaluation
python src/evaluate_model.py --model google/mt5-small --split test

# With sample limit
python src/evaluate_model.py --model google/mt5-small --split test --max_samples 500
```

### 📁 Output Files
```
./data/{name}/                   # Default: Results or Results_{timestamp}
├── predictions_test.txt         # Model-generated summaries
├── references_test.txt          # Ground truth summaries
├── inputs_test.txt              # Original input texts
└── score/
    ├── metrics.json             # Detailed metrics (JSON)
    └── metrics_readable.txt     # Human-readable scores
```

---

## 💬 Summarize Custom Text

Generate summaries for custom Thai text input.

```bash
# Basic usage
python src/summarize.py --model ./model/FineTuned-100 --text "ข้อความที่ต้องการสรุป..."

# Zero-shot with pretrained mT5
python src/summarize.py --model google/mt5-small --text "ข้อความที่ต้องการสรุป..."

# Long text example
python src/summarize.py --model ./model/FineTuned-100 --text "กรุงเทพมหานคร - วันนี้กรมอุตุนิยมวิทยาพยากรณ์อากาศว่าจะมีฝนฟ้าคะนองในพื้นที่กรุงเทพและปริมณฑล..."
```

---

## ⚙️ Command Arguments

### Training (`train.py`)
| Argument | Description | Default |
|----------|-------------|---------|
| `--fraction` | Fraction of training data to use (0-1) | `1.0` |

### Evaluation (`evaluate_model.py`)
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model path or HF ID | **(required)** |
| `--split` | Dataset split (`validation` / `test`) | `test` |
| `--max_samples` | Limit samples for faster eval | `None` (full) |
| `--batch_size` | Evaluation batch size | `8` |
| `--name` | Output folder name under `./data/` | `Results` |
| `--overwrite_output_dir` | Overwrite existing output folder | `False` |

### Summarize (`summarize.py`)
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model path or HF ID | **(required)** |
| `--text` | Input text to summarize | **(required)** |

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