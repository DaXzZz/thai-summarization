# 🧠 Thai Summarization (mT5 + ThaiSum Dataset)

This project implements **Thai text summarization** using **mT5**, trained and evaluated on the **ThaiSum** dataset.  
It explores three training strategies — **Full Fine-tuning**, **Parameter-efficient LoRA**, and **Keyword-based Fine-tuning** —  
and evaluates their performance using **ROUGE** and **BERTScore**.

---

## 📂 Project Structure

```
thai-summarization/
│
├── data/                        # Evaluation results and logs
├── model/                       # Trained model checkpoints
├── src/                         # Source code
│   ├── preprocess.py            # Dataset loading & preprocessing
│   ├── train.py                 # Full fine-tuning & LoRA
│   ├── train_keyword.py         # Keyword-based fine-tuning
│   ├── evaluate_model.py        # Evaluation (ROUGE, BERTScore)
│   └── summarize.py             # Generate summaries
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

> ⚠️ Install **PyTorch after** installing requirements.txt

#### 🖥️ CUDA GPU
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### 🍏 Apple Silicon (M-series)
```bash
pip3 install torch torchvision
```

---

## 🚀 Training Methods

### 🧩 1. Full Fine-tuning

Fine-tune the entire **mT5** model on ThaiSum.

```bash
python src/train.py --epochs 2 --learning_rate 5e-5 --batch_size 8 --size 1.0
```

### ⚙️ 2. LoRA Fine-tuning (Parameter-efficient)

Fine-tune only small adapter layers (~0.4% of parameters).

```bash
python src/train.py --lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
    --epochs 2 --learning_rate 1e-3 --batch_size 8
```

### 🔑 3. Keyword-based Fine-tuning (Ours)

Train mT5 with **keyword-augmented input**.
The model receives both extracted keywords and article body:

```
Keywords: k1, k2, k3 | Article: <text>
```

```bash
python src/train_keyword.py --epochs 2 --learning_rate 5e-5 --keyword_mode_train overlap
```

---

## 📊 Evaluation

Evaluate any trained model or pre-trained mT5 using **ROUGE** and **BERTScore**.

### Example

```bash
# Fine-tuned
python src/evaluate_model.py --model ./model/FineTuned-100 --split test

# LoRA model
python src/evaluate_model.py --model ./model/LoRA-100 --split test

# Keyword-based (with keyword extraction during eval)
python src/evaluate_model.py --model ./model/Keyword-100 --split test --use_keywords
```

### Output

```
./data/{name}/
├── predictions_test.txt      # Model-generated summaries
├── references_test.txt       # Gold summaries
├── inputs_test.txt           # Input articles (with or w/o keywords)
└── score/
    ├── metrics.json          # Detailed metrics
    └── metrics_readable.txt  # Human-readable results
```

---

## 🧠 Evaluation Metrics

| Metric              | Description                                                                |
| :------------------ | :------------------------------------------------------------------------- |
| **ROUGE-1 / 2 / L** | Measures lexical overlap between generated and reference summaries         |
| **BERTScore (F1)**  | Measures semantic similarity between summaries using contextual embeddings |

---

## 🧪 Experimental Results

| Model                        |  ROUGE-1  |  ROUGE-2  |  ROUGE-L  | BERTScore (F1) | Train Time |
| :--------------------------- | :-------: | :-------: | :-------: | :------------: | :--------: |
| **Zero-shot mT5**            |    2.73   |    0.73   |    2.71   |      78.28     |      —     |
| **Fine-tuned mT5**           |   53.44   |   34.02   |   53.38   |      95.36     |   ~12 hrs  |
| **LoRA Fine-tuned mT5**      | **53.94** | **34.60** | **53.86** |    **95.47**   |   ~7 hrs   |
| **Keyword-based mT5 (Ours)** |   42.60   |   25.33   |   42.45   |      93.29     |   ~11 hrs  |

🟩 **LoRA achieved comparable performance to full fine-tuning** while training <1% parameters.
🟦 **Keyword-based model** produced summaries that are semantically correct (high BERTScore)
but stylistically different (lower ROUGE).

---

## 💬 Custom Summarization

```bash
python src/summarize.py --model ./model/FineTuned-100 \
  --text "กรุงเทพมหานคร - วันนี้กรมอุตุนิยมวิทยาพยากรณ์อากาศว่าจะมีฝนฟ้าคะนอง..."
```

---

## 📖 References

* **Dataset:** [ThaiSum (PyThaiNLP)](https://huggingface.co/datasets/pythainlp/thaisum)
* **Base Model:** [google/mt5-small](https://huggingface.co/google/mt5-small)
* **LoRA Technique:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
* **Keyword-based Approach:** Adapted from *Automatic Thai Text Summarization Using Keyword-Based Abstractive Method (2022)*
* **Evaluation:** [Hugging Face Evaluate – ROUGE, BERTScore](https://huggingface.co/docs/evaluate)

---

<div align="center">

**Author:** Nontapat Chucharnchai
**Advisor:** —
**Environment:** Python 3.10+, PyTorch, Hugging Face Transformers
**License:** Research / Academic Use Only

</div>
