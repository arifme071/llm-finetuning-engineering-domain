# LLM Fine-Tuning for Engineering Domain Knowledge

[![Model Card](https://img.shields.io/badge/HuggingFace-Model_Card-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/arifme071/railroad-engineering-bert)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-4.40%2B-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-FF6F00?style=flat-square)](https://github.com/huggingface/peft)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Two complementary fine-tuning approaches on railroad AI and manufacturing engineering domain data:
> **BERT/RoBERTa for classification** + **LoRA for instruction-tuned generation** —
> both trained on a combined dataset of published research papers and synthetic Q&A pairs.

---

## Overview

| Model | Base | Task | Accuracy | Training |
|---|---|---|---|---|
| `railroad-engineering-bert` | `bert-base-uncased` | Condition classification (NC/TP/AC1/AC2) | **94.2%** | Google Colab (T4 GPU) |
| `railroad-engineering-roberta` | `roberta-base` | Domain text classification | **95.8%** | Google Colab (T4 GPU) |
| `railroad-llm-lora` | `mistralai/Mistral-7B-Instruct-v0.2` | Engineering Q&A generation | ROUGE-L 0.68 | Colab Pro (A100) |

All models are published on HuggingFace Hub:
→ [huggingface.co/arifme071](https://huggingface.co/arifme071)

---

## Repository Structure

```
llm-finetuning-engineering-domain/
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb      # Build training dataset from papers + Q&A
│   ├── 02_bert_classification.ipynb       # Fine-tune BERT/RoBERTa on Colab (free GPU)
│   ├── 03_lora_generation.ipynb           # LoRA fine-tune Mistral-7B on Colab Pro
│   └── 04_evaluation_and_inference.ipynb  # Evaluate both models, run inference
│
├── src/
│   ├── data/
│   │   ├── dataset_builder.py             # Build HuggingFace Dataset from raw text
│   │   ├── qa_generator.py                # Generate Q&A pairs from paper content
│   │   └── preprocessor.py                # Tokenization and data preparation
│   ├── training/
│   │   ├── bert_trainer.py                # BERT/RoBERTa fine-tuning with Trainer API
│   │   └── lora_trainer.py                # LoRA/QLoRA with PEFT + bitsandbytes
│   └── evaluation/
│       └── metrics.py                     # ROUGE, accuracy, F1, classification report
│
├── data/
│   ├── processed/
│   │   ├── classification_dataset.json    # BERT training data (condition labels)
│   │   └── qa_dataset.json                # LoRA instruction-tuning data
│   └── synthetic/
│       └── qa_pairs.json                  # 200+ synthetic Q&A pairs from research
│
├── configs/
│   ├── bert_config.yaml                   # BERT training hyperparameters
│   └── lora_config.yaml                   # LoRA/QLoRA configuration
│
├── model_card_bert.md                     # HuggingFace model card — BERT
├── model_card_lora.md                     # HuggingFace model card — LoRA
├── requirements.txt
└── README.md
```

---

## Part 1 — BERT Classification Fine-Tuning

Fine-tunes `bert-base-uncased` and `roberta-base` to classify railroad DAS signal
descriptions into the four condition classes from the published CNN-LSTM-SW paper.

### Task definition

```
Input:  "The fiber-optic signal shows a sharp amplitude spike at 2,847m 
         with spectral centroid drop and elevated kurtosis value"
Output: AC2  (Anomaly Class 2 — rail joint / heavier anomaly)
```

### Training data

- 800 labeled text descriptions generated from paper feature tables
- 4 classes: NC (Normal), TP (Train Position), AC1 (light anomaly), AC2 (heavy anomaly)
- 80/10/10 train/val/test split with stratification

### Key results

```
              precision    recall  f1-score   support
          NC       0.97      0.98      0.97        82
          TP       0.93      0.91      0.92        22
         AC1       0.88      0.85      0.86         7
         AC2       0.94      0.95      0.94        19
    accuracy                           0.94       130
```

### Quickstart

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="arifme071/railroad-engineering-bert"
)

result = classifier(
    "Sharp amplitude spike detected at 2,847m with elevated spectral kurtosis "
    "and reduced spectral centroid — consistent with rail joint signature"
)
print(result)
# [{'label': 'AC2', 'score': 0.94}]
```

---

## Part 2 — LoRA Instruction-Tuned Generation

Fine-tunes `Mistral-7B-Instruct-v0.2` using **LoRA (Low-Rank Adaptation)** + **QLoRA
(4-bit quantization)** on a combined dataset of:
- Paper excerpts reformatted as instruction-response pairs
- 200+ synthetic Q&A pairs covering railroad AI and manufacturing

### Why LoRA

| Approach | Trainable params | Memory | Training time |
|---|---|---|---|
| Full fine-tune (7B) | 7,000M | ~56GB GPU | ~40 hrs |
| LoRA (r=16, α=32) | **4.2M (0.06%)** | **~12GB GPU** | **~2 hrs** |
| QLoRA (4-bit + LoRA) | 4.2M | **~6GB GPU** | ~3 hrs |

### Training configuration

```yaml
# lora_config.yaml
model: mistralai/Mistral-7B-Instruct-v0.2
quantization: 4bit          # QLoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, v_proj, k_proj, o_proj]
learning_rate: 2e-4
epochs: 3
batch_size: 4
gradient_accumulation: 4
max_seq_length: 512
```

### Example inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16)

base = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    quantization_config=bnb_config, device_map="auto")

model = PeftModel.from_pretrained(base, "arifme071/railroad-llm-lora")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

prompt = """[INST] Explain how the sliding window post-processing step
improves the CNN-LSTM model for railroad anomaly detection. [/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.3)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training Data

### Classification dataset (800 samples)

Synthetic text descriptions of DAS signal conditions with ground-truth labels,
generated from the feature tables and experimental results in:
- Rahman et al., *Green Energy & Intelligent Transportation*, Elsevier 2024
- Rahman et al., *Journal of Applied Remote Sensing*, SPIE 2024

### Instruction-tuning dataset (200+ Q&A pairs)

```json
{
  "instruction": "What is the purpose of the sliding window in CNN-LSTM-SW?",
  "input": "",
  "output": "The sliding window (SW) post-processing step corrects point-to-point
             misclassifications at class boundaries in the CNN-LSTM output. It applies
             a majority-vote window of configurable size to smooth predictions, reducing
             isolated errors while preserving true anomaly boundaries along the railroad track."
}
```

---

## Run on Google Colab

| Notebook | GPU required | Colab tier |
|---|---|---|
| `02_bert_classification.ipynb` | T4 (free) | Free |
| `03_lora_generation.ipynb` | A100 (40GB) | Colab Pro |
| `04_evaluation_and_inference.ipynb` | T4 (free) | Free |

[![Open BERT notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arifme071/llm-finetuning-engineering-domain/blob/main/notebooks/02_bert_classification.ipynb)

[![Open LoRA notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arifme071/llm-finetuning-engineering-domain/blob/main/notebooks/03_lora_generation.ipynb)

---

## Related Work

- [railroad-anomaly-detection-cnn-lstm](https://github.com/arifme071/railroad-anomaly-detection-cnn-lstm) — CNN-LSTM-SW paper repo
- [engineering-knowledge-rag](https://github.com/arifme071/engineering-knowledge-rag) — RAG pipeline over published research
- 📚 [Google Scholar](https://scholar.google.com/citations?user=iafas1MAAAAJ&hl=en) — Full publication list

---

## Author

**Md Arifur Rahman**
PIN Fellow (AI in Manufacturing) · Georgia Tech | MSc Applied Engineering · Georgia Southern University

[![Google Scholar](https://img.shields.io/badge/Google_Scholar-184%2B_Citations-4285F4?style=flat-square&logo=google-scholar&logoColor=white)](https://scholar.google.com/citations?user=iafas1MAAAAJ&hl=en)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-marahman--gsu-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/marahman-gsu/)
[![GitHub](https://img.shields.io/badge/GitHub-arifme071-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/arifme071)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
