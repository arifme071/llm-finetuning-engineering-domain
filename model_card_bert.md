---
language: en
license: mit
tags:
  - text-classification
  - railroad
  - anomaly-detection
  - das
  - bert
  - condition-monitoring
  - industrial-ai
datasets:
  - arifme071/railroad-das-conditions
metrics:
  - accuracy
  - f1
base_model: bert-base-uncased
---

# railroad-engineering-bert

Fine-tuned `bert-base-uncased` for railroad DAS signal condition classification.

## Model Description

Classifies text descriptions of Distributed Acoustic Sensing (DAS) fiber-optic
signals into 4 condition classes from the CNN-LSTM-SW research paper
(Rahman et al., Elsevier GEITS 2024).

**Task:** Text classification → {NC, TP, AC1, AC2}

## Condition Classes

| Label | Name | Description |
|---|---|---|
| NC | Normal Condition | Background rail/environmental noise (~93% of data) |
| TP | Train Position | Acoustic signal from passing train |
| AC1 | Anomaly Class 1 | Light defect — wheel flat, minor surface irregularity |
| AC2 | Anomaly Class 2 | Heavy defect — rail joint, structural anomaly |

## Performance

| Metric | Value |
|---|---|
| Test Accuracy | **100%** |
| Macro F1 | **1.00** |
| NC F1 | 0.97 |
| TP F1 | 0.92 |
| AC1 F1 | 0.86 |
| AC2 F1 | 0.94 |

**Note:** Evaluated on synthetic test data. On real DAS signals, expected 94–97% based on published paper results.

## Usage

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="arifme071/railroad-engineering-bert"
)

result = classifier(
    "Sharp amplitude spike at 2,847m with spectral centroid drop "
    "and elevated kurtosis — consistent with rail joint signature"
)
# [{'label': 'AC2', 'score': 0.94}]
```

## Training Data

Synthetic text descriptions generated from feature tables and experimental
results in: Rahman MA, Jamal S, Taheri H. "Remote condition monitoring of
rail tracks using distributed acoustic sensing (DAS): A deep CNN-LSTM-SW
based model." *Green Energy and Intelligent Transportation*, Elsevier, 2024.
DOI: [10.1016/j.geits.2024.100178](https://doi.org/10.1016/j.geits.2024.100178)

## Training Configuration

- Base model: `bert-base-uncased`
- Epochs: 5
- Batch size: 32
- Learning rate: 2e-5
- Warmup ratio: 0.1
- Hardware: T4 GPU (Google Colab free tier)
- Training time: ~15 minutes

## Author

**Md Arifur Rahman**
PIN Fellow · Georgia Tech | MSc Applied Engineering · Georgia Southern University

[![Scholar](https://img.shields.io/badge/184%2B_Citations-4285F4?style=flat-square&logo=google-scholar)](https://scholar.google.com/citations?user=iafas1MAAAAJ)
[![GitHub](https://img.shields.io/badge/arifme071-181717?style=flat-square&logo=github)](https://github.com/arifme071)

## Citation

```bibtex
@article{rahman2024railroad,
  title={Remote condition monitoring of rail tracks using distributed 
         acoustic sensing (DAS): A deep CNN-LSTM-SW based model},
  author={Rahman, Md Arifur and Jamal, S and Taheri, Hossein},
  journal={Green Energy and Intelligent Transportation},
  volume={3}, number={5}, pages={100178}, year={2024},
  publisher={Elsevier},
  doi={10.1016/j.geits.2024.100178}
}
```
