# Taxonomy-Driven AI-Based Detection of Prompt Injection Attacks
## Submission Package — CSC-601 Artificial Intelligence (Lab)

**Authors:** Abdul Samad & Maqbool Hussain
**Department:** Computer Science, Institute of Management Sciences, Peshawar
**Instructor:** Mr. Ali Haider

---

## Contents

| File / Folder | Description |
|---|---|
| `final_paper.docx` | Complete research paper (IEEE format, all sections + figures) |
| `prompt_injection_dataset.csv` | Labeled dataset — 586 samples, 6 classes |
| `train_model.py` | Python model training script |
| `figures/` | All result figures (PNG) |
| `results.csv` | Generated automatically when you run the model |

---

## How to Run the Model

### Step 1 — Install requirements
```
pip install scikit-learn pandas numpy matplotlib seaborn openpyxl
```

### Step 2 — Place files in the same folder
Make sure these are together:
```
train_model.py
prompt_injection_dataset.csv
```

### Step 3 — Run
```
python train_model.py
```

### Output
- Prints accuracy, F1, precision, recall for all 3 models
- Saves all figures to `figures/` folder
- Saves `results.csv` summary table

---

## Dataset Classes

| Label | Class | Samples |
|---|---|---|
| 0 | Benign | 101 |
| 1 | Instruction Override | 101 |
| 2 | Role Manipulation | 99 |
| 3 | Data Exfiltration | 100 |
| 4 | Obfuscation / Encoding | 86 |
| 5 | Social Engineering | 99 |
| | **Total** | **586** |

---

## Model Results Summary

| Model | Accuracy | F1-Score |
|---|---|---|
| Logistic Regression | 94.07% | 0.942 |
| **SVM (RBF)** | **97.46%** | **0.975** |
| Random Forest | 83.90% | 0.831 |

Best model: **SVM with RBF kernel**
