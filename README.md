#  Epileptic Seizure Prediction — Preprocessing, Regularization & Generalisation Study

> **Machine Learning — Semester Major Assignment**
> **Author:** Bushra Bakht
> **Course:** Data Mining 


---

## 📌 Overview

This project systematically investigates how **preprocessing pipeline ordering**, **model complexity**, **regularization strategies (L1, L2, Elastic Net)**, and **class-imbalance handling** affect the generalisation performance of logistic regression classifiers for EEG-based epileptic seizure prediction.

Three synthetic EEG-representative datasets are evaluated across **2 preprocessing pipelines**, **3 regularization methods**, **6 class-balancing strategies**, and a full **bias-variance (overfitting/underfitting) demonstration** — covering **50+ controlled experiments** in total.

---

## 🎯 Research Questions

| # | Question |
|---|----------|
| **Q1** | Does the *ordering* of preprocessing operations measurably affect generalisation performance? |
| **Q2** | Which regularization (L1, L2, Elastic Net) generalises best across heterogeneous EEG datasets? |
| **Q3** | Does Elastic Net consistently outperform L1 and L2 individually? |
| **Q4** | How does class-imbalance handling interact with regularization choice? |

---

## 🏆 Key Findings

| Finding | Evidence |
|---------|----------|
| **Preprocessing order matters** | Correct pipeline ordering improves F1 by **1.3–4.8%** across all datasets |
| **Elastic Net generalises best** | Most consistent F1 with lowest cross-dataset variance (σ=0.089) |
| **EN advantage grows with imbalance** | +1.31% over L1 on DS2, +0.87% on DS3 — marginal on DS1 |
| **Class balancing is critical** | F1 improves dramatically on imbalanced data with SMOTE + class weighting |
| **Accuracy is misleading** | DS3 shows 91.8% accuracy yet only 58.2% F1 — **PR-AUC** is the right metric |

### 🥇 Best Configuration
**Pipeline A (correct order) + Elastic Net + SMOTE + Class Weighting**
→ Best **PR-AUC: 0.7012**, Best **ROC-AUC: 0.8512** on severely imbalanced DS3

---

## 📊 Datasets

| ID | Dataset | Samples | Features | Seizure Rate | Feature Type |
|----|---------|---------|----------|--------------|-------------|
| DS1 | UCI Epileptic Seizure (Andrzejak) | 11,500 | 30 | **20%** | Extracted spectral & statistical |
| DS2 | CHB-MIT Scalp EEG (Shoeb) | 8,000 | 25 | **8%** | Hjorth parameters + time-series stats |
| DS3 | Kaggle Melbourne iEEG | 6,000 | 20 | **5%** | Frequency-domain (intracranial EEG) |

> **Justification:** The three datasets span moderate → severe → extreme class imbalance (20% → 8% → 5%) and cover all major EEG feature modalities, enabling robust evaluation of preprocessing and regularization strategies across clinically realistic conditions.

---

## 🔬 Methodology

### Preprocessing Pipelines

**Pipeline A — Signal-First:**
```
Z-score Normalize  →  Noise Removal (±3σ clip)  →  ANOVA F-test Feature Selection (k=15)
```

**Pipeline B — Feature-First:**
```
Statistical/FFT Feature Extraction  →  RobustScaler  →  PCA (95% variance)
```

> **Research Insight:** Ordering is a critical hyperparameter. Applying σ-clipping *before* normalization uses arbitrary thresholds and degrades F1 by 1.3–4.8%. Feature selection on raw (unscaled) data introduces scale bias.

---

### Model

**Logistic Regression** (scikit-learn) with three regularization variants:

| Regularizer | Penalty | Effect |
|-------------|---------|--------|
| **L1 (Lasso)** | `λΣ\|W\|` | Sparsity — drives irrelevant coefficients to zero |
| **L2 (Ridge)** | `(λ/2)Σ\|\|W\|\|²` | Shrinkage — proportionally reduces all coefficients |
| **Elastic Net** | `λ[α\|W\| + (1-α)\|\|W\|\|²]` | Combined — sparsity + stability (α=0.5) |

Decision function:

$$P(y=1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

Regularized cost function:

$$J(W,b) = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum \|W\|^2$$

---

### Class-Imbalance Strategies

- No balancing (baseline)
- Class weighting (`class_weight='balanced'`)
- SMOTE oversampling
- Random oversampling
- Random undersampling
- SMOTE + Class weighting (combined)

---

### Evaluation Metrics

| Metric | Why Used |
|--------|----------|
| **PR-AUC** | Primary metric — sensitive to minority class, not inflated by majority |
| **F1-Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Overall discrimination ability |
| Accuracy | Reported but *not* relied upon for imbalanced data |
| Precision / Recall | Tradeoff analysis for clinical deployment decisions |

**Evaluation protocol:** Stratified 80/20 train-test split (seed=42), 5-fold cross-validation for learning curves.

---

## 📈 Results Summary

### Baseline Logistic Regression (Pipeline A, C=1.0, balanced weights)

| Dataset | Accuracy | F1-Score | PR-AUC | ROC-AUC |
|---------|----------|----------|--------|---------|
| DS1 (UCI-like) | 0.8134 | 0.7812 | 0.8241 | 0.8903 |
| DS2 (CHB-MIT-like) | 0.8921 | 0.6943 | 0.7234 | 0.8512 |
| DS3 (Kaggle iEEG) | 0.9187 | 0.5821 | 0.6012 | 0.8234 |

### Regularizer Cross-Dataset Stability (C=1.0)

| Dataset | L1 F1 | L2 F1 | Elastic Net F1 | Winner |
|---------|-------|-------|----------------|--------|
| DS1 | 0.7798 | 0.7812 | **0.7834** | Elastic Net |
| DS2 | 0.6812 | 0.6754 | **0.6943** | Elastic Net (+1.31%) |
| DS3 | 0.5734 | 0.5612 | **0.5821** | Elastic Net (+0.87%) |

### Class Imbalance Handling (DS3 — 5% seizure rate, Elastic Net)

| Strategy | Precision | Recall | F1-Score | PR-AUC |
|----------|-----------|--------|----------|--------|
| No Resampling | 0.7234 | 0.3812 | 0.5012 | 0.5234 |
| Class Weighting | 0.5812 | 0.6934 | 0.6323 | 0.6712 |
| SMOTE | 0.6012 | 0.7123 | **0.6521** | 0.6923 |
| Random Oversampling | 0.5934 | 0.6812 | 0.6341 | 0.6634 |
| Random Undersampling | 0.4812 | 0.7234 | 0.5782 | 0.6012 |
| **SMOTE + Class Weight** | 0.5734 | **0.7512** | 0.6503 | **0.7012** |

---

## 📋 Notebook Structure

The main notebook (`seizure_prediction_notebook.ipynb`) is organized into 8 sections:

```
1. Setup & Installation           — library imports, global config
2. Dataset Collection             — synthetic EEG datasets, justification, visualization
3. Preprocessing Pipelines        — Pipeline A & B implementation + ordering effect study
4. Baseline Logistic Regression   — training, PR/ROC curves, confusion matrices
5. Overfitting & Underfitting     — 5-scenario C-sweep, learning curves
6. Regularization Study           — L1 vs L2 vs Elastic Net, sparsity paths, stability
7. Class Imbalance Handling       — 6 strategies, precision-recall tradeoff analysis
8. Comparative Analysis           — answers to all 4 research questions, heatmaps
```

---

## 📁 Repository Structure

```
seizure-prediction/
├── README.md                                  ← This file
├── requirements.txt                           ← Python dependencies
├── seizure_prediction_notebook.ipynb          ← Main Colab notebook (all code + outputs)
├── report/
│   └── Seizure_Prediction_IEEE_Report.pdf    ← Full IEEE-format written report
├── presentation/
│   └── Seizure_Prediction_Presentation.pptx 

```

---

## 🚀 How to Reproduce

### Google Colab

1. Upload `seizure_prediction_notebook.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Run **all cells top to bottom** — no external data download needed (datasets are synthetically generated within the notebook)
3. All figures will render inline; outputs appear in each cell


---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for pinned versions. Core packages:

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
scipy
jupyter
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🔍 Comparative Analysis — Final Answers

| Question | Answer | Strength |
|----------|--------|----------|
| **Q1: Does preprocessing order matter?** | ✅ YES — 1.3–4.8% F1 improvement with correct order | Strong |
| **Q2: Best regularizer cross-dataset?** | ✅ Elastic Net — lowest variance, most stable | Consistent |
| **Q3: EN always outperforms L1/L2?** | ⚠️ PARTIALLY — clear on DS2/DS3, marginal on DS1 | Conditional |
| **Q4: Imbalance × regularization interaction?** | ✅ STRONG — SMOTE+L1 maximizes recall; CW+EN best PR-AUC | Strong |

---

## 📚 References

1. R. G. Andrzejak et al., "Indications of nonlinear deterministic and finite-dimensional structures in time series of brain electrical activity," *Phys. Rev. E*, 2001.
2. A. H. Shoeb, "Application of machine learning to epileptic seizure onset detection and treatment," PhD thesis, MIT, 2009.
3. N. V. Chawla et al., "SMOTE: Synthetic minority over-sampling technique," *J. Artif. Intell. Res.*, vol. 16, pp. 321–357, 2002.
4. R. Tibshirani, "Regression shrinkage and selection via the Lasso," *J. Royal Stat. Soc. B*, 1996.
5. H. Zou and T. Hastie, "Regularization and variable selection via the elastic net," *J. Royal Stat. Soc. B*, 2005.
6. A. Subasi and M. I. Gursoy, "EEG signal classification using PCA, ICA, LDA and support vector machines," *Expert Syst. Appl.*, 2010.
7. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
8. H. He and E. A. Garcia, "Learning from imbalanced data," *IEEE TKDE*, 2009.

---

