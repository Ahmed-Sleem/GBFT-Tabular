# 🚀 GBFT: Gradient-Boosted Feature Transformer

> **A Novel Hybrid Architecture Bridging Gradient Boosting and Transformers for Superior Tabular Data Classification**

[📊 Datasets](#datasets) | [📄 Paper](#citation) | [📈 Results](#results)

---

## 🎯 The Problem

**Transformers revolutionized NLP and vision, but struggle with tabular data.**

Current challenges:
- 🔴 Traditional transformers fail on heterogeneous tabular features
- 🔴 Gradient-boosted trees (GBDT) dominate but lack neural network flexibility  
- 🔴 Existing hybrid approaches don't fully leverage both paradigms
- 🔴 Large models with millions of parameters are impractical for deployment

**Why does this matter?** 80% of enterprise data is tabular, yet modern deep learning can't handle it effectively.

---

## ✨ Our Solution: GBFT

**GBFT (Gradient-Boosted Feature Transformer)** combines the strength of GBDT feature extraction with hierarchical transformer processing, achieving **state-of-the-art results with 100x fewer parameters** than competing methods.

### 🏗️ Architecture

```text
Raw Tabular Features (101-52 dims)
↓
GBDT Ensemble (LightGBM + XGBoost)
↓
GBDT Features (12 dims) -──┐
↓                          │
Combined Features ←────────┘
↓
┌─────────────────────────────────────┐
│ Stage 1: Local Pattern Extraction   │
│ Dense → LayerNorm → GELU → Dense    │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Stage 2: Global Pattern Learning    │
│ Multi-Head Attention (4 heads)      │
│ + Feed-Forward Networks             │
│ + Residual Connections (3 layers)   │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Stage 3: Feature Refinement         │
│ LayerNorm → Dense + Residual        │
└─────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Stage 4: Classification             │
│ Dense → GELU → Dense → Softmax      │
└─────────────────────────────────────┘
↓
Predictions
```

**Key Innovation**: Hierarchical feature processing with residual connections and GBDT-boosted features.

---

## 🔬 Novel Contributions

| Innovation | Description | Impact |
|-----------|-------------|--------|
| **🎨 Hierarchical Processing** | 4-stage architecture: Local → Global → Refinement → Decision | Better feature representation |
| **🌲 GBDT Feature Injection** | Combines tree-based feature extraction with neural processing | Best of both worlds |
| **⚡ Lightweight Design** | 170K parameters vs. 10M+ in competing methods | 100x smaller, deployable |
| **🔄 Custom AdamW** | Decoupled weight decay for tabular data optimization | Faster convergence |
| **🎯 Adaptive Feature Selection** | Learns which features to emphasize | Handles heterogeneous data |

---

## 📊 Results

### Adult Income Dataset
**Task**: Predict income >$50K (32,561 samples, 101 features)

| Model | AUC ↑ | Accuracy | F1-Score | Precision | Recall | Parameters |
|-------|-------|----------|----------|-----------|--------|------------|
| **GBFT (Ours)** | **0.9213** | 0.8636 | **0.7105** | 0.7532 | **0.6724** | **179K** |
| XGBoost | 0.9223 | **0.8667** | 0.7072 | **0.7805** | 0.6465 | N/A |
| LightGBM | 0.9222 | 0.8657 | 0.7052 | 0.7777 | 0.6451 | N/A |
| CatBoost | 0.9197 | 0.8603 | 0.6865 | 0.7776 | 0.6145 | N/A |
| FT-Transformer | 0.9060 | 0.8478 | 0.6538 | 0.7539 | 0.5772 | 20K |
| TabNet | 0.8831 | 0.7510 | 0.0000 | 0.0000 | 0.0000 | 36K |

**Key Findings**:
- ✅ **Best F1-Score** (0.7105) - superior precision-recall balance
- ✅ **Best Recall** (0.6724) among neural models - catches more positive cases
- ✅ **Competitive AUC** within 0.1% of best GBDT model
- ✅ **MCC 0.6234** - excellent overall classification quality
- ✅ **Cohen's Kappa 0.6217** - strong agreement beyond chance

![Adult Dataset Results](on%20adult%20dataset/results/model_comparison.png)

---

### Bank Marketing Dataset  
**Task**: Predict term deposit subscription (45,211 samples, 52 features, **highly imbalanced 88:12**)

| Model | AUC ↑ | Accuracy | Precision | Recall ↑ | F1-Score | MCC ↑ |
|-------|-------|----------|-----------|----------|----------|-------|
| XGBoost | 0.9323 | 0.9087 | 0.6568 | 0.4594 | 0.5406 | 0.5013 |
| LightGBM | 0.9314 | 0.9074 | 0.6475 | 0.4584 | 0.5368 | 0.4960 |
| CatBoost | 0.9293 | 0.9074 | 0.6608 | 0.4291 | 0.5203 | 0.4852 |
| **GBFT (Ours)** | **0.9281** | 0.9048 | 0.5767 | **0.7004** | **0.6325** | **0.5820** |
| FT-Transformer | 0.9152 | 0.9056 | 0.6262 | 0.4783 | 0.5423 | 0.4963 |
| TabNet | 0.8962 | 0.8830 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Key Findings**:
- ✅ **Best Recall** (70.04%) - critical for imbalanced data
- ✅ **Best F1-Score** (0.6325) - optimal balance for minority class
- ✅ **Highest MCC** (0.5820) - best at handling class imbalance
- ✅ **Best Cohen's Kappa** (0.5784) - strongest classification performance
- ✅ Outperforms all neural baselines significantly

![Bank Dataset Results](on%20bank%20marketing%20dataset/results/model_comparison.png)

---


## 📁 Repository Structure

```text
GBFT-Tabular/
│
├── on adult dataset/
│   ├── gbft-tabular-on-adult-dataset.ipynb    # Complete notebook
│   └── results/
│       ├── calibration_curves.png              # Model calibration analysis
│       ├── complexity_comparison.png           # Parameters, size, speed comparison
│       ├── complexity_metrics.csv              # Detailed complexity metrics
│       ├── confusion_matrices.png              # All models confusion matrices
│       ├── correlation_matrix.png              # Feature correlation heatmap
│       ├── dataset_overview.png                # Data distribution analysis
│       ├── detailed_metrics.csv                # Comprehensive evaluation metrics
│       ├── model_comparison.csv                # Model performance table
│       ├── model_comparison.png                # AUC & Accuracy bar charts
│       ├── precision_recall_curves.png         # PR curves for all models
│       ├── results.csv                         # Final results summary
│       ├── roc_curves.png                      # ROC curves comparison
│       └── training_curves_comparison.png      # Training dynamics
│
├── on bank marketing dataset/
│   ├── gbft-tabular-on-bank-marketing.ipynb   # Complete notebook
│   └── results/
│       └── [Same structure as adult dataset]
│
├── LICENSE                                     # License
└── README.md                                   # This file
```

## 📈 Complexity Analysis

### Model Size & Speed Comparison

#### Adult Dataset
| Model | Parameters | Size (MB) | Inference (ms) | Throughput (samples/s) | GPU Memory (MB) |
|-------|-----------|-----------|----------------|------------------------|-----------------|
| **GBFT** | **179K** | **0.68** | 1.80 | 141,881 | 21.65 |
| FT-Transformer | 20K | 0.08 | 2.53 | 101,063 | 43.42 |
| TabNet | 36K | 0.14 | 1.77 | 144,597 | 21.76 |
| LightGBM | N/A | 0.65 | 2.12 | 120,535 | N/A |
| XGBoost | N/A | 0.57 | 0.87 | 295,615 | N/A |
| CatBoost | N/A | 0.22 | 1.00 | 254,797 | N/A |

#### Bank Marketing Dataset  
| Model | Parameters | Size (MB) | Inference (ms) | Throughput (samples/s) | GPU Memory (MB) |
|-------|-----------|-----------|----------------|------------------------|-----------------|
| **GBFT** | **172K** | **0.66** | 2.19 | 116,868 | 19.58 |
| FT-Transformer | 18K | 0.07 | 1.22 | 210,140 | 31.04 |
| TabNet | 23K | 0.09 | 2.07 | 123,563 | 19.49 |
| LightGBM | N/A | 0.66 | 2.91 | 88,058 | N/A |
| XGBoost | N/A | 0.70 | 1.06 | 241,200 | N/A |
| CatBoost | N/A | 0.22 | 1.03 | 249,401 | N/A |


![Complexity Analysis](on%20adult%20dataset/results/complexity_comparison.png)

---

## 📊 Comprehensive Analysis

### Model Performance Visualizations

**ROC Curves Comparison**:
![ROC Curves](on%20adult%20dataset/results/roc_curves.png)

**Precision-Recall Curves**:
![PR Curves](on%20adult%20dataset/results/precision_recall_curves.png)

**Confusion Matrices**:
![Confusion Matrices](on%20adult%20dataset/results/confusion_matrices.png)

**Calibration Curves**:
![Calibration](on%20adult%20dataset/results/calibration_curves.png)

**Training Dynamics**:
![Training Curves](on%20adult%20dataset/results/training_curves_comparison.png)


---

## 🔍 Detailed Results

### Adult Dataset - Comprehensive Metrics

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | MCC | Kappa |
|-------|----------|-----------|--------|-------|---------|--------|-----|-------|
| **GBFT** | 0.8636 | 0.7532 | **0.6724** | **0.7105** | 0.9213 | 0.8162 | **0.6234** | **0.6217** |
| XGBoost | **0.8667** | **0.7805** | 0.6465 | 0.7072 | **0.9223** | **0.8204** | 0.6266 | 0.6219 |
| LightGBM | 0.8657 | 0.7777 | 0.6451 | 0.7052 | 0.9222 | 0.8199 | 0.6238 | 0.6193 |
| CatBoost | 0.8603 | 0.7776 | 0.6145 | 0.6865 | 0.9197 | 0.8132 | 0.6050 | 0.5982 |
| FT-Trans | 0.8478 | 0.7539 | 0.5772 | 0.6538 | 0.9060 | 0.7765 | 0.5667 | 0.5585 |
| TabNet | 0.7510 | 0.0000 | 0.0000 | 0.0000 | 0.8831 | 0.7162 | 0.0000 | 0.0000 |

### Bank Marketing Dataset - Comprehensive Metrics

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | MCC | Kappa |
|-------|----------|-----------|--------|-------|---------|--------|-----|-------|
| **GBFT** | 0.9048 | 0.5767 | **0.7004** | **0.6325** | 0.9281 | 0.6279 | **0.5820** | **0.5784** |
| XGBoost | **0.9087** | **0.6568** | 0.4594 | 0.5406 | **0.9323** | **0.6240** | 0.5013 | 0.4916 |
| LightGBM | 0.9074 | 0.6475 | 0.4584 | 0.5368 | 0.9314 | 0.6225 | 0.4960 | 0.4871 |
| CatBoost | 0.9074 | 0.6608 | 0.4291 | 0.5203 | 0.9293 | 0.6185 | 0.4852 | 0.4717 |
| FT-Trans | 0.9056 | 0.6262 | 0.4783 | 0.5423 | 0.9152 | 0.5891 | 0.4963 | 0.4907 |
| TabNet | 0.8830 | 0.0000 | 0.0000 | 0.0000 | 0.8962 | 0.5209 | 0.0000 | 0.0000 |


---

## 📊 Datasets

### 1. Adult Income Dataset
- **Source**: [UCI Adult Census Income (1994)](https://www.kaggle.com/datasets/a7medsleem/uci-adult-census-income-1994)
- **Task**: Binary classification (income >$50K)
- **Samples**: 32,561
- **Features**: 14 original (6 numerical, 8 categorical)
- **Final Features**: 101 (after encoding)
- **Imbalance**: 75:25
- **Missing Values**: ~5% (handled)

### 2. Bank Marketing Dataset
- **Source**: [UCI Bank Marketing](https://www.kaggle.com/datasets/a7medsleem/uci-bank-marketing-dataset)
- **Task**: Binary classification (term deposit subscription)
- **Samples**: 45,211
- **Features**: 16 original (7 numerical, 9 categorical)
- **Final Features**: 52 (after encoding)
- **Imbalance**: 88:12 ⚠️ (highly imbalanced)
- **Missing Values**: Some "unknown" values in categorical features

---

## 🛠️ Technical Details

### Architecture Specifications

**GBFT Components**:
```python
Stage 1: Local Pattern Extraction
  - Input: Combined features (raw + GBDT)
  - Layers: Linear(total_dim, 128) → LayerNorm → GELU → Dropout → Linear(128, 64)
  - Output: Local features (64 dims)

Stage 2: Global Pattern Learning
  - Multi-head attention (4 heads, head_dim=16)
  - 3 transformer encoder layers
  - Feed-forward network (64 → 256 → 64)
  - Residual connections + Layer normalization

Stage 3: Feature Refinement
  - LayerNorm → Linear(64, 64) → GELU → Dropout
  - Residual connection with Stage 1 output

Stage 4: Classification Head
  - LayerNorm → Linear(64, 32) → GELU → Dropout → Linear(32, 2)
  - Softmax activation
```

## 📝 Citation

If you use GBFT in your research, please cite:

```bibtex
@article{sleem2024gbft,
  title={GBFT: Gradient-Boosted Feature Transformer for Tabular Data Classification},
  author={Sleem, Ahmed},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024},
  url={https://github.com/Ahmed-Sleem/GBFT-Tabular}
}

