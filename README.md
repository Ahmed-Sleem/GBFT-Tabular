# GBFT: Gradient-Boosted Feature Transformer

[![Kaggle](https://img.shields.io/badge/Kaggle-Notebooks-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Bridging Tabular Learning and Natural Language Understanding through Tree-Based Semantic Encoding**  
> Ahmed Sleem, Nihal Ahmed Adly, Gamila S. Elfayoumi, Ahmed B. Zaky  
> Egypt-Japan University of Science and Technology (E-JUST)

**[📊 Results](#-main-results)** | **[💻 Notebooks](#-code-notebooks)** | **[📈 Visualizations](#-visualizations)**

---

## 📋 Overview

We introduce GBFT, a framework combining tree-based learning with neural networks through two architectures:

- **BERT-GBFT**: Encodes tree decision paths as natural language processed by BERT (AUC 0.929, F1 0.622, **+15.2% over XGBoost**)
- **Enhanced GBFT**: Hierarchical GBDT + Transformer fusion achieving **best F1-scores** (0.711 Adult, 0.633 Bank Marketing)

**Key Achievements**:
- 🏆 7.19% ablation-validated BERT semantic contribution
- 🏆 17% F1 improvement over XGBoost on imbalanced data (88:12)
- ⚡ 179-184K parameters (100× smaller than neural baselines)
- ⚡ Sub-2ms inference for real-time deployment

---

## 🏆 Main Results

### BERT-GBFT: Semantic Understanding (Bank Marketing Dataset)

<div align="center">

| Model | AUC | F1 | Precision | Recall | MCC |
|-------|-----|-----|-----------|--------|-----|
| **BERT-GBFT** | 0.929 | **0.622** | 0.578 | **0.671** | **0.568** |
| XGBoost | **0.932** | 0.541 | 0.657 | 0.459 | 0.501 |
| LightGBM | 0.931 | 0.537 | 0.648 | 0.458 | 0.496 |
| Enhanced GBFT | 0.928 | 0.633 | 0.577 | 0.700 | 0.582 |
| FT-Transformer | 0.915 | 0.542 | 0.626 | 0.478 | 0.496 |

</div>

**Key Finding**: **15.2% F1 improvement** over XGBoost (0.622 vs 0.541) while enabling natural language explanations like *"Why was customer X classified as high-risk?"*

<p align="center">
  <img src="bert version/results/model_comparison.png" width="70%"/>
  <br><i>BERT-GBFT Performance Comparison</i>
</p>

---

### Ablation Study: BERT Semantic Contribution Validation

<div align="center">

| Variant | F1 | AUC | Recall | Parameters |
|---------|-----|-----|--------|------------|
| **BERT-GBFT (Full)** | **0.621** | 0.927 | **0.671** | 271K |
| Without BERT Stream | 0.579 | 0.926 | 0.549 | 155K |
| **Improvement** | **+7.19%** | +0.1% | **+22.2%** | +75% |

</div>

**Conclusion**: BERT semantic embeddings provide **statistically significant performance gains** beyond architectural complexity, validating the semantic understanding hypothesis.

---

### Enhanced GBFT: Maximum Performance

#### Adult Income Dataset (32,561 samples, 76:24 imbalance)

<div align="center">

| Model | AUC | F1 | Precision | Recall | MCC |
|-------|-----|-----|-----------|--------|-----|
| **Enhanced GBFT** | 0.921 | **0.711** | 0.753 | **0.672** | 0.623 |
| XGBoost | **0.922** | 0.707 | **0.781** | 0.647 | **0.627** |
| LightGBM | 0.922 | 0.705 | 0.778 | 0.645 | 0.624 |
| CatBoost | 0.920 | 0.687 | 0.778 | 0.615 | 0.605 |
| FT-Transformer | 0.906 | 0.654 | 0.754 | 0.577 | 0.567 |

</div>

<p align="center">
  <img src="on adult dataset/results/model_comparison.png" width="70%"/>
  <br><i>Enhanced GBFT - Adult Income Dataset</i>
</p>

---

#### Bank Marketing Dataset (45,211 samples, 88:12 severe imbalance)

<div align="center">

| Model | AUC | F1 | Precision | Recall | MCC |
|-------|-----|-----|-----------|--------|-----|
| **Enhanced GBFT** | 0.928 | **0.633** | 0.577 | **0.700** | **0.582** |
| XGBoost | **0.932** | 0.541 | 0.657 | 0.459 | 0.501 |
| LightGBM | 0.931 | 0.537 | 0.648 | 0.458 | 0.496 |
| CatBoost | 0.929 | 0.520 | 0.661 | 0.429 | 0.485 |
| FT-Transformer | 0.915 | 0.542 | 0.626 | 0.478 | 0.496 |

</div>

**Key Finding**: **17% F1 improvement** over XGBoost (0.633 vs 0.541) with **53% recall improvement** (0.700 vs 0.459) on severely imbalanced data.

<p align="center">
  <img src="on bank marketing dataset/results/model_comparison.png" width="70%"/>
  <br><i>Enhanced GBFT - Bank Marketing Dataset</i>
</p>

---

## 📈 Visualizations

### ROC & Precision-Recall Curves

<table>
<tr>
<td width="50%"><img src="bert version/results/roc_curves.png" width="100%"/></td>
<td width="50%"><img src="bert version/results/precision_recall_curves.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>ROC Curves (BERT-GBFT)</b></td>
<td align="center"><b>Precision-Recall Curves</b></td>
</tr>
</table>

### BERT Semantic Embeddings & Model Analysis

<table>
<tr>
<td width="50%"><img src="bert version/results/bert_embeddings_tsne.png" width="100%"/></td>
<td width="50%"><img src="bert version/results/embedding_comparison.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>BERT Decision Path Embeddings (t-SNE)</b></td>
<td align="center"><b>BERT vs GBDT Feature Comparison (PCA)</b></td>
</tr>
</table>

### Training Dynamics & Complexity Analysis

<table>
<tr>
<td width="50%"><img src="on adult dataset/results/training_curves_comparison.png" width="100%"/></td>
<td width="50%"><img src="on adult dataset/results/complexity_comparison.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>Training Convergence</b></td>
<td align="center"><b>Model Complexity Comparison</b></td>
</tr>
</table>

### Confusion Matrices & Calibration

<table>
<tr>
<td width="50%"><img src="bert version/results/confusion_matrices.png" width="100%"/></td>
<td width="50%"><img src="bert version/results/calibration_curves.png" width="100%"/></td>
</tr>
<tr>
<td align="center"><b>Confusion Matrices (All Models)</b></td>
<td align="center"><b>Model Calibration Analysis</b></td>
</tr>
</table>

---

## 💻 Code Notebooks

All experiments run on **Kaggle** with reproducible notebooks:

### BERT-GBFT (Semantic Understanding)
- **[GBFT-tabular-with-bert.ipynb](bert%20version/GBFT-tabular-with-bert.ipynb)** - Main BERT-GBFT experiment on Bank Marketing dataset
- **[gbft-tabular-with-bert-ablation-study.ipynb](bert%20version/gbft-tabular-with-bert-ablation-study-ipynb.ipynb)** - Ablation study validating BERT semantic contribution (7.19% improvement)
- **Results**: All visualizations and metrics in [bert version/results/](bert%20version/results/)

### Enhanced GBFT (Maximum Performance)
- **[gbft-tabular-on-adult-dataset.ipynb](on%20adult%20dataset/gbft-tabular-on-adult-dataset.ipynb)** - Enhanced GBFT on Adult Income (32,561 samples, F1=0.711)
- **[gbft-tabular-on-bank-marketing.ipynb](on%20bank%20marketing%20dataset/gbft-tabular-on-bank-marketing.ipynb)** - Enhanced GBFT on Bank Marketing (45,211 samples, F1=0.633)
- **Results**: Comprehensive analysis in [on adult dataset/results/](on%20adult%20dataset/results/) and [on bank marketing dataset/results/](on%20bank%20marketing%20dataset/results/)

---

## 📁 Repository Structure
```text
GBFT-Tabular/
│
├── bert version/
│ ├── GBFT-tabular-with-bert.ipynb # BERT-GBFT main experiment
│ ├── gbft-tabular-with-bert-ablation-study.ipynb # Ablation study
│ ├── gbft-tabular-with-bert2.ipynb # Additional BERT variant
│ └── results/
│ ├── model_comparison.png # Performance comparison
│ ├── bert_embeddings_tsne.png # Semantic embeddings visualization
│ ├── embedding_comparison.png # BERT vs GBDT features (PCA)
│ ├── roc_curves.png # ROC curves
│ ├── precision_recall_curves.png # PR curves
│ ├── confusion_matrices.png # Confusion matrices
│ ├── calibration_curves.png # Model calibration
│ ├── complexity_comparison.png # Efficiency analysis
│ ├── correlation_matrix.png # Feature correlations
│ ├── dataset_overview.png # Data distribution
│ ├── detailed_metrics.csv # Complete metrics
│ └── results_bert_enhanced.csv # Results summary
│
├── on adult dataset/
│ ├── gbft-tabular-on-adult-dataset.ipynb # Enhanced GBFT experiment
│ └── results/
│ ├── model_comparison.png
│ ├── roc_curves.png
│ ├── precision_recall_curves.png
│ ├── confusion_matrices.png
│ ├── calibration_curves.png
│ ├── training_curves_comparison.png
│ ├── complexity_comparison.png
│ ├── correlation_matrix.png
│ ├── dataset_overview.png
│ ├── results.csv
│ ├── detailed_metrics.csv
│ └── complexity_metrics.csv
│
├── on bank marketing dataset/
│ ├── gbft-tabular-on-bank-marketing.ipynb # Enhanced GBFT experiment
│ └── results/
│ └── [Same structure as adult dataset]
│
├── LICENSE # Apache-2.0 License
└── README.md # This file
```

---

## 📊 Datasets

### 1. Adult Income Dataset
- **Source**: [UCI Adult Census Income (Kaggle)](https://www.kaggle.com/datasets/a7medsleem/uci-adult-census-income-1994)
- **Task**: Binary classification (income >$50K)
- **Samples**: 32,561 | **Features**: 101 (after encoding) | **Imbalance**: 76:24

### 2. Bank Marketing Dataset
- **Source**: [UCI Bank Marketing (Kaggle)](https://www.kaggle.com/datasets/a7medsleem/uci-bank-marketing-dataset)
- **Task**: Binary classification (term deposit subscription)
- **Samples**: 45,211 | **Features**: 52 (after encoding) | **Imbalance**: 88:12 (severe)

**Preprocessing**: Stratified train/validation/test split (60/20/20), StandardScaler for numerical features, one-hot encoding for categorical features, random seed 42 for reproducibility.

---

## 🔬 Key Contributions

1. **Semantic tabular understanding method**: Tree decision paths → Natural language → BERT embeddings
2. **Rigorous ablation validation**: 7.19% F1 improvement from BERT semantic embeddings (not just architecture)
3. **State-of-the-art F1 on imbalanced data**: 17% improvement over XGBoost on Bank Marketing (88:12 imbalance)
4. **Efficient deployment-ready architecture**: 179-184K parameters (100× smaller than neural baselines), sub-2ms inference

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{sleem2024gbft,
  title={Bridging Tabular Learning and Natural Language Understanding through Tree-Based Semantic Encoding},
  author={Sleem, Ahmed and Adly, Nihal Ahmed and Elfayoumi, Gamila S. and Zaky, Ahmed B.},
  year={2024},
  institution={Egypt-Japan University of Science and Technology (E-JUST)},
  url={https://github.com/Ahmed-Sleem/GBFT-Tabular}
}
```

## 📧 Contact

**Ahmed Sleem**  
📧 ahmad.muhamad@ejust.edu.eg  
📧 ahmedsleemsocial@gmail.com
🏛️ Egypt-Japan University of Science and Technology (E-JUST), Alexandria, Egypt

**Co-authors**:
- Nihal Ahmed Adly (nihal.abdelmonem@ejust.edu.eg)
- Gamila S. Elfayoumi (gamila.elfayoumi@ejust.edu.eg)
- Ahmed B. Zaky (ahmed.zaky@ejust.edu.eg, ahmed.zaky@feng.bu.edu.eg)

---

## 📜 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

```text
Copyright 2024 Ahmed Sleem, Egypt-Japan University of Science and Technology

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

<div align="center">
Egypt-Japan University of Science and Technology (E-JUST)

⭐ If you find this work useful, please cite our paper and star this repository!
</div>

