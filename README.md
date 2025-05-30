# Sentiment Analysis Mini-Challenge (NPR MC2)

**Author:** Arian Iseni

---

## Overview

In this repository, I implement the FHNW “NPR Mini-Challenge 2: Sentiment Analysis” project. I explore how different amounts of manually annotated data—and the addition of weak labels—impact the performance of transformer-based sentiment classifiers.

---

## Project Structure

```
.
├── data
│   ├── archiv/NPR-Mini-Challenge-2-Sentiment-Analysis-1741246991.pdf
│   ├── labeled.parquet
│   ├── train_25.parquet
│   ├── train_50.parquet
│   ├── train_100.parquet
│   ├── train_150.parquet
│   ├── train_200.parquet
│   ├── train_250.parquet
│   ├── train_300.parquet
│   ├── train_350.parquet
│   ├── train_400.parquet
│   ├── validation.parquet
│   ├── unlabeled.parquet
│   └── unlabeled_pseudo.parquet
├── images
│   ├── training_size_300.png
│   └── umap_explanation.png
├── notebooks
│   ├── eda.ipynb
│   ├── embeddings.ipynb
│   ├── seed_check.ipynb
│   ├── splitting.ipynb
│   ├── weak_labeling.ipynb
│   └── exports
│       ├── eda.html
│       ├── embeddings.html
│       ├── seed_check.html
│       ├── splitting.html
│       └── weak_labeling.html
├── src
│   ├── classic-baseline.py
│   ├── fine-tuning.py
│   ├── transfer-learning.py
│   └── utils.py
├── results
│   ├── classical_combined.csv
│   ├── fine_tuning_combined.csv
│   ├── transfer_combined.csv
│   └── weak_labeling
│       └── best_weak_labeler_LogisticRegression.joblib
├── main.ipynb
├── requirements.txt
├── USE-OF-AI.md
├── NPR-Mini-Challenge-2-Sentiment-Analysis-1741246991.pdf
└── README.md
```

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/npr-mc2-sentiment-analysis.git
   cd npr-mc2-sentiment-analysis
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**

   * I place any additional `.parquet` files into the `data/` directory.
   * The provided splits (`train_*`, `validation.parquet`, `unlabeled.parquet`) are ready to use.

---

## Usage

### Exploratory Data Analysis

The exploratory data analysis is available in `notebooks/eda.ipynb`.

### Embeddings and Visualization

Analyzied embeddings with umap and distribution of the positive and negative labels was performed in `notebooks/embeddings.ipynb`

### Seed and Split Stability

To make sure the runs are reproducable, I've tested the seed settings in the `notebooks/seed_check.ipynb`, where i also proof that a static seed is needed.

I create hierarchical data splits:

Data splitting strategy is done in `notebooks/splitting.ipynb`

### Weak Label Generation

Finding the best embedding model and experimenting with different weak-labeling techniques was done in `notebooks/weak_labeling.ipynb`

### Model Training

For each training `classic`, `transfer`, `fine-tuning` is a separate python script in the folder `src`.

## Dependencies

See `requirements.txt` for a full list. Key packages I use include:

* transformers
* sentence-transformers
* scikit-learn
* pandas
* numpy
* wandb
* matplotlib
* umap-learn

---

## AI Usage

I document details on AI and ChatGPT usage in `USE-OF-AI.md`.
