# === classical_baseline.py ===
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import os
import json
import joblib
import wandb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_data(train_path, val_path):
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_train['text'] = df_train['title'] + ' ' + df_train['content']
    df_val['text']   = df_val['title']   + ' ' + df_val['content']
    return df_train['text'], df_train['label'], df_val['text'], df_val['label']


def train_and_evaluate(train_X, train_y, val_X, val_y):
    # 1) Vectorize with TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_X)
    X_val   = vectorizer.transform(val_X)

    # 2) Train a simple classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_y)

    # 3) Predict and compute metrics
    preds = clf.predict(X_val)
    precision, recall, f1, _ = precision_recall_fscore_support(val_y, preds, average='binary')
    acc = accuracy_score(val_y, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}, vectorizer, clf


def main():
    parser = argparse.ArgumentParser(description="Classical TF-IDF + Logistic Regression Baseline over multiple splits")
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[25,50,100,150,200,250,300],
                        help='List of training split sizes')
    parser.add_argument('--val', default='data/validation.parquet',
                        help='Path to validation parquet file')
    parser.add_argument('--output_dir', default='results/classical',
                        help='Base directory to save models and metrics')
    args = parser.parse_args()

    for size in args.sizes:
        train_path = f"data/train_{size}.parquet"
        subdir = os.path.join(args.output_dir, str(size))
        os.makedirs(subdir, exist_ok=True)

        # Initialize W&B for this split
        run_name = f"classical_{size}"
        wandb.init(
            project="npr_mc2",
            name=run_name,
            reinit=True,
            config={
                "train_split": train_path,
                "val_split": args.val,
                "vectorizer_max_features": 5000,
                "vectorizer_ngram_range": (1,2),
                "classifier": "LogisticRegression",
                "max_iter": 1000
            }
        )

        # Load and run
        train_X, train_y, val_X, val_y = load_data(train_path, args.val)
        metrics, vectorizer, clf = train_and_evaluate(train_X, train_y, val_X, val_y)

        # Log and save
        wandb.log(metrics)
        metrics_file = os.path.join(subdir, 'classical_results.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        joblib.dump(vectorizer, os.path.join(subdir, 'tfidf_vectorizer.joblib'))
        joblib.dump(clf, os.path.join(subdir, 'logreg_model.joblib'))

        # Save to W&B
        wandb.save(metrics_file)
        wandb.save(os.path.join(subdir, 'tfidf_vectorizer.joblib'))
        wandb.save(os.path.join(subdir, 'logreg_model.joblib'))
        wandb.finish()

        print(f"Finished classical baseline for size={size}. Metrics in {metrics_file}")

    print("All classical baseline splits completed.")


if __name__ == '__main__':
    main()
