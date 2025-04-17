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
    parser = argparse.ArgumentParser(description="Classical TF-IDF + Logistic Regression Baseline with W&B logging")
    parser.add_argument('--train', default='data/train_100.parquet', help='Path to training parquet file')
    parser.add_argument('--val',   default='data/validation.parquet', help='Path to validation parquet file')
    parser.add_argument('--output_dir', default='results/classical', help='Directory to save models and metrics')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize W&B
    run_name = f"classical_{os.path.basename(args.train).split('.')[0]}"
    wandb.init(
        project="npr_mc2",
        name=run_name,
        config={
            "train_data": args.train,
            "val_data": args.val,
            "vectorizer_max_features": 5000,
            "vectorizer_ngram_range": (1,2),
            "classifier": "LogisticRegression",
            "max_iter": 1000
        }
    )

    train_X, train_y, val_X, val_y = load_data(args.train, args.val)
    metrics, vectorizer, clf = train_and_evaluate(train_X, train_y, val_X, val_y)

    # log metrics to W&B
    wandb.log(metrics)

    # Save results locally
    metrics_path = os.path.join(args.output_dir, 'classical_results.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(vectorizer, os.path.join(args.output_dir, 'tfidf_vectorizer.joblib'))
    joblib.dump(clf, os.path.join(args.output_dir, 'logreg_model.joblib'))

    # save artifacts to W&B
    wandb.save(metrics_path)
    wandb.save(os.path.join(args.output_dir, 'tfidf_vectorizer.joblib'))
    wandb.save(os.path.join(args.output_dir, 'logreg_model.joblib'))

    wandb.finish()

    print(f"Classical baseline metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
