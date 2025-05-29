import pandas as pd
import numpy as np
import argparse
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import set_seed

set_seed()

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
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def main():
    parser = argparse.ArgumentParser(description="Classical TF-IDF + Logistic Regression Baseline over multiple splits")
    parser.add_argument('--train_dir', default='data', help='Directory containing train_{size}.parquet files')
    parser.add_argument('--val', default='data/validation.parquet',
                        help='Path to validation parquet file')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save metrics CSVs')
    args = parser.parse_args()

    # Always save to results/classic regardless of --output_dir
    output_dir = os.path.join('results', 'classic')
    os.makedirs(output_dir, exist_ok=True)

    # Find all train_*.parquet files in the train_dir
    train_files = [f for f in os.listdir(args.train_dir) if f.startswith('train_') and f.endswith('.parquet')]
    train_files = sorted(train_files, key=lambda x: int(x.split('_')[1].split('.')[0]))

    if not train_files:
        print(f"No train_*.parquet files found in {args.train_dir}")
        return

    for train_file in train_files:
        # Extract train size from filename
        try:
            size = int(train_file.split('_')[1].split('.')[0])
        except Exception:
            print(f"Skipping file {train_file}: could not parse train size.")
            continue

        train_path = os.path.join(args.train_dir, train_file)
        print(f"Processing train size {size} from {train_path} ...")
        train_X, train_y, val_X, val_y = load_data(train_path, args.val)
        metrics = train_and_evaluate(train_X, train_y, val_X, val_y)
        metrics['train_size'] = size

        # Save metrics as a single-row CSV for this run
        results_df = pd.DataFrame([metrics])
        results_csv = os.path.join(output_dir, f'classical_results_{size}.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"Finished classical baseline for size={size}. Metrics saved to {results_csv}")

    print("All classical baseline splits completed.")

if __name__ == '__main__':
    main()
