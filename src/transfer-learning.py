import pandas as pd
import numpy as np
import argparse
import os
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_data(train_path, val_path):
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    df_train['text'] = df_train['title'] + ' ' + df_train['content']
    df_val['text']   = df_val['title']   + ' ' + df_val['content']
    return df_train[['text','label']], df_val[['text','label']]


def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length')


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


def train_for_size(size, epochs, batch_size, val_path, output_dir):
    # Paths
    train_path = f"data/train_{size}.parquet"
    run_name = f"transfer_{size}_{epochs}e"

    # W&B init
    wandb.init(
        project='npr_mc2',
        name=run_name,
        reinit=True,
        config={
            'model': MODEL_NAME,
            'train_size': size,
            'epochs': epochs,
            'batch_size': batch_size
        }
    )

    # Tokenizer & data collator
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer)

    # Data
    train_df, val_df = load_data(train_path, val_path)
    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    # Freeze backbone
    for param in model.base_model.parameters():
        param.requires_grad = False

    # TrainingArguments
    args = TrainingArguments(
        output_dir=os.path.join(output_dir, f"{size}_{epochs}e"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=1,
        report_to='wandb',
        run_name=run_name
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # Train + Eval
    trainer.train()
    metrics = trainer.evaluate()

    # Save metrics
    df = pd.DataFrame([metrics])
    metrics_file = os.path.join(output_dir, f"transfer_metrics_{size}_{epochs}e.csv")
    df.to_csv(metrics_file, index=False)
    wandb.log(metrics)
    wandb.save(metrics_file)
    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Transfer Learning over multiple train sizes"
    )
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[25,50,100,150,200,250,300],
                        help='List of training sizes')
    parser.add_argument('--epochs', nargs='+', type=int,
                        default=[3,10,20],
                        help='List of epoch settings')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=8,
                        help='Batch size per device')
    parser.add_argument('--val', default='data/validation.parquet',
                        help='Validation data parquet')
    parser.add_argument('--output_dir', '-o', default='results/transfer',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for size in args.sizes:
        for ep in args.epochs:
            train_for_size(size, ep, args.batch_size,
                           args.val, args.output_dir)


if __name__ == '__main__':
    main()
