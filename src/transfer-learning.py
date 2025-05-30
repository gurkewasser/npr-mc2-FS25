import pandas as pd
import numpy as np
import os
import glob
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from utils import set_seed

set_seed()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_data(train_path, val_path):
    df_train = pd.read_parquet(train_path)
    df_val   = pd.read_parquet(val_path)
    df_train['text'] = df_train['title'] + ' ' + df_train['content']
    df_val['text']   = df_val['title']   + ' ' + df_val['content']
    return df_train[['text','label']], df_val[['text','label']]

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, padding=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def train_and_evaluate(train_path, val_path, size, epochs=50, batch_size=8):
    run_name = f"transfer_{size}_{epochs}e"
    wandb.init(
        project='npr_mc2-main-2',
        name=run_name,
        reinit=True,
        config={
            'model': MODEL_NAME,
            'train_size': size,
            'epochs': epochs,
            'batch_size': batch_size
        }
    )

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorWithPadding(tokenizer)

    train_df, val_df = load_data(train_path, val_path)
    train_ds = Dataset.from_pandas(train_df)
    val_ds   = Dataset.from_pandas(val_df)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds   = val_ds.map(tokenize, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
    val_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    for param in model.base_model.parameters():
        param.requires_grad = False

    args = TrainingArguments(
        output_dir=os.path.join("results/transfer", f"{size}_{epochs}e"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=1,
        report_to='wandb',
        run_name=run_name
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)]
    )

    trainer.train()
    metrics = trainer.evaluate()
    metrics.update({'size': size, 'epochs': epochs})

    df = pd.DataFrame([metrics])
    metrics_file = os.path.join("results/transfer", f"transfer_metrics_{size}_3.csv")
    df.to_csv(metrics_file, index=False)
    wandb.log(metrics)
    wandb.save(metrics_file)
    wandb.finish()

if __name__ == '__main__':
    train_files = sorted(glob.glob("data/train_*.parquet")) 
    val_path = "data/validation.parquet"
    os.makedirs("results/transfer", exist_ok=True)
    epochs = 50
    batch_size = 32

    for train_path in train_files:
        base = os.path.basename(train_path)
        try:
            size = int(base.split("_")[1].split(".")[0])
        except Exception:
            print(f"Could not parse size from {base}, skipping.")
            continue
        train_and_evaluate(
            train_path=train_path,
            val_path=val_path,
            size=size,
            epochs=epochs,
            batch_size=batch_size
        )
