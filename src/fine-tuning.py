from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
import datetime
import os
import wandb

# ---------- Global Config ----------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ---------- Helpers ----------
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="binary"
    )
    acc = accuracy_score(p.label_ids, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ---------- Main Training Function ----------
def train_and_evaluate(
    size,
    num_epochs,
    val_path="data/validation.parquet",
    base_path="data/train_"
):
    # Start a fresh wandb run
    run_name = f"fine-tuning_{size}_{num_epochs}e"
    wandb.init(
        project="npr_mc2-test",
        name=run_name,
        reinit=True
    )
    log(f"Training with {size} samples for {num_epochs} epochs...")

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./results/{size}_{num_epochs}e",
        run_name=run_name,
        report_to="wandb",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        logging_dir=f'./logs/{size}_{num_epochs}e',
    )

    # Load & preprocess data
    df_train = pd.read_parquet(f"{base_path}{size}.parquet")
    df_val   = pd.read_parquet(val_path)
    df_train["text"] = df_train["title"] + " " + df_train["content"]
    df_val["text"]   = df_val["title"]   + " " + df_val["content"]

    train_ds = Dataset.from_pandas(df_train[["text","label"]])
    val_ds   = Dataset.from_pandas(df_val[["text","label"]])

    train_ds = train_ds.map(tokenize_function, batched=True)
    val_ds   = val_ds.map(tokenize_function, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    val_ds.set_format(  type="torch", columns=["input_ids","attention_mask","label"])

    # Model & Trainer (with early stopping)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    # Train + Eval
    trainer.train()
    metrics = trainer.evaluate()
    metrics["train_size"]  = size
    metrics["num_epochs"]  = num_epochs

    # Finish the wandb run
    wandb.finish()

    # Save this run’s metrics to its own CSV
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    metrics_file = os.path.join(
        out_dir,
        f"baseline_metrics_{size}_{num_epochs}e.csv"
    )
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
    log(f"✅ Metrics saved to {metrics_file}\n")

    return metrics


if __name__ == "__main__":
    sizes = [25, 50, 100, 150, 200, 250, 300]
    epoch_settings = [3, 10, 20]

    for size in sizes:
        for num_epochs in epoch_settings:
            train_and_evaluate(size, num_epochs)