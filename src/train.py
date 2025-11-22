"""
train.py
Description: This main file is used to train the model
Authors: Matthew Hake, Ben Chidley, Garret Keyhani, Joshua Smith 
Date: 11/21/2025
"""

SEED = 42
import json
import os
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from data import IMDBDataModule
from model import IMDBBiLSTM

def evaluate_and_save_examples(model: IMDBBiLSTM, dm: IMDBDataModule, ckpt_path: str = "best", device: str = None):

    ckpt_file = ckpt_path
    model = IMDBBiLSTM.load_from_checkpoint(ckpt_file)
    model.eval()
    model.to(device)

    test_dataset = dm.test_dataset 
    tokenizer = dm.tokenizer

    batch_size = dm.batch_size
    texts = []
    labels = []
    input_id_batches = []

    for i in range(0, len(test_dataset), batch_size):
        batch_slice = test_dataset[i : i + batch_size]
        batch_texts = batch_slice["text"]
        batch_labels = batch_slice["label"]
        enc = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=dm.max_length,
            return_tensors="pt"
        )
        input_id_batches.append(enc["input_ids"])
        texts.extend(batch_texts)
        labels.extend(batch_labels)

    all_preds = []
    all_labels = []
    all_texts = []

    with torch.no_grad():
        for input_ids in input_id_batches:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            all_preds.extend(preds)

    all_labels = labels 
    all_texts = texts

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["negative", "positive"], digits=4)

    print(f"\nTest accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:\n")
    print(report)

    misclassified = []
    for text, true, pred in zip(all_texts, all_labels, all_preds):
        if true != pred:
            misclassified.append({"text": text, "true": int(true), "pred": int(pred)})
            if len(misclassified) >= 3:
                break

    out_file = "misclassified_examples.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in misclassified:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(misclassified)} misclassified examples to {out_file}")

def main():
    dm = IMDBDataModule(batch_size=32, max_length=256)
    dm.prepare_data()
    dm.setup()

    model = IMDBBiLSTM(
        vocab_size=dm.vocab_size,
        pad_idx=dm.pad_idx,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3,
        lr=1e-3,
    )

    wandb_logger = WandbLogger(project="CS-366 Assignment 3", name="increase dropout to .5", log_model=False)
    wandb_logger.log_hyperparams({
        "batch_size": dm.batch_size,
        "max_length": dm.max_length,
        "embedding_dim": model.hparams.embedding_dim,
        "hidden_dim": model.hparams.hidden_dim,
        "num_layers": model.hparams.num_layers,
        "dropout": model.hparams.dropout,
        "lr": model.hparams.lr,
        "seed": SEED,
    })

    checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-model"
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=3, mode="min")

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint, early_stop],
        logger=wandb_logger,
        deterministic=True,
    )

    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)

    best_path = checkpoint.best_model_path if checkpoint.best_model_path else "best-model.ckpt"
    if best_path:
        try:
            best_ckpt = best_path
            eval_model = IMDBBiLSTM.load_from_checkpoint(best_ckpt)
            evaluate_and_save_examples(eval_model, dm, ckpt_path=best_ckpt)
        except Exception as e:
            print("Could not load checkpoint for extra evaluation, will attempt to evaluate with current model:", e)
            evaluate_and_save_examples(model, dm)

if __name__ == "__main__":
    main()
