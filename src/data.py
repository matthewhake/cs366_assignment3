"""
data.py
Description: This main file to preprocess the data
Authors: Matthew Hake, Ben Chidley, Garret Keyhani, Joshua Smith 
Date: 11/21/2025
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset, concatenate_datasets
import pytorch_lightning as pl
import numpy as np

pl.seed_everything(42)

# Source: https://www.sabrepc.com/blog/Deep-Learning-and-AI/text-classification-with-bert
# Use: Explains how to initialize a generic text classification dataset
class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row['text']
        label = row['label']

        # Cut if review is too long. Add blank space if review is too short
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }


class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, max_length=256):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def download_and_split(self):
        # Loading imdb dataset
        train = load_dataset("imdb", split="train")
        test  = load_dataset("imdb", split="test")
       
        # Combining preset train and test (initially 50/50)
        full = concatenate_datasets([train,test]).shuffle(seed=42)
        
        # t/v/t split 70/15/15
        total = len(full)
        train_size = int(0.7 * total)
        val_size   = int(0.15 * total)
        test_size  = total - train_size - val_size
        splits = full.train_test_split(
            train_size=train_size, 
            test_size=val_size + test_size,
            seed=42
        )
        self.train_dataset = splits['train']
        temp_dataset = splits['test']
        val_test_splits = temp_dataset.train_test_split(test_size=test_size, seed=42)
        self.val_dataset = val_test_splits['train']
        self.test_dataset = val_test_splits['test']

    def setup(self, stage=None):
        if self.train_dataset is None:
            self.download_and_split()

        self.vocab_size = self.tokenizer.vocab_size
        self.pad_idx = self.tokenizer.pad_token_id

    def train_dataloader(self):
        return DataLoader(
            IMDBDataset(self.train_dataset, self.tokenizer, self.max_length),
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2)

    def val_dataloader(self):
        return DataLoader(
            IMDBDataset(self.val_dataset, self.tokenizer, self.max_length),
            batch_size=self.batch_size, 
            num_workers=2)

    def test_dataloader(self):
        return DataLoader(
            IMDBDataset(self.test_dataset, self.tokenizer, self.max_length),
            batch_size=self.batch_size, 
            num_workers=2)
