"""
@ Description: This main file to preprocess the data
@ Authors: Matthew Hake, Ben Chidley, Garret Keyhani, Josh Smith 
@ Create Time: 
"""
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from datasets import load_dataset, concatinate_datasets
import pytorch_lightning as pl
import numpy as np

pl.seed_everything(42)

class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, max_length=256):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def download_and_split(self):
        #loading imdb dataset
        train = load_dataset("imdb", split="train")
        test  = load_dataset("imdb", split="test")
       
        #combining preset train and test (initially 50/50)
        full = concatinate_datasets([train,test]).shuffle(seed=42)
        
        #t/v/t split 70/15/15
        total = len(full)
        train_size = int(0.7 * total)
        val_size   = int(0.15 * total)
        test_size  = total - train_size - val_size
        splits = full_ds.train_test_split(
            train_size=train_size, 
            test_size=val_size + test_size,
            seed=42
            )
        self.train_dataset = splits['train']
        temp_dataset = splits['test']
        val_test_splits = temp_dataset.train_test_split(test_size=test_size, seed=42)
        self.val_dataset = val_test_splits['train']
        self.test_dataset = val_test_splits['test']
