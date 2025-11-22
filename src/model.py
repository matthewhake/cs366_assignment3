"""
model.py
Description: Bi-LSTM sentiment classifier for IMDB reviews
uses BERT tokenizer's vocab size (passed in as vocab_size)
Trains its own nn.Embedding
Bidirectional LSTM
Final linear layer to 2 classes, negative/positive
built as a PyTorch LightningModule so it plugs into train.py easily
Authors: Matthew Hake, Ben Chidley, Garret Keyhani, Joshua Smith 
Date: 11/21/2025
"""

from typing import Any, Dict

import torch
from torch import nn
import pytorch_lightning as pl

# Structure of this class follows the recommended PyTorch Lightning pattern:
# 1. computations in __init__
# 2. training_step /validation_step /test_step
# 3. configure_optimizers for optimizer definition
# Source: https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html

class IMDBBiLSTM(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        lr: float,
    ) -> None:

        # We use a trainable nn.Embedding with the BERT tokenizer vocabulary
        # (vocab_size, pad_idx) and a bidirectional LSTM for sentence-level
        # example Bi-LSTM sentiment model (embedding + bidirectional LSTM): https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525

        # Hugging Face tokenizer docs (vocab_size, special tokens, pad token): https://huggingface.co/docs/transformers/main/main_classes/tokenizer
        super().__init__()
        self.save_hyperparameters()

        # Trainable embedding layer (no pretrained encoder)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Fully connected classifier: 2 * hidden_dim (forward + backward) ... 2 classes
        self.fc = nn.Linear(hidden_dim * 2, 2)

        # Cross entropy for 2-class classification
        self.criterion = nn.CrossEntropyLoss()

    # FORWARD
    # See typical PyTorch Bi-LSTM classification patterns: https://www.scaler.com/topics/pytorch/lstm-pytorch/

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        IN:
        input_ids: Tensor of token ids, shape [batch_size, seq_len].
        OUT:
        logits: Tensor of shape [batch_size, 2].
        """
        embedded = self.embedding(input_ids)

        lstm_out, (h_n, c_n) = self.lstm(embedded)

        forward_last = h_n[-2, :, :]  # [batch, hidden_dim]
        backward_last = h_n[-1, :, :]
        final_repr = torch.cat((forward_last, backward_last), dim=1)

        final_repr = self.dropout(final_repr)

        logits = self.fc(final_repr)

        return logits
    
    # Step by step LightningModule tutorial using self.log:
    # https://lightning.ai/pages/community/tutorial/step-by-step-walk-through-of-pytorch-lightning/

    def _step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        # Shared logic for train/val/test steps 
        input_ids = batch["input_ids"]
        labels = batch["label"] #[batch]

        logits = self(input_ids)
        loss = self.criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Lightning will handle sending these to W&B if you use WandbLogger
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    # LIGHTNING HOOKS
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._step(batch, stage="train")
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, stage="val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, stage="test")

    def configure_optimizers(self) -> Any:
        # Use Adam or AdamW as required by prof chen
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
