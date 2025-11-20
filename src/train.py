"""
@ Description: This main file is used to train the model
@ Authors: Matthew Hake, Ben Chidley, Garret Keyhani, Josh Smith 
@ Create Time:
"""
SEED = 42
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import IMDBDataset
from model import IMDBBiLSTM

def main():
    dm = IMDBDataset(batch_size=32,max_length=256)
    dm.prepare_data()
    dm.setup()

    model = IMDBBiLSTM(
        vocab_size=dm.vocab_size,
        pad_idx=dm.pad_idx,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=1
    )
    checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="best-model"
    )

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint]
    )

    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()