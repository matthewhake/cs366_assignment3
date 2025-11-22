# Assignment 3: IMDB Sentiment Classification using Bi-LSTM

A small PyTorch / PyTorch Lightning assignment repository containing model, training code, and experiment logs.

## What this repo contains

- `src/` - source code for the assignment
	- `train.py` - training entrypoint
	- `model.py` - model definition
	- `data.py` - data loading / preprocessing
- `lightning_logs/` - PyTorch Lightning TensorBoard logs (per-run)
- `wandb/` - Weights & Biases run exports and artifacts

## Quick start

Requirements

- Python 3.8+ recommended
- A virtual environment is strongly recommended

Install dependencies

If a top-level `requirements.txt` exists, install via:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If there is no top-level `requirements.txt`, a requirements file can be found in some `wandb` run exports (e.g. `wandb/*/files/requirements.txt`). You can also install common ML packages used by this project:

```bash
pip install torch torchvision pytorch-lightning wandb
```

Run training script

Use the training entrypoint in `src/train.py`. Typical usage:

```bash
# from project root
python src/train.py

# or run module-style
python -m src.train

# view available options
python src/train.py --help
```

Training outputs and logs

- TensorBoard-style logs are written to `lightning_logs/version_*`.
- W&B exports (if enabled) appear under `wandb/`.

Repository structure

```
.
├── README.md
├── src/
│   ├── data.py
│   ├── model.py
│   └── train.py
├── lightning_logs/
└── wandb/
```

Debug notes

- Run a quick syntax check across the `src/` package:

```bash
python -m py_compile src/*.py
```

Created for CS366 assignment 3.
