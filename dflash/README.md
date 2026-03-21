# DFlash Draft Model Training

Train the DFlash draft model from scratch using distillation data from Alpamayo-R1.

## Overview

```
dflash/
├── train_dflash.py          # Train draft model from scratch (DDP)
├── generate_data.py         # Generate distillation data from Alpamayo-R1
├── download_chunks.py       # Download PhysicalAI-AV dataset chunks
├── run_training.sh          # Launch training (4 GPUs)
├── run_generate_data.sh     # Launch data generation (8 GPUs)
└── model/
    ├── dflash.py             # DFlashDraftModel architecture
    └── utils.py              # Model utilities
```

## Step 1: Download dataset

Download chunks from the [PhysicalAI-AV dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles). Each chunk contains ~100 clips.

```bash
python dflash/download_chunks.py \
    --num-chunks 400 \
    --cache-dir ~/data/physicalai_av/hf_cache \
    --workers 4
```

## Step 2: Generate distillation data

Extract `(hidden_state, future_tokens)` pairs from Alpamayo-R1 for each clip. Runs on 8 GPUs in parallel.

```bash
# All 400 chunks (recommended: use the wrapper script)
./dflash/run_generate_data.sh 0 400 ~/data/dflash_train

# Single GPU (for testing)
python dflash/generate_data.py \
    --cache-dir ~/data/physicalai_av/hf_cache \
    --output-dir ~/data/dflash_train \
    --start-chunk 0 --end-chunk 50
```

**Arguments** (`run_generate_data.sh`):

| Argument | Default | Description |
|----------|---------|-------------|
| `start_chunk` | 0 | First chunk to process |
| `end_chunk` | 400 | Last chunk (exclusive) |
| `output_dir` | `/data/dflash_train_1` | Output directory |
| `target_layers` | `24,30,31,32,34` | Comma-separated layer IDs |
| `stride` | 1 | Sliding window stride |
| `block_sizes` | `8,16` | Comma-separated block sizes |
| `samples_per_clip` | 3 | Number of t0 samples per clip |

## Step 3: Train draft model

Train a DFlash draft model from scratch using the distillation data. The model uses Alpamayo's vocabulary (155,698 tokens) and loads `embed_tokens` and `lm_head` from Alpamayo.

```bash
# Recommended: use the wrapper script (4 GPUs)
./dflash/run_training.sh

# Or run directly
torchrun --nproc_per_node=4 dflash/train_dflash.py \
    --target-model ~/models/Alpamayo-R1-10B \
    --data-dir ~/data/dflash_train/b8 \
    --output-dir ~/exp/dflash
```

**Key hyperparameters** (edit in `run_training.sh`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EPOCHS` | 15 | Training epochs |
| `BATCH_SIZE` | 64 | Per-GPU batch size |
| `LEARNING_RATE` | 3e-4 | Scaled by world size |
| `BLOCK_SIZE` | 8 | Must match distillation data |
| `NUM_DRAFT_LAYERS` | 2 | Number of draft decoder layers |
| `TARGET_LAYERS` | `24,30,31,32,34` | Alpamayo layers to distill from |
| `PREFIX_WEIGHT_GAMMA` | 3 | Geometric decay for prefix-weighted CE |

## Pre-trained model

Skip training and download the pre-trained draft model directly:

```bash
hf download FlashDriveVLA/Alpamayo-DFlash --local-dir ~/models/dflash
```
