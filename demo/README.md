# Demo Video Generation

Generate demo videos comparing 6 inference methods on the same clip, showing the cumulative effect of each optimization.

## Overview

```
demo/
├── run_all_methods.py    # Run 6 methods on a single clip
├── generate_videos.py    # Generate videos from inference output
└── generate_frame.py     # Render a single debug frame (PNG)
```

## Methods

Each method builds on the previous:

| # | Method | Description |
|---|--------|-------------|
| 1 | Baseline | Vanilla `vlm.generate()`, no optimizations |
| 2 | + System optimizations | torch.compile + StaticCache + fused projections |
| 3 | + Streaming | KV cache reuse across frames |
| 4 | + DFlash | Speculative decoding (block-parallel) |
| 5 | + 4-step diffusion | Reduced diffusion steps (10 -> 4) |
| 6 | + AWQ INT4 | 4-bit weight quantization |

## Step 1: Run inference

```bash
# Run all 6 methods on default clip
python demo/run_all_methods.py

# Run only specific methods (e.g., baseline, DFlash, AWQ)
python demo/run_all_methods.py --only 1,4,6

# List available clips
python demo/run_all_methods.py --list-clips

# Run specific clip, limit frames
python demo/run_all_methods.py --clip-index 2 --num-frames 50
```

Output structure:
```
~/exp/demo/run_MMDD_HHMMSS/
├── config.json
├── 1_baseline/frame_XXXX.json
├── 2_sys_opt/frame_XXXX.json
├── 3_streaming/frame_XXXX.json
├── 4_dflash/frame_XXXX.json
├── 5_dflash_4step/frame_XXXX.json
├── 6_awq/frame_XXXX.json
└── summary.json
```

## Step 2: Debug frames

Render individual PNG frames first to verify the visual layout (trajectory overlay, timing panel, CoC text, metrics) before committing to a full video render.

```bash
# Single method, single frame
python demo/generate_frame.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --method 6_awq --frame 10
```

Iterate on the layout until it looks right, then proceed to video generation.

## Step 3: Generate videos

Each video shows camera image with trajectory overlay, timing breakdown, streaming CoC text, and metrics. Frame dwell time maps to pipeline latency — faster methods produce shorter videos.

```bash
# Generate all 6 videos
python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS

# Fast 480p preview
python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --resolution 480

# Only specific methods
python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --only 1,4,6

# Slower playback for analysis
python demo/generate_videos.py --log-dir ~/exp/demo/run_MMDD_HHMMSS --slowdown 2
```
