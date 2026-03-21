# Evaluation Guide

Evaluation scripts for Alpamayo-R1 with different optimization configurations. All scripts evaluate on the `physicalai_av` dataset and report per-step timing, minADE, and aggregate statistics.

## Scripts

### Full clip evaluation (120 timesteps, 1.7s–13.6s at 10Hz)

| Script | Mode | Optimizations | Description |
|--------|------|---------------|-------------|
| `eval_baseline.py` | Non-streaming | None | Baseline using `vlm.generate()` from main branch |
| `eval_system_opt.py` | Non-streaming | torch.compile | Manual decode loop with static cache |
| `eval_streaming.py` | Streaming | torch.compile, KV reuse | KV cache reuse across frames, no DFlash |
| `eval_dflash_streaming.py` | Streaming | DFlash + torch.compile + KV reuse | Full streaming with speculative decoding |
| `eval_dflash_streaming_awq.py` | Streaming | DFlash + AWQ INT4 + KV reuse | AWQ-quantized streaming with DFlash |

### Single-frame quick test (1 timestep at t0=5.1s)

For fast iteration and debugging. Runs N clips at a single timestep.

| Script | Optimizations | Description |
|--------|---------------|-------------|
| `eval_dflash_clips.py` | DFlash (external) | External accelerator, configurable opts |
| `eval_dflash_clips_opt.py` | DFlash (integrated) | Integrated DFlash path |
| `eval_dflash_clips_awq.py` | DFlash + AWQ INT4 | AWQ-quantized model with DFlash |

## Usage

### Full clip evaluation

```bash
# Baseline (no optimizations)
python eval/eval_baseline.py

# torch.compile only (no DFlash, no streaming)
python eval/eval_system_opt.py

# Streaming only (no DFlash)
python eval/eval_streaming.py

# DFlash + streaming (BF16)
python eval/eval_dflash_streaming.py

# DFlash + AWQ + streaming (recommended)
python eval/eval_dflash_streaming_awq.py
```

### Single-frame quick test

```bash
# DFlash with external accelerator (configurable opts)
python eval/eval_dflash_clips.py
python eval/eval_dflash_clips.py --use-patch
python eval/eval_dflash_clips.py --use-patch --fuse-gate-up
python eval/eval_dflash_clips.py --use-compile

# DFlash integrated path
python eval/eval_dflash_clips_opt.py
python eval/eval_dflash_clips_opt.py --use-compile

# DFlash + AWQ INT4
python eval/eval_dflash_clips_awq.py
python eval/eval_dflash_clips_awq.py --use-compile
```

## Modes

### Non-streaming

Each timestep is independent — full prefill + decode per frame. Scripts ending in `_clips` use a single timestep (t0=5.1s) for quick iteration.

Multi-timestep scripts (`eval_baseline.py`, `eval_system_opt.py`) run 120 timesteps per clip (1.7s to 13.6s at 10Hz) but re-encode every frame from scratch.

### Streaming

KV cache is reused across frames within a clip. The first frame is a full prefill (4 cameras x 4 frames = 16 images), subsequent frames add only 4 new images incrementally. Runs 120 timesteps per clip (1.7s to 13.6s at 10Hz).

Streaming scripts: `eval_streaming.py`, `eval_dflash_streaming.py`, `eval_dflash_streaming_awq.py`.

## Options

### Common options (all scripts)

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `/data/scratch/zekaili/Alpamayo-R1-10B` | Path to Alpamayo-R1 model |
| `--clip-ids-file` | `/data/scratch/zekaili/physicalai_av/clip_ids.json` | JSON file with clip IDs |
| `--num-clips` | 100 | Number of clips to evaluate |
| `--diffusion-steps` | 10 | Diffusion inference steps |
| `--cache-dir` | `/data/scratch/zekaili/physicalai_av/hf_cache` | Dataset cache directory |
| `--output-dir` | `~/exp/eval_results` | Directory for result JSON files |

### Streaming / multi-timestep scripts

| Flag | Default | Description |
|------|---------|-------------|
| `--num-traj-samples` | 6 | K for minADE_K |
| `--max-tokens` | 128 | Max Chain-of-Causation tokens to generate |
| `--warmup-steps` | 3 | First N streaming steps per clip excluded from metrics |

### DFlash scripts

| Flag | Default | Description |
|------|---------|-------------|
| `--draft-model` | `/data/scratch/zekaili/Alpamayo-DFlash` | Path to DFlash draft model |
| `--max-tokens` | 64 (clips) / 128 (streaming) | Max CoC tokens to generate |
| `--warmup` | 1-3 | Warmup clips/steps excluded from metrics |
| `--use-compile` | off | Enable torch.compile (max-autotune) |

### DFlash external accelerator (`eval_dflash_clips.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--use-patch` | off | Enable StaticCache + patched modules |
| `--fuse-gate-up` | off | Enable fused gate-up projection |
| `--use-capture-ids` | off | Use capture layer IDs (no hooks) |
| `--use-compile` | off | Enable all above + torch.compile |

### AWQ scripts (`eval_dflash_clips_awq.py`, `eval_dflash_streaming_awq.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--awq-checkpoint` | `/data/scratch/zekaili/quant_cache/alpamayo-all-w4-g128-v2.pt` | AWQ quantized checkpoint |
| `--w-bit` | 4 | Quantization bit width |
| `--group-size` | 128 | AWQ group size |
| `--no-fuse-qkv` | off | Disable fused QKV projection |
| `--no-fuse-gate-up` | off | Disable fused gate-up projection |
| `--quantize-VLM-only` | off | Only quantize VLM, keep expert in BF16 |
| `--expert-group-size` | 64 | Group size for expert quantization |

## Metrics

| Metric | Description |
|--------|-------------|
| **minADE_K** | Minimum average displacement error across K trajectory samples (meters) |
| **minADE_1** | ADE of a single trajectory sample |
| **Acceptance rate** | Fraction of DFlash draft tokens accepted by verifier |
| **Mean acceptance length** | Average number of draft tokens accepted per iteration |
| **Tokens/sec** | Decode throughput |
| **Total time** | End-to-end time per timestep (encode + prefill + decode + action) |

### Timing breakdown

- **encode_time_ms** — ViT visual encoding
- **prefill_time_ms** — LLM prefill (KV cache fill)
- **decode_time_ms** — Autoregressive / DFlash speculative decode
- **dflash_loop_time_ms** — DFlash draft-verify loop only
- **action_time_ms** — Diffusion trajectory generation

## Output

Results are saved as JSON to `--output-dir` with the config, per-step results, and aggregate summary. File naming follows the pattern:

```
dflash_streaming_awq_4bit_fuse11_K6_d10_100clips.json
baseline_K6_d10_100clips.json
```
