#!/usr/bin/env python3
"""Convert ParoQuant optimization output (.pt files) to a single AlpamayoR1 checkpoint.

Takes the per-layer .pt files produced by optimize_alpamayo.py and converts them
into a complete model checkpoint that can be loaded by load_paroquant_model().

Supports quantizing VLM layers, expert layers, or both.

Usage:
    # VLM only
    python paroquant/real_quant_alpamayo.py \\
        --model ~/models/Alpamayo-R1-10B \\
        --vlm-result-dir ~/models/paroquant_results/Alpamayo-R1-10B/vlm \\
        --output-path ~/models/quant_cache/alpamayo-paro-w4.pt

    # VLM + Expert
    python paroquant/real_quant_alpamayo.py \\
        --model ~/models/Alpamayo-R1-10B \\
        --vlm-result-dir ~/models/paroquant_results/Alpamayo-R1-10B/vlm \\
        --expert-result-dir ~/models/paroquant_results/Alpamayo-R1-10B/expert \\
        --output-path ~/models/quant_cache/alpamayo-paro-w4.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

# paroquant-ref/ contains the upstream ParoQuant library
sys.path.append(str(Path(__file__).resolve().parent.parent / "paroquant"))
# src/ contains the alpamayo_r1 package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from inference_engine.utils.checkpoint_utils import replace_linears_from_pt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert ParoQuant .pt results to AlpamayoR1 checkpoint"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to base AlpamayoR1 model")
    parser.add_argument("--vlm-result-dir", type=str, required=True,
                        help="Directory with VLM layer .pt files from optimize_alpamayo.py")
    parser.add_argument("--expert-result-dir", type=str, default=None,
                        help="Directory with expert layer .pt files (optional)")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Output .pt checkpoint path")
    args = parser.parse_args()

    from alpamayo_r1.models.alpamayo_r1p5 import Alpamayo1_5

    log.info(f"Loading base Alpamayo1_5 from {args.model}...")
    model = Alpamayo1_5.from_pretrained(args.model, dtype=torch.float16)

    # Replace VLM language model linears
    log.info(f"Replacing VLM linears from {args.vlm_result_dir}...")
    vlm_lm = model.vlm.model.language_model
    replace_linears_from_pt(
        vlm_lm,
        args.vlm_result_dir,
        prefix="layers.",
        ignore_suffix=("lm_head",),
    )

    # Optionally replace expert linears
    if args.expert_result_dir:
        log.info(f"Replacing expert linears from {args.expert_result_dir}...")
        replace_linears_from_pt(
            model.expert,
            args.expert_result_dir,
            prefix="layers.",
            ignore_suffix=("lm_head",),
        )

    # Save full model state dict as .pt
    log.info(f"Saving checkpoint to {args.output_path}...")
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.half()
    state_dict = model.state_dict()
    torch.save(state_dict, args.output_path)

    log.info(f"Saved checkpoint with {len(state_dict)} keys to {args.output_path}")


if __name__ == "__main__":
    main()
