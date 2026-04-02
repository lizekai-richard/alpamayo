import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import alpamayo_r1
sys.modules["alpamayo1_5"] = alpamayo_r1

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "paroquant"))

import numpy as np
import torch

from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import load_paroquant_model_v1p5
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import convert_model_to_marlin_w4a8
from alpamayo_r1 import helper
from alpamayo_r1.utils.quantization.paroquant_marlin_w4a8 import save_paroquant_pretrained

model = load_paroquant_model_v1p5(
    model_path="/data/scratch/zekaili/Alpamayo1_5-Finetuned-new",
    paro_checkpoint="/data/scratch/zekaili/quant_cache/alpamayo-1.5-finetuned-new-paro-w4-vlm-mm.pt",
    mode="streaming",
)
n = convert_model_to_marlin_w4a8(model)
if n == 0:
    raise SystemExit("No layers converted!")

save_path = "/data/scratch/zekaili/Alpamayo1_5-finetuned-PARO"
save_paroquant_pretrained(model, save_path)