import torch
import numpy as np
import time

from alpamayo_r1.models.alpamayo_r1_compile import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper
from alpamayo_r1.models.patches import patch_for_torch_compile

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model = AlpamayoR1.from_pretrained("./Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

patch_for_torch_compile(model, mode="streaming", fuse_qkv=True, fuse_gate_up=True)

