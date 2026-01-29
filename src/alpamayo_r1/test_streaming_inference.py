# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import torch
import numpy as np

from alpamayo_r1.models.alpamayo_r1_streaming import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


# Example clip ID
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
print(f"Loading dataset for clip_id: {clip_id}...")
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)
print("Dataset loaded.")
messages = helper.create_message(data["image_frames"].flatten(0, 1))

model = AlpamayoR1.from_pretrained("./Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
print(inputs.input_ids.shape)
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}

model_inputs = helper.to_device(model_inputs, "cuda")
input_ids = model_inputs["tokenized_data"]["input_ids"].clone()

torch.cuda.manual_seed_all(42)
tokens_per_image = 720
feature_dim = 1536
with torch.autocast("cuda", dtype=torch.bfloat16):
    # for i in range(5):
    #     if i > 0:
    #         new_pixel_values = torch.randn(
    #             4 * tokens_per_image, 
    #             feature_dim, 
    #             dtype=model_inputs["tokenized_data"]["pixel_values"].dtype, 
    #             device=model_inputs["tokenized_data"]["pixel_values"].device
    #         )
    #         model_inputs["tokenized_data"]["pixel_values"] = new_pixel_values
    #         model_inputs["tokenized_data"]["input_ids"] = input_ids

    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_streaming_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
        max_generation_length=256,
        return_extra=True,
    )

    print("pred_xyz: ", pred_xyz.shape)
    print("pred_rot: ", pred_rot.shape)

# the size is [batch_size, num_traj_sets, num_traj_samples]
print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()
print("minADE:", min_ade, "meters")
print(
    "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
    "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
    "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
)


def create_sliding_window_input(num_windows, data):
    streaming_inputs = []
    for i in range(num_windows):
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        }
        model_inputs = helper.to_device(model_inputs, "cuda")
        streaming_inputs.append(model_inputs)
    return streaming_inputs


def run_streaming_inference(streaming_inputs):
    ego_history_xyz, ego_history_rot = None, None
    pred_xyzs, pred_rots = [], []
    for model_inputs in streaming_inputs:
        if ego_history_xyz is not None:
            model_inputs["ego_history_xyz"] = ego_history_xyz
            model_inputs["ego_history_rot"] = ego_history_rot
        
        # pred_xyz: [1, 1, 1, 64, 3]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_streaming_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1,  # Feel free to raise this for more output trajectories and CoC traces.
                max_generation_length=256,
                return_extra=True,
            )
        ego_history_xyz = pred_xyz[0][:, :, :16, :]
        ego_history_rot = pred_rot[0][:, :, :16, :, :]
        pred_xyzs.append(pred_xyz[0])
        pred_rots.append(pred_rot[0])
    return pred_xyzs, pred_rots


def test_streaming_inference():
    streaming_inputs = create_sliding_window_input(1, data)
    print(torch.cuda.memory_summary(device="cuda", abbreviated=True))
    pred_xyzs, pred_rots = run_streaming_inference(streaming_inputs)
    print("pred_xyzs: ", pred_xyzs.shape)
    print("pred_rots: ", pred_rots.shape)


# if __name__ == "__main__":
#     test_streaming_inference()