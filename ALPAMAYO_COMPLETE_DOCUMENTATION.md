# Alpamayo-R1 完整技术文档

本文档整合了 Alpamayo-R1 模型的所有技术分析，包括架构、组件、推理流程、Attention 机制等详细内容。

## 📋 完整目录

### 第一部分：模型架构与推理流程
1. [整体架构概览](#1-整体架构概览)
2. [核心组件详解](#2-核心组件详解)
3. [推理流程详解](#3-推理流程详解)
4. [数据流与张量形状](#4-数据流与张量形状)

### 第二部分：Expert Model 详解
5. [什么是 Expert Model？](#5-什么是-expert-model)
6. [Expert Model 的架构](#6-expert-model-的架构)
7. [Expert Model 的作用](#7-expert-model-的作用)
8. [与 VLM 的关系](#8-与-vlm-的关系)

### 第三部分：Attention 机制分析
9. [Causal vs Non-Causal Attention](#9-causal-vs-non-causal-attention)
10. [VLM 的 Causal Attention](#10-vlm-的-causal-attention)
11. [Expert Model 的 Non-Causal Attention](#11-expert-model-的-non-causal-attention)

### 第四部分：Vision Token Attention
12. [图像数据组织](#12-图像数据组织)
13. [Vision Token 生成](#13-vision-token-生成)
14. [Vision Token Attention 规则](#14-vision-token-attention-规则)
15. [View 和 Frame 的顺序关系](#15-view-和-frame-的顺序关系)
16. [Vision Token Attention 证据分析](#16-vision-token-attention-证据分析)

### 第五部分：图像处理与 Vision Encoder
17. [图像处理流程详解](#17-图像处理流程详解)
18. [Vision Encoder 位置与调用](#18-vision-encoder-位置与调用)

---

# 第一部分：模型架构与推理流程

## 1. 整体架构概览

Alpamayo-R1 是一个 **Vision-Language-Action (VLA)** 模型，用于自动驾驶场景的轨迹预测和推理。模型采用**混合架构**，结合了视觉-语言模型（VLM）和专家模型（Expert Model）。

```
┌─────────────────────────────────────────────────────────────┐
│                    Alpamayo-R1 模型架构                      │
└─────────────────────────────────────────────────────────────┘

输入层
  ├── 多相机图像 (4个相机 × 4帧)
  ├── 历史轨迹 (ego_history_xyz, ego_history_rot)
  └── 时间戳信息

VLM Backbone (Qwen3-VL-8B-Instruct)
  ├── 视觉编码器 (处理多相机图像)
  ├── 语言模型 (生成 Chain-of-Causation 推理)
  └── 轨迹token融合 (fuse_traj_tokens)

Expert Model (基于 VLM 的文本配置)
  ├── Action Input Projection (action_in_proj)
  ├── Expert Transformer (处理动作序列)
  └── Action Output Projection (action_out_proj)

Diffusion Model
  └── Flow Matching / 去噪过程

Action Space
  └── 动作空间到轨迹的转换

输出层
  ├── 预测轨迹 (pred_xyz, pred_rot)
  └── Chain-of-Causation 文本 (CoC reasoning)
```

---

## 2. 核心组件详解

### 2.1 ReasoningVLA (基础类)

**位置**: `alpamayo_r1/models/base_model.py`

**功能**: 提供 VLA 模型的基础框架

**关键组件**:
- **VLM Backbone**: 基于 Qwen3-VL-8B-Instruct 的视觉-语言模型
  - 处理多模态输入（图像 + 文本）
  - 支持轨迹token的特殊处理
  - 使用 Flash Attention 2 优化

- **轨迹Token系统**:
  - `traj_tokenizer`: 未来轨迹的tokenizer
  - `hist_traj_tokenizer`: 历史轨迹的tokenizer
  - 特殊token: `<|traj_history_start|>`, `<|traj_future_start|>`, `<|cot_start|>` 等

- **TrajectoryFusionMixin**: 
  - `fuse_traj_tokens()`: 将历史轨迹编码为token并融合到输入序列中

### 2.2 AlpamayoR1 (专家模型)

**位置**: `alpamayo_r1/models/alpamayo_r1.py`

**继承**: `ReasoningVLA`

**新增组件**:

#### Expert Model
```python
# 基于 VLM 的文本配置创建专家模型
expert_config = copy.deepcopy(self.vlm.config.text_config)
self.expert = AutoModel.from_config(expert_config)
```
- 用于处理动作序列的 Transformer 模型
- 不包含 `embed_tokens`（使用 action_in_proj 的输出）

#### Action Space
```python
self.action_space: ActionSpace = hyu.instantiate(config.action_space_cfg)
```
- 定义动作空间（如加速度-曲率空间）
- 提供 `traj_to_action()` 和 `action_to_traj()` 转换方法

#### Diffusion Model
```python
self.diffusion: BaseDiffusion = hyu.instantiate(
    config.diffusion_cfg,
    x_dims=self.action_space.get_action_space_dims(),
)
```
- 用于在动作空间中进行去噪采样
- 支持 Flow Matching 等扩散方法

#### Action Projection Layers

**Action Input Projection** (`action_in_proj`):
- 将动作序列投影到专家模型的隐藏空间
- 使用 Fourier 编码处理时间步信息
- 输出: `(batch_size, num_waypoints, hidden_size)`

**Action Output Projection** (`action_out_proj`):
- 将专家模型的输出投影回动作空间
- 输出: `(batch_size, num_waypoints, action_dim)`

### 2.3 数据处理模块

#### 数据加载 (`load_physical_aiavdataset.py`)

**输入**:
- `clip_id`: 数据片段ID
- `t0_us`: 时间戳（微秒）

**输出字典**:
```python
{
    "image_frames": (N_cameras, num_frames, 3, H, W),  # 多相机图像
    "ego_history_xyz": (1, 1, num_history_steps, 3),   # 历史位置
    "ego_history_rot": (1, 1, num_history_steps, 3, 3), # 历史旋转
    "ego_future_xyz": (1, 1, num_future_steps, 3),     # 未来位置（ground truth）
    "ego_future_rot": (1, 1, num_future_steps, 3, 3),  # 未来旋转（ground truth）
    ...
}
```

**关键处理**:
- 坐标转换：将世界坐标系转换为 t0 时刻的局部坐标系
- 时间采样：历史轨迹 16 步（1.6秒@10Hz），未来轨迹 64 步（6.4秒@10Hz）
- 图像加载：4 个相机，每个相机 4 帧

#### 消息构建 (`helper.py`)

**`create_message()`** 函数构建多模态消息：

```python
messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a driving assistant..."}]
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": frame} for frame in frames  # 多相机图像
        ] + [
            {
                "type": "text",
                "text": "<|traj_history_start|>...<|traj_history_end|>output the chain-of-thought..."
            }
        ]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "<|cot_start|>"}]
    }
]
```

### 2.4 Processor

**`get_processor()`** 函数：
- 基于 Qwen3-VL-2B-Instruct 的 processor
- 设置图像像素范围：`min_pixels=163840`, `max_pixels=196608`
- 使用自定义 tokenizer（包含轨迹token）

---

## 3. 推理流程详解

基于 `test_inference.py` 的完整推理流程：

### 阶段 1: 数据准备

```python
# 1.1 加载数据集
clip_id = "030c760c-ae38-49aa-9ad8-f5650a545d26"
data = load_physical_aiavdataset(clip_id, t0_us=5_100_000)

# 1.2 构建消息
messages = helper.create_message(data["image_frames"].flatten(0, 1))
# messages 包含: system prompt + 多相机图像 + 用户指令
```

**数据形状**:
- `image_frames`: `(N_cameras, num_frames, 3, H, W)` → flatten 为 `(N_cameras*num_frames, 3, H, W)`
- `ego_history_xyz`: `(1, 1, 16, 3)`
- `ego_history_rot`: `(1, 1, 16, 3, 3)`

### 阶段 2: 模型初始化

```python
# 2.1 加载预训练模型
model = AlpamayoR1.from_pretrained(
    "nvidia/Alpamayo-R1-10B",
    dtype=torch.bfloat16
).to("cuda")

# 2.2 获取 processor
processor = helper.get_processor(model.tokenizer)
```

**模型组件**:
- `model.vlm`: VLM backbone (Qwen3-VL)
- `model.expert`: Expert model
- `model.diffusion`: Diffusion model
- `model.action_space`: Action space
- `model.action_in_proj`: 输入投影层
- `model.action_out_proj`: 输出投影层

### 阶段 3: 输入处理

```python
# 3.1 应用聊天模板并tokenize
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

# 3.2 融合历史轨迹token
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": data["ego_history_xyz"],
    "ego_history_rot": data["ego_history_rot"],
}
model_inputs = helper.to_device(model_inputs, "cuda")
```

**关键步骤**:
- `apply_chat_template()`: 将消息转换为模型输入格式
- `fuse_traj_tokens()`: 将历史轨迹编码为离散token并替换占位符

### 阶段 4: VLM 自回归生成 (Chain-of-Causation)

**位置**: `sample_trajectories_from_data_with_vlm_rollout()` 方法

```python
# 4.1 融合轨迹token到输入
input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)

# 4.2 配置生成参数
generation_config.top_p = 0.98
generation_config.temperature = 0.6
generation_config.num_return_sequences = num_traj_samples
generation_config.max_new_tokens = max_generation_length

# 4.3 使用 ExpertLogitsProcessor 屏蔽轨迹token
logits_processor = LogitsProcessorList([
    ExpertLogitsProcessor(
        traj_token_offset=self.config.traj_token_start_idx,
        traj_vocab_size=self.config.traj_vocab_size,
    )
])

# 4.4 生成 CoC 推理文本
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,  # 在 <traj_future_start> 后停止
    logits_processor=logits_processor,
    **tokenized_data,
)
```

**关键机制**:
- **ExpertLogitsProcessor**: 在生成 CoC 时屏蔽离散轨迹token，确保只生成文本推理
- **停止条件**: 在遇到 `<|traj_future_start|>` token 后停止生成
- **KV Cache**: 保存 prompt 的 key-value cache 供后续使用

**输出**:
- `vlm_outputs.sequences`: 生成的完整序列（包含 CoC 推理）
- `vlm_outputs.past_key_values`: KV cache（用于后续专家模型）

### 阶段 5: Diffusion 采样轨迹

#### 5.1 准备专家模型输入

```python
# 5.1.1 找到 <traj_future_start> 位置
traj_future_start_positions = (vlm_outputs.sequences == eos_token_id).int().argmax(dim=1)

# 5.1.2 设置位置ID和注意力掩码
position_ids = torch.arange(n_diffusion_tokens, device=device)
position_ids = einops.repeat(position_ids, "l -> 3 b l", b=b_star)
delta = vlm_outputs.rope_deltas + offset[:, None]
position_ids += delta.to(position_ids.device)

# 5.1.3 构建注意力掩码（屏蔽 padding）
attention_mask = torch.zeros(
    (b_star, 1, n_diffusion_tokens, prompt_cache.get_seq_length() + n_diffusion_tokens),
    ...
)
```

#### 5.2 定义去噪步骤函数

```python
def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x: (B*, *action_dim) - 噪声动作
    # t: 时间步
    
    # 5.2.1 将噪声动作投影到专家模型的token嵌入
    future_token_embeds = self.action_in_proj(x, t)
    # 形状: (b*, n_diffusion_tokens, hidden_size)
    
    # 5.2.2 运行专家模型（使用 KV cache）
    expert_out_base = self.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=prompt_cache,  # 使用 VLM 的 KV cache
        attention_mask=attention_mask,
        use_cache=True,
    )
    
    # 5.2.3 裁剪 KV cache（移除新添加的token）
    prompt_cache.crop(prefill_seq_len)
    
    # 5.2.4 投影回动作空间
    last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
    pred = self.action_out_proj(last_hidden)
    # 形状: (b*, Tf, C_action) - 预测的噪声/向量场
    
    return pred
```

**关键设计**:
- **KV Cache 复用**: 专家模型复用 VLM 生成的 KV cache，实现高效的上下文传递
- **非因果注意力**: 如果配置了 `expert_non_causal_attention=True`，专家模型可以使用双向注意力

#### 5.3 执行 Diffusion 采样

```python
# 5.3.1 采样动作
sampled_action = self.diffusion.sample(
    batch_size=total_batch,  # B * num_traj_samples * num_traj_sets
    step_fn=step_fn,
    device=device,
    return_all_steps=False,
    **diffusion_kwargs,
)

# 5.3.2 将动作转换为轨迹
pred_xyz, pred_rot = self.action_space.action_to_traj(
    sampled_action,
    hist_xyz_rep,  # 重复的历史轨迹
    hist_rot_rep,
)
```

**Diffusion 过程**:
- 从噪声开始，通过多步去噪生成动作序列
- 每一步调用 `step_fn` 进行去噪
- 最终得到干净的动作序列

### 阶段 6: 后处理与输出

```python
# 6.1 重塑输出形状
pred_xyz = einops.rearrange(
    pred_xyz, "(b ns nj) ... -> b ns nj ...",
    ns=num_traj_sets, nj=num_traj_samples
)
# 最终形状: (B, num_traj_sets, num_traj_samples, T, 3)

# 6.2 提取 CoC 文本（如果请求）
if kwargs.get("return_extra", False):
    extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
    # extra["cot"] 包含每个轨迹的 Chain-of-Causation 推理文本
```

### 阶段 7: 评估指标计算

```python
# 7.1 计算 minADE
gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
min_ade = diff.min()  # 最小平均位移误差
```

---

## 4. 数据流与张量形状

### 完整数据流

```
输入数据
  ├── image_frames: (4, 4, 3, H, W)
  ├── ego_history_xyz: (1, 1, 16, 3)
  └── ego_history_rot: (1, 1, 16, 3, 3)
         ↓
消息构建 (create_message)
  └── messages: List[Dict] (多模态消息)
         ↓
Processor (apply_chat_template)
  └── tokenized_data: Dict
      ├── input_ids: (1, L_prompt)
      └── attention_mask: (1, L_prompt)
         ↓
轨迹token融合 (fuse_traj_tokens)
  └── input_ids: (1, L_prompt) [历史轨迹token已替换]
         ↓
VLM 生成 (generate)
  ├── vlm_outputs.sequences: (num_traj_samples, L_total)
  ├── vlm_outputs.past_key_values: KV Cache
  └── vlm_outputs.rope_deltas: RoPE 偏移
         ↓
Diffusion 采样准备
  ├── position_ids: (3, num_traj_samples, n_diffusion_tokens)
  └── attention_mask: (num_traj_samples, 1, n_diffusion_tokens, L_total)
         ↓
Diffusion 采样循环 (step_fn)
  ├── x (噪声动作): (B*, n_diffusion_tokens, action_dim)
  ├── action_in_proj → future_token_embeds: (B*, n_diffusion_tokens, hidden_size)
  ├── expert → last_hidden: (B*, n_diffusion_tokens, hidden_size)
  └── action_out_proj → pred: (B*, n_diffusion_tokens, action_dim)
         ↓
动作到轨迹转换 (action_to_traj)
  ├── pred_xyz: (B*, n_diffusion_tokens, 3)
  └── pred_rot: (B*, n_diffusion_tokens, 3, 3)
         ↓
输出重塑
  ├── pred_xyz: (1, num_traj_sets, num_traj_samples, 64, 3)
  └── pred_rot: (1, num_traj_sets, num_traj_samples, 64, 3, 3)
```

### 关键张量形状总结

| 阶段 | 张量名称 | 形状 | 说明 |
|------|---------|------|------|
| 输入 | `image_frames` | `(4, 4, 3, H, W)` | 4个相机，每相机4帧 |
| 输入 | `ego_history_xyz` | `(1, 1, 16, 3)` | 历史16步位置 |
| VLM输入 | `input_ids` | `(1, L_prompt)` | Tokenized输入序列 |
| VLM输出 | `vlm_outputs.sequences` | `(num_traj_samples, L_total)` | 生成的完整序列 |
| Diffusion | `x` (噪声动作) | `(B*, n_diffusion_tokens, action_dim)` | 动作空间中的噪声 |
| Expert输入 | `future_token_embeds` | `(B*, n_diffusion_tokens, hidden_size)` | 投影后的token嵌入 |
| Expert输出 | `last_hidden` | `(B*, n_diffusion_tokens, hidden_size)` | 专家模型隐藏状态 |
| 最终输出 | `pred_xyz` | `(1, num_traj_sets, num_traj_samples, 64, 3)` | 预测轨迹位置 |
| 最终输出 | `pred_rot` | `(1, num_traj_sets, num_traj_samples, 64, 3, 3)` | 预测轨迹旋转 |

**符号说明**:
- `B*`: 批次大小 × 轨迹样本数
- `n_diffusion_tokens`: 扩散token数量（通常等于动作空间的时间步数，如64）
- `L_prompt`: Prompt长度
- `L_total`: 总序列长度（prompt + 生成部分）

---

# 第二部分：Expert Model 详解

## 5. 什么是 Expert Model？

### 基本定义

**Expert Model** 是 Alpamayo-R1 中的一个**专门的 Transformer 模型**，用于处理**动作序列**（action sequences）。

### 在代码中的位置

```python
# alpamayo_r1.py 第 73-74 行
class AlpamayoR1(ReasoningVLA):
    """Expert model for reasoning VLA."""
```

### 核心特点

- **专门处理动作**：不同于 VLM 处理文本和图像，Expert Model 专门处理动作序列
- **基于 VLM 架构**：使用与 VLM 相同的文本模型架构（text_config）
- **非因果 Attention**：使用 non-causal attention，允许所有时间步之间互相 attention
- **与 Diffusion 配合**：在 diffusion 采样过程中，用于预测去噪方向

---

## 6. Expert Model 的架构

### 创建过程

```python
# alpamayo_r1.py 第 87-94 行
# 1. 复制 VLM 的文本配置
expert_config = copy.deepcopy(self.vlm.config.text_config)

# 2. 应用自定义配置（如果有）
if config.expert_cfg is not None:
    for key, value in config.expert_cfg.items():
        setattr(expert_config, key, value)

# 3. 创建 Expert Model
self.expert = AutoModel.from_config(expert_config)

# 4. 删除 embed_tokens（因为使用 action_in_proj 的输出）
del self.expert.embed_tokens
```

### 关键组件

Expert Model 与以下组件配合工作：

```python
# 输入投影：将动作序列投影到 Expert Model 的隐藏空间
self.action_in_proj = hyu.instantiate(
    config.action_in_proj_cfg,
    in_dims=self.action_space.get_action_space_dims(),
    out_dim=expert_config.hidden_size,  # 输出维度匹配 Expert Model
)

# Expert Model 本身
self.expert = AutoModel.from_config(expert_config)

# 输出投影：将 Expert Model 的输出投影回动作空间
self.action_out_proj = hyu.instantiate(
    config.action_out_proj_cfg,
    in_features=expert_config.hidden_size,
    out_features=self.action_space.get_action_space_dims()[-1],
)
```

### 架构示意图

```
输入：噪声动作序列 (x, t)
    ↓
Action Input Projection (action_in_proj)
    ↓
[动作序列 embeddings] (future_token_embeds)
    ↓
Expert Model (Transformer)
    ├── 输入：future_token_embeds
    ├── KV Cache：prompt_cache (来自 VLM)
    ├── Attention：Non-Causal (双向)
    └── 输出：hidden states
    ↓
Action Output Projection (action_out_proj)
    ↓
输出：预测的噪声/向量场 (pred)
```

---

## 7. Expert Model 的作用

### 在 Diffusion 采样中的作用

Expert Model 是 **diffusion 采样过程中的去噪函数**（denoising function）：

```python
# alpamayo_r1.py 第 255-284 行
def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Diffusion 的去噪步骤函数"""
    # x: 噪声动作序列 (B*, 64, action_dim)
    # t: 时间步
    
    # 1. 将动作序列投影为 token embeddings
    future_token_embeds = self.action_in_proj(x, t)
    # 形状: (B*, 64, hidden_size)
    
    # 2. Expert Model 处理
    expert_out_base = self.expert(
        inputs_embeds=future_token_embeds,
        past_key_values=prompt_cache,  # 复用 VLM 的 KV cache
        attention_mask=attention_mask,
        is_causal=False,  # 非因果 attention
    )
    
    # 3. 提取 hidden states
    last_hidden = expert_out_base.last_hidden_state[:, -n_diffusion_tokens:]
    
    # 4. 投影回动作空间
    pred = self.action_out_proj(last_hidden)
    # 形状: (B*, 64, action_dim) - 预测的噪声/向量场
    
    return pred
```

### 关键功能

#### 功能 1: 上下文理解
- **输入**：动作序列的 embeddings + VLM 的 KV cache
- **作用**：理解视觉上下文（图像）、历史轨迹、CoC 推理文本
- **输出**：基于上下文理解的动作预测

#### 功能 2: 时序建模
- **输入**：所有时间步的动作 embeddings（并行）
- **作用**：使用 non-causal attention 建立时间步之间的关系
- **输出**：全局一致的动作序列

#### 功能 3: 去噪预测
- **输入**：噪声动作序列 + 时间步
- **作用**：预测去噪方向（噪声或向量场）
- **输出**：用于 diffusion 下一步的预测

### 与 VLM 的分工

| 组件 | 作用 | 输入 | 输出 |
|------|------|------|------|
| **VLM** | 生成 CoC 推理文本 | 图像、历史轨迹、用户指令 | CoC 文本 tokens + KV cache |
| **Expert Model** | 处理动作序列 | 动作 embeddings + KV cache | 动作预测（去噪方向） |

**关键区别**：
- **VLM**：处理**文本生成**（自回归，causal attention）
- **Expert Model**：处理**动作序列**（并行，non-causal attention）

---

## 8. 与 VLM 的关系

### 架构相似性

```python
# Expert Model 使用与 VLM 相同的文本模型架构
expert_config = copy.deepcopy(self.vlm.config.text_config)
self.expert = AutoModel.from_config(expert_config)
```

**相同点**：
- 相同的 Transformer 架构（层数、隐藏维度、注意力头数等）
- 相同的参数结构

**不同点**：
- **VLM**：包含视觉编码器 + 文本模型
- **Expert Model**：只有文本模型部分（没有视觉编码器）
- **Expert Model**：没有 `embed_tokens`（使用 `action_in_proj` 的输出）

### KV Cache 复用

**关键设计**：Expert Model **复用 VLM 生成的 KV cache**

```python
# VLM 生成 CoC 后
prompt_cache = vlm_outputs.past_key_values  # 包含所有 prompt 的 KV

# Expert Model 使用这个 cache
expert_out_base = self.expert(
    inputs_embeds=future_token_embeds,
    past_key_values=prompt_cache,  # ⭐ 复用 VLM 的 KV cache
    ...
)
```

**好处**：
- **效率**：避免重复计算 prompt 的 KV cache
- **一致性**：Expert Model 和 VLM 看到相同的上下文
- **内存**：共享 KV cache，节省内存

### 上下文传递

Expert Model 通过 KV cache 可以访问：
- **图像 tokens**（通过 VLM 的视觉编码器）
- **历史轨迹 tokens**
- **CoC 推理文本 tokens**
- **用户指令 tokens**

这使得 Expert Model 能够基于**完整的上下文**来预测动作。

---

# 第三部分：Attention 机制分析

## 9. Causal vs Non-Causal Attention

在 Alpamayo 中，**不同的模型组件使用不同的 attention 类型**：

| 模型组件 | Attention 类型 | 原因 |
|---------|---------------|------|
| **VLM (生成 CoC)** | **Causal (因果)** | 自回归生成，需要因果掩码 |
| **Expert Model (处理动作)** | **Non-Causal (非因果)** | 并行处理所有未来时间步 |

---

## 10. VLM 的 Causal Attention

### 代码位置

```python
# alpamayo_r1.py 第 192-198 行
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,
    logits_processor=logits_processor,
    **tokenized_data,
)
```

### 为什么使用 Causal Attention？

**VLM 生成 Chain-of-Causation (CoC) 文本时**：
- 使用标准的**自回归生成**（autoregressive generation）
- 每个 token 只能看到**之前的 tokens**
- 这是 LLM 的标准行为

**Causal Mask 示例**：
```
Token 0:  [1, 0, 0, 0, 0]  ← 只能看到自己
Token 1:  [1, 1, 0, 0, 0]  ← 可以看到 Token 0 和自己
Token 2:  [1, 1, 1, 0, 0]  ← 可以看到 Token 0, 1 和自己
Token 3:  [1, 1, 1, 1, 0]  ← 可以看到 Token 0, 1, 2 和自己
Token 4:  [1, 1, 1, 1, 1]  ← 可以看到所有之前的 tokens
```

**原因**：
- 生成过程是**顺序的**：先生成 Token 0，再生成 Token 1，...
- 必须使用 causal mask 才能保证生成的一致性
- 这是所有自回归语言模型的标准做法

### VLM 的 Attention 范围

在生成 CoC 时：
- **Prompt 部分**（图像、历史轨迹、用户指令）：所有 tokens 可见
- **生成部分**（CoC 文本）：使用 causal mask，只能看到之前的 tokens

---

## 11. Expert Model 的 Non-Causal Attention

### 代码位置

```python
# config.py 第 36 行
expert_non_causal_attention: bool = True,  # 默认值

# alpamayo_r1.py 第 250-252 行
forward_kwargs = {}
if self.config.expert_non_causal_attention:
    forward_kwargs["is_causal"] = False  # ⭐ 关键：设置为非因果

# alpamayo_r1.py 第 269-276 行
expert_out_base = self.expert(
    inputs_embeds=future_token_embeds,  # 所有未来时间步的 token embeddings
    position_ids=position_ids,
    past_key_values=prompt_cache,  # 包含 prompt 和 CoC 的 KV cache
    attention_mask=attention_mask,
    use_cache=True,
    **forward_kwargs,  # is_causal=False
)
```

### 为什么使用 Non-Causal Attention？

**Expert Model 处理动作序列时**：
- 输入是**所有未来时间步的 token embeddings**（例如 64 个时间步）
- 这些 embeddings 是**并行输入**的（不是顺序生成）
- 使用 **non-causal attention** 允许所有时间步之间互相 attention

**Non-Causal Mask 示例**（假设 5 个时间步）：
```
Time 0:  [1, 1, 1, 1, 1]  ← 可以看到所有时间步
Time 1:  [1, 1, 1, 1, 1]  ← 可以看到所有时间步
Time 2:  [1, 1, 1, 1, 1]  ← 可以看到所有时间步
Time 3:  [1, 1, 1, 1, 1]  ← 可以看到所有时间步
Time 4:  [1, 1, 1, 1, 1]  ← 可以看到所有时间步
```

**关键区别**：
- **Causal**: 时间步 i 只能看到时间步 0 到 i
- **Non-Causal**: 时间步 i 可以看到所有时间步（0 到 T-1）

### 为什么 Expert Model 使用 Non-Causal？

#### 1. 并行处理需求

**Diffusion 采样过程**：
```python
# 在 diffusion 的每一步，所有时间步的动作都是并行处理的
def step_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # x: (B*, 64, action_dim) - 所有 64 个时间步的噪声动作
    future_token_embeds = self.action_in_proj(x, t)  # 并行投影所有时间步
    expert_out = self.expert(
        inputs_embeds=future_token_embeds,  # 并行输入所有时间步
        ...
    )
    return pred  # 并行输出所有时间步的预测
```

**关键点**：
- 所有时间步的 embeddings 是**同时输入**的
- 不是顺序生成，而是**并行处理**
- 因此可以使用 non-causal attention

#### 2. 时序依赖建模

**Non-Causal Attention 的优势**：
- **双向信息流**：每个时间步可以看到过去和未来的信息
- **全局一致性**：所有时间步可以协调一致，避免局部不一致
- **更好的轨迹平滑性**：未来时间步可以影响过去时间步的预测

#### 3. 与 Diffusion 的配合

**Diffusion 模型的特点**：
- 在去噪过程中，所有时间步是**同时优化**的
- 需要全局一致性，而不是局部因果性
- Non-causal attention 更适合这种并行优化过程

---

# 第四部分：Vision Token Attention

## 12. 图像数据组织

### 原始数据结构

从 `load_physical_aiavdataset.py` 中可以看到：

```python
# 默认加载 4 个相机，每个相机 4 帧
camera_features = [
    CAMERA_CROSS_LEFT_120FOV,      # 相机 0
    CAMERA_FRONT_WIDE_120FOV,      # 相机 1
    CAMERA_CROSS_RIGHT_120FOV,     # 相机 2
    CAMERA_FRONT_TELE_30FOV,       # 相机 3
]

# 图像形状
image_frames: (N_cameras=4, num_frames=4, 3, H, W)
```

### 时间帧顺序

```python
# 第 161-165 行：图像时间戳
# 如果 num_frames=4，加载时间点：[t0-0.3s, t0-0.2s, t0-0.1s, t0]
image_timestamps = np.array(
    [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) 
     for i in range(num_frames)],
    dtype=np.int64,
)
```

**时间顺序**（从早到晚）：
- Frame 0: t0 - 0.3s
- Frame 1: t0 - 0.2s
- Frame 2: t0 - 0.1s
- Frame 3: t0 (当前时刻)

### 图像展平顺序

在 `test_inference.py` 第 33 行：

```python
messages = helper.create_message(data["image_frames"].flatten(0, 1))
```

**`flatten(0, 1)` 的效果**：
- 输入：`(4, 4, 3, H, W)` - (N_cameras, num_frames, C, H, W)
- 输出：`(16, 3, H, W)` - (N_cameras * num_frames, C, H, W)

**展平后的顺序**（**先按相机，再按帧**）：

```
Index 0:  Camera 0, Frame 0 (t0-0.3s) - cross_left, 最早
Index 1:  Camera 0, Frame 1 (t0-0.2s) - cross_left
Index 2:  Camera 0, Frame 2 (t0-0.1s) - cross_left
Index 3:  Camera 0, Frame 3 (t0)      - cross_left, 当前
Index 4:  Camera 1, Frame 0 (t0-0.3s) - front_wide, 最早
Index 5:  Camera 1, Frame 1 (t0-0.2s) - front_wide
Index 6:  Camera 1, Frame 2 (t0-0.1s) - front_wide
Index 7:  Camera 1, Frame 3 (t0)      - front_wide, 当前
Index 8:  Camera 2, Frame 0 (t0-0.3s) - cross_right, 最早
Index 9:  Camera 2, Frame 1 (t0-0.2s) - cross_right
Index 10: Camera 2, Frame 2 (t0-0.1s) - cross_right
Index 11: Camera 2, Frame 3 (t0)      - cross_right, 当前
Index 12: Camera 6, Frame 0 (t0-0.3s) - front_tele, 最早
Index 13: Camera 6, Frame 1 (t0-0.2s) - front_tele
Index 14: Camera 6, Frame 2 (t0-0.1s) - front_tele
Index 15: Camera 6, Frame 3 (t0)      - front_tele, 当前
```

**关键特点**：
- **View-first ordering**：先按相机（view）分组，再按时间（frame）排序
- 每个相机的 4 帧按时间顺序排列（从早到晚）
- 所有相机的同一时间帧**不连续**（例如，所有 Frame 0 分散在不同位置）

---

## 13. Vision Token 生成

### Message 构建

在 `helper.py` 的 `create_message()` 函数中：

```python
def create_message(frames: torch.Tensor):
    """frames: (16, 3, H, W) after flatten(0, 1)"""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant..."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame} for frame in frames  # 16 张图像
            ] + [
                {"type": "text", "text": "<|traj_history_start|>...<|traj_history_end|>..."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}]
        },
    ]
```

### Processor 处理

在 `test_inference.py` 中：

```python
processor = helper.get_processor(model.tokenizer)  # Qwen3-VL processor

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)
```

**Qwen3-VL Processor 的处理流程**：
1. 对每张图像进行预处理（resize, normalize）
2. 通过 Vision Encoder 将图像编码为 vision tokens
3. 将 vision tokens 插入到文本 token 序列中
4. 构建 attention mask

---

## 14. Vision Token Attention 规则

### 关键理解：Prompt vs Generation

在 VLM 生成时，序列分为两部分：

```
[Prompt Tokens] [Generation Tokens]
     ↑                ↑
  全连接          Causal Mask
```

- **Prompt Tokens**：包括 vision tokens、历史轨迹 tokens、用户指令 tokens
- **Generation Tokens**：CoC 文本 tokens（自回归生成）

### Vision Token 的 Attention

**Vision tokens 都在 Prompt 中**，因此：

#### 规则 1: Vision Tokens 之间全连接

```
Vision Token 0 (Camera 0, Frame 0)  ←→  Vision Token 1 (Camera 0, Frame 1)
Vision Token 0 (Camera 0, Frame 0)  ←→  Vision Token 4 (Camera 1, Frame 0)
Vision Token 0 (Camera 0, Frame 0)  ←→  Vision Token 15 (Camera 6, Frame 3)
... (所有 vision tokens 之间都可以 attention)
```

**原因**：
- Vision tokens 都在 prompt 中
- Prompt 中的 tokens 之间**不受 causal mask 限制**
- 它们可以**全连接 attention**

#### 规则 2: Vision Tokens 与生成 Tokens

```
Vision Token i  →  Generation Token j:  ✅ 可以（如果 j > i 的位置）
Generation Token j  →  Vision Token i:  ✅ 可以（vision token 在 prompt 中）
```

**原因**：
- Vision tokens 在 prompt 中，始终可见
- 生成的 tokens 可以 attention 到所有 prompt tokens（包括 vision tokens）

### 不是 View-Wise Causal

**重要澄清**：Vision tokens **不是 view-wise causal**，而是：

1. **Vision tokens 之间全连接**（都在 prompt 中）
2. **生成 tokens 对 vision tokens 全连接**（vision tokens 在 prompt 中）
3. **只有生成 tokens 之间是 causal**（自回归生成）

### 完整的 Attention Mask 结构

假设：
- Vision tokens: 16 张图像 × 256 tokens/图像 = 4096 tokens
- 历史轨迹 tokens: 48 tokens
- 用户指令 tokens: 20 tokens
- 生成 tokens: 100 tokens（CoC 文本）

**总序列长度**：4096 + 48 + 20 + 100 = 4264 tokens

**Attention Mask 矩阵**：

**形状**：`(1, 1, 4264, 4264)` - (batch, heads, seq_len, seq_len)

**结构示意**（简化，只显示关键部分）：

```
                Vision    Traj    Text    Gen0  Gen1  Gen2  ...
                [V0...V15] [T0...] [U0...] [G0] [G1] [G2]  ...
Vision V0       [1...1]    [1...1] [1...1] [1]  [1]  [1]   ...  ← 全连接
Vision V1       [1...1]    [1...1] [1...1] [1]  [1]  [1]   ...  ← 全连接
...
Vision V15      [1...1]    [1...1] [1...1] [1]  [1]  [1]   ...  ← 全连接
Traj T0         [1...1]    [1...1] [1...1] [1]  [1]  [1]   ...  ← 全连接
...
Text U0         [1...1]    [1...1] [1...1] [1]  [1]  [1]   ...  ← 全连接
...
Gen G0          [1...1]    [1...1] [1...1] [1]  [0]  [0]   ...  ← Causal
Gen G1          [1...1]    [1...1] [1...1] [1]  [1]  [0]   ...  ← Causal
Gen G2          [1...1]    [1...1] [1...1] [1]  [1]  [1]   ...  ← Causal
...
```

**说明**：
- `1` 表示可以 attention（mask = 0，不屏蔽）
- `0` 表示不能 attention（mask = -inf，屏蔽）

**关键观察**：
1. **Vision tokens 行**：所有位置都是 `1`（全连接）
2. **Prompt tokens 行**：所有位置都是 `1`（全连接）
3. **Generation tokens 行**：
   - 对 prompt tokens：都是 `1`（可以看到所有 prompt）
   - 对 generation tokens：causal mask（只能看到之前的生成 tokens）

---

## 15. View 和 Frame 的顺序关系

### 当前实现：View-First Ordering

**顺序**：先按 View（相机），再按 Frame（时间）

```
[View0_Frame0, View0_Frame1, View0_Frame2, View0_Frame3,
 View1_Frame0, View1_Frame1, View1_Frame2, View1_Frame3,
 View2_Frame0, View2_Frame1, View2_Frame2, View2_Frame3,
 View3_Frame0, View3_Frame1, View3_Frame2, View3_Frame3]
```

**优点**：
- 每个相机的帧连续，便于建立时序关系
- 实现简单（直接 flatten）

**缺点**：
- 同一时刻的不同视角**不连续**
- 模型需要跨越较远距离才能关联同一时刻的多视角信息

### 替代方案：Frame-First Ordering

如果使用 Frame-First 顺序：

```
[View0_Frame0, View1_Frame0, View2_Frame0, View3_Frame0,  # 时刻 t0-0.3s 的所有视角
 View0_Frame1, View1_Frame1, View2_Frame1, View3_Frame1,  # 时刻 t0-0.2s 的所有视角
 View0_Frame2, View1_Frame2, View2_Frame2, View3_Frame2,  # 时刻 t0-0.1s 的所有视角
 View0_Frame3, View1_Frame3, View2_Frame3, View3_Frame3]  # 时刻 t0 的所有视角
```

**优点**：
- 同一时刻的多视角信息连续，便于融合
- 更符合"多视角同时观察"的物理直觉

**缺点**：
- 同一相机的时序信息不连续
- 需要重新组织数据

### 为什么选择 View-First？

1. **实现简单**：直接使用 `flatten(0, 1)` 即可
2. **Attention 机制补偿**：虽然同一时刻的视角不连续，但 attention 机制可以跨越距离建立关联
3. **训练数据一致性**：训练时可能就使用这种顺序，保持一致性

---

## 16. Vision Token Attention 证据分析

### 代码证据

#### 1. Processor 返回的 attention_mask

**代码位置**：`test_inference.py:38-45`

```python
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
# inputs 包含: ['input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw']
```

**实际内容**：
```python
# 实际测试结果
attention_mask shape: torch.Size([1, 75])
attention_mask sample: tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
```

**说明**：
- 这个 `attention_mask` 是 **1D mask**，用于标记有效 token vs padding
- **不是用来控制 causal attention 的**
- 它只是告诉模型哪些位置是有效的（1）和哪些是 padding（0）

#### 2. VLM Generate 调用

**代码位置**：`alpamayo_r1.py:192-198`

```python
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,
    logits_processor=logits_processor,
    **tokenized_data,  # 包含 attention_mask, pixel_values, image_grid_thw
)
```

**关键观察**：
- **没有显式设置 `is_causal=False`**
- **没有显式设置特殊的 attention mask**
- 使用的是 `vlm.generate()` 的**默认行为**

#### 3. 默认行为推断

**标准 Transformer 生成行为**：
- `generate()` 方法内部会自动处理 causal mask
- **Prompt 部分**（包括 vision tokens）：使用**全连接 attention**
- **生成部分**：使用 **causal mask**

**证据来源**：
1. Transformers 库的标准实现
2. Qwen3-VL 的模型架构（基于 Transformer Decoder）
3. 代码中没有显式覆盖这个行为

#### 4. 间接证据：Expert Model 的对比

**代码位置**：`alpamayo_r1.py:251-252`

```python
if self.config.expert_non_causal_attention:
    forward_kwargs["is_causal"] = False  # ⭐ 显式设置非因果
```

**对比**：
- **Expert Model**：**显式设置** `is_causal=False`
- **VLM Generate**：**没有显式设置**，使用默认行为

**推断**：
- 如果 VLM 需要非因果 attention，应该像 Expert Model 一样显式设置
- 但 VLM 的 prompt 部分（包括 vision tokens）默认就是全连接的
- 只有生成部分使用 causal mask

### 重要澄清

**代码中没有显式证据**证明 vision tokens 是全连接的。这是基于 **`vlm.generate()` 的标准行为推断**。

**正确的表述应该是**：

1. **Prompt 部分（包括 vision tokens）**：
   - 在 `vlm.generate()` 的 prompt 阶段，使用**全连接 attention**
   - 这是 Transformers 库的默认行为
   - **代码中没有显式设置，但也没有覆盖这个行为**

2. **生成部分**：
   - 使用 **causal mask**（这是 `generate()` 的默认行为）

### 如何验证？

要真正验证 vision tokens 是否全连接，需要：

1. **查看 Qwen3-VL 的源码**：
   ```python
   # 在 transformers 库中查看
   # transformers/models/qwen3_vl/modeling_qwen3_vl.py
   ```

2. **运行时检查**：
   ```python
   # 在 generate 过程中打印 attention mask
   # 或者使用 hook 查看实际的 attention 权重
   ```

3. **查看模型配置**：
   ```python
   # 检查 vlm.config 中是否有相关设置
   ```

### 代码证据总结

| 证据类型 | 证据内容 | 强度 |
|---------|---------|------|
| **直接证据** | 代码中显式设置 vision tokens 全连接 | ❌ 无 |
| **间接证据** | `vlm.generate()` 默认行为 | ✅ 强（基于标准实现） |
| **对比证据** | Expert Model 显式设置 `is_causal=False` | ✅ 中等（说明如果需要非因果会显式设置） |
| **Processor 输出** | `attention_mask` 只是标记有效 token | ✅ 弱（不控制 causal） |

### 最终结论

1. **代码中没有显式证据**证明 vision tokens 是全连接的
2. **基于标准行为推断**：vision tokens 在 prompt 阶段应该是全连接的
3. **需要进一步验证**：查看 Qwen3-VL 源码或运行时检查

---

# 总结

## 关键设计特点

### 1. 混合架构
- **VLM**: 负责视觉理解和 Chain-of-Causation 推理生成
- **Expert Model**: 专门处理动作序列，复用 VLM 的上下文

### 2. KV Cache 复用
- VLM 生成的 KV cache 被专家模型复用，避免重复计算
- 实现高效的上下文传递

### 3. 轨迹Token系统
- 历史轨迹编码为离散token，无缝融入语言模型
- 未来轨迹通过扩散模型在连续动作空间中生成

### 4. 两阶段生成
- **阶段1**: VLM 生成文本推理（CoC）
- **阶段2**: Expert + Diffusion 生成轨迹动作

### 5. 可解释性
- 每个预测轨迹都伴随 Chain-of-Causation 推理文本
- 提供决策过程的自然语言解释

### 6. Attention 机制
- **VLM**: Causal attention（自回归生成）
- **Expert Model**: Non-causal attention（并行处理）
- **Vision Tokens**: 在 prompt 中，全连接 attention

---

# 第五部分：图像处理与 Vision Encoder

## 17. 图像处理流程详解

本文档详细说明 processor 如何将输入图像从 `[4, 4, 3, 1080, 1920]` 处理成 `pixel_values [11520, 1536]`。

### 17.1 输入数据形状

#### 原始输入

```python
# load_physical_aiavdataset.py 返回
image_frames: (N_cameras=4, num_frames=4, C=3, H=1080, W=1920)
```

**含义**：
- `4` 个相机（views）
- 每个相机 `4` 帧（frames）
- 每帧图像：`(3, 1080, 1920)` - RGB 通道，高度 1080，宽度 1920

#### 图像展平

```python
# test_inference.py 第 33 行
messages = helper.create_message(data["image_frames"].flatten(0, 1))
```

**`flatten(0, 1)` 的效果**：
- 输入：`(4, 4, 3, 1080, 1920)`
- 输出：`(16, 3, 1080, 1920)`

**展平顺序**（View-First）：
```
[Camera0_Frame0, Camera0_Frame1, Camera0_Frame2, Camera0_Frame3,
 Camera1_Frame0, Camera1_Frame1, Camera1_Frame2, Camera1_Frame3,
 Camera2_Frame0, Camera2_Frame1, Camera2_Frame2, Camera2_Frame3,
 Camera3_Frame0, Camera3_Frame1, Camera3_Frame2, Camera3_Frame3]
```

### 17.2 Processor 处理流程

#### Message 构建

```python
# helper.py 第 28-67 行
def create_message(frames: torch.Tensor):
    """frames: (16, 3, 1080, 1920)"""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": frame} for frame in frames  # 16 张图像
            ] + [{"type": "text", "text": "..."}]
        }
    ]
```

**关键点**：
- 每张图像作为独立的 `{"type": "image", "image": frame}` 项
- 图像顺序：View-First（先按相机，再按帧）

#### Processor 调用

```python
# test_inference.py 第 38-45 行
processor = helper.get_processor(model.tokenizer)  # Qwen3-VL processor

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
```

**Processor 配置**：
```python
# helper.py 第 70-79 行
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    min_pixels=163840,   # 最小像素数
    max_pixels=196608,   # 最大像素数
)
```

### 17.3 Processor 内部处理步骤

#### 步骤 1: 图像预处理

Qwen3-VL Processor 对每张图像执行以下操作：

##### Resize（调整大小）

**目标**：将图像调整到目标像素范围内（163840 - 196608 像素）

**原始图像**：
- 尺寸：`(1080, 1920)`
- 像素数：`1080 × 1920 = 2,073,600`

**缩放计算**：
```python
target_pixels = (min_pixels + max_pixels) / 2  # ≈ 180,224
scale_factor = sqrt(target_pixels / original_pixels)  # ≈ 0.295
```

**缩放后尺寸**：
- 保持宽高比缩放
- 目标像素数：约 180,224 像素
- **实际尺寸**：`(320, 576)` 像素（最接近原始宽高比 1.78）
- 验证：`320 × 576 = 184,320` 像素 ✓（在目标范围内）

##### Patch 分割

**Patch 大小**：`16 × 16` 像素

**每张图像的 Patches**：
- 720 patches per image
- 图像尺寸：`(320, 576)` 像素
- H patches：`320 / 16 = 20`
- W patches：`576 / 16 = 36`
- Total patches：`20 × 36 = 720` ✓

**验证**：
```
16 张图像 × 720 patches/图像 = 11,520 patches
输出 pixel_values: [11520, 1536] ✓
```

##### Vision Encoder 处理

**重要**：Vision Encoder 在 **processor 内部**被调用！

每张图像的 patches 通过 Vision Encoder（Qwen3VLVisionModel）：

```
输入: (720, 3, 16, 16)  # 720 个 patches，每个 16×16×3
    ↓
Vision Encoder (Qwen3VLVisionModel)
    ├── Patch Embedding
    ├── Position Embedding  
    ├── Transformer Layers
    └── 输出: (720, 1152)  # Vision Encoder 的输出维度是 1152
    ↓
投影到 1536 维（如果需要）
    ↓
最终: (720, 1536)  # 720 个 patch embeddings，每个 1536 维
```

**关键参数**：
- `patch_size = 16`：每个 patch 是 16×16 像素
- Vision Encoder `hidden_size = 1152`：Vision Encoder 的输出维度
- 最终 `hidden_size = 1536`：投影后的维度（与文本模型匹配）

**调用位置**：
- Vision Encoder 在 `processor.apply_chat_template()` **内部**被调用
- 不是在 `vlm.generate()` 内部调用
- `pixel_values` 输出时已经是 embeddings

#### 步骤 2: 批量处理

**所有图像的处理**：
```
16 张图像 × 720 patches/图像 = 11,520 patches
每个 patch: 1536 维特征
最终输出: (11520, 1536)
```

### 17.4 形状变换详解

#### 完整变换流程

```
输入: [4, 4, 3, 1080, 1920]
    ↓ flatten(0, 1)
[16, 3, 1080, 1920]  # 16 张图像
    ↓ create_message
messages with 16 images
    ↓ processor.apply_chat_template
    ├── 对每张图像:
    │   ├── Resize to target pixels (163840-196608)
    │   ├── Split into patches (16×16 each)
    │   └── Encode with Vision Encoder
    │       → (720, 1536) per image
    └── Concatenate all images
        → (11520, 1536)
```

#### 详细计算

**每张图像的处理**：

**输入**：`(3, 1080, 1920)`

**步骤 1: Resize**
- 目标像素：约 180,224（在 163840-196608 范围内）
- 保持宽高比：`1080:1920 ≈ 1:1.78`
- **实际尺寸**：`(320, 576)` 像素
- 验证：`320 × 576 = 184,320` ✓（在目标范围内）
- 宽高比：`576 / 320 = 1.80` ✓（接近原始 1.78）

**步骤 2: Patch 分割**
- Patch 大小：`16 × 16`
- H patches：`320 / 16 = 20`
- W patches：`576 / 16 = 36`
- Total patches：`20 × 36 = 720` ✓

**步骤 3: Vision Encoder**
- 输入：`(720, 3, 16, 16)` - 720 个 patches
- 输出：`(720, 1536)` - 720 个 patch embeddings

**批量处理**：

**16 张图像**：
```
16 × 720 = 11,520 patches
每个 patch: 1536 维
最终: (11520, 1536) ✓
```

### 17.5 关键参数

#### Processor 配置

```python
# helper.py
MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(
    BASE_PROCESSOR_NAME,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)
```

#### Image Processor 参数

```python
# Qwen3-VL Image Processor
patch_size = 16              # 每个 patch 16×16 像素
hidden_size = 1536           # Vision Encoder 输出维度
min_pixels = 163840          # 最小像素数
max_pixels = 196608          # 最大像素数
```

#### 实际处理结果

- **每张图像 patches**：720
- **每张图像像素**：184,320（在目标范围内）
- **图像尺寸**：`(320, 576)` 像素
- **Patch 布局**：`20 × 36` patches per image（20 行 × 36 列）

### 17.6 形状变换总结

| 阶段 | 形状 | 说明 |
|------|------|------|
| **原始输入** | `(4, 4, 3, 1080, 1920)` | 4个相机×4帧 |
| **展平后** | `(16, 3, 1080, 1920)` | 16张图像 |
| **Resize** | `(16, 3, 320, 576)` | 调整到目标像素（184,320像素） |
| **Patch 分割** | `(16, 720, 3, 16, 16)` | 每张图像720个patches（20×36） |
| **Vision Encoder** | `(16, 720, 1536)` | 每张图像720个embeddings |
| **最终输出** | `(11520, 1536)` | 所有图像的patches拼接 |

### 17.7 关键数字

- **输入图像数**：16 张（4 相机 × 4 帧）
- **每张图像 patches**：720
- **总 patches**：11,520
- **每个 patch 维度**：1,536
- **Patch 大小**：16 × 16 像素
- **目标像素范围**：163,840 - 196,608

---

## 18. Vision Encoder 位置与调用

本文档说明 Vision Encoder 在 Alpamayo 代码中的位置和访问方式。

### 18.1 Vision Encoder 的位置

#### 在 Qwen3-VL 模型中的位置

Vision Encoder 是 **Qwen3-VL 模型的一个组件**，位于模型内部。

**模型结构**：
```
Qwen3VLForConditionalGeneration
  ├── visual (Qwen3VLVisionModel)  ⭐ Vision Encoder 在这里
  └── model (Qwen3VLModel)
      └── language_model (Text Model)
```

#### 在 Alpamayo 代码中的访问路径

**代码位置**：`base_model.py` 第 381 行

```python
# base_model.py 第 367-381 行
def _initialize_qwenvl3_vlm(self, config: ReasoningVLAConfig) -> None:
    """Initialize Qwen3-VL VLM backbone."""
    vlm_config = Qwen3VLConfig.from_pretrained(
        config.vlm_name_or_path,
        dtype=config.model_dtype,
        attn_implementation=config.attn_implementation,
    )
    self.vlm = Qwen3VLForConditionalGeneration(vlm_config)
```

**访问 Vision Encoder**：
```python
# 在 Alpamayo 代码中
vision_encoder = model.vlm.visual  # ⭐ 正确的路径
print(type(vision_encoder))  # <class 'transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel'>
```

### 18.2 Vision Encoder 的配置

**Vision Config**：
```python
# Qwen3-VL Vision Config
vision_config = {
    "hidden_size": 1152,      # Vision Encoder 的隐藏维度
    "patch_size": 16,         # Patch 大小
    # ... 其他配置
}
```

**注意**：
- Vision Encoder 的输出维度是 **1152**（不是 1536）
- 1536 是经过投影后的维度，或者是文本模型的隐藏维度

### 18.3 Vision Encoder 的工作流程

**实际流程**（在 processor 内部）：

```
输入: 原始图像 [16, 3, 1080, 1920]
    ↓ processor.image_processor
    ├── Resize: (1080, 1920) → (320, 576)
    ├── Patch 分割: 20 × 36 = 720 patches per image
    └── 输出: [16, 720, 3, 16, 16]  # 16张图像，每张720个patches
    ↓ processor.apply_chat_template (内部)
    ├── 调用 Vision Encoder (model.vlm.visual)
    │   ├── Patch Embedding
    │   ├── Position Embedding
    │   ├── Transformer Layers
    │   └── 输出: [11520, 1152]  # Vision tokens (1152 是 vision encoder 输出)
    └── 投影到 1536 维（如果需要）
    ↓
最终输出: pixel_values [11520, 1536]  # ⭐ 已经是 vision encoder 的输出
    ↓
传递给 vlm.generate()
    ├── 与 text tokens 融合
    └── 进行自回归生成
```

**关键点**：
- Vision Encoder 在 `processor.apply_chat_template()` **内部**被调用
- 不是在 `vlm.generate()` 内部调用
- `pixel_values` 进入 `sample_trajectories_from_data_with_vlm_rollout` 时已经是 embeddings

### 18.4 代码中的使用

**重要发现**：Vision Encoder 在 **processor 阶段**就被调用了！

**在 Processor 阶段**：

```python
# test_inference.py 第 38-45 行
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
)
# inputs["pixel_values"] 已经是 [11520, 1536] ⭐ Vision Encoder 已执行
```

**关键证据**：
- `pixel_values` 的形状是 `[11520, 1536]`
- `1536` 是 vision encoder 的输出维度（不是原始图像）
- 这说明 **vision encoder 在 `processor.apply_chat_template()` 内部被调用**

**在 VLM Generate 时**：

```python
# alpamayo_r1.py 第 192-198 行
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,  # ⭐ 已经是 vision encoder 的输出
    **tokenized_data,
)
```

**完整流程**：
1. `processor.apply_chat_template()` 
   - 预处理图像（resize, normalize）
   - **调用 Vision Encoder 编码图像** ⭐
   - 输出 `pixel_values: [11520, 1536]`（已经是 embeddings）
2. `vlm.generate()` 
   - 接收已经编码的 `pixel_values`
   - 将 vision embeddings 与 text tokens 融合
   - 进行自回归生成

### 18.5 查看 Vision Encoder 的方法

```python
# 在 Python 中查看
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B")

# 访问 Vision Encoder
vision_encoder = model.vlm.visual  # ⭐ 正确的路径
print(type(vision_encoder))  # <class 'transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionModel'>
print(vision_encoder)

# 查看配置
print(model.vlm.config.vision_config)
```

### 18.6 Vision Encoder 位置总结

| 位置 | 路径 | 说明 |
|------|------|------|
| **在 Qwen3-VL 中** | `model.visual` | Vision Encoder 组件（Qwen3VLVisionModel） |
| **在 Alpamayo 中** | `self.vlm.visual` | 通过 VLM 访问 |
| **配置** | `self.vlm.config.vision_config` | Vision Encoder 配置 |

### 18.7 关键代码位置

1. **VLM 初始化**：`base_model.py:367-381`
   ```python
   self.vlm = Qwen3VLForConditionalGeneration(vlm_config)
   ```

2. **Vision Encoder 调用**：在 `processor.apply_chat_template()` 内部 ⭐
   ```python
   inputs = processor.apply_chat_template(messages, ...)
   # inputs["pixel_values"] 已经是 [11520, 1536]，Vision Encoder 已执行
   ```

3. **VLM Generate 使用已编码的 pixel_values**：`alpamayo_r1.py:192-198`
   ```python
   vlm_outputs = self.vlm.generate(
       pixel_values=pixel_values,  # 已经是 vision encoder 的输出
       ...
   )
   ```

---

## 参考资料

- Alpamayo 代码库：`src/alpamayo_r1/` 目录
- Qwen3-VL 文档：https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
- Transformers 库文档
- Alpamayo 论文：https://arxiv.org/abs/2511.00088

---

**文档版本**: 1.1  
**最后更新**: 2025-01-28  
**整合内容**: 架构文档、Expert Model 详解、Attention 机制分析、Vision Token 分析、图像处理流程、Vision Encoder 位置说明
