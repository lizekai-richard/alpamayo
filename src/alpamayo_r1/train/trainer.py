import logging
import math
import os
import shutil
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader

from alpamayo_r1 import helper

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    num_epochs: int = 1
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"  # "cosine" or "linear"

    # Batching
    gradient_accumulation_steps: int = 1

    # Eval
    eval_every_n_steps: int = 500
    eval_steps: int | None = None  # cap eval iterations, None = full

    # Checkpointing
    output_dir: str = "checkpoints"
    save_every_n_steps: int = 500
    save_total_limit: int = 3  # rotate old checkpoints
    resume_from_checkpoint: str | None = None

    # Logging
    log_every_n_steps: int = 10
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    # Training stage
    training_stage: str = "vlm"  # "vlm" or "expert"

    # Rollout
    rollout_top_p: float = 0.98
    rollout_temperature: float = 0.6
    rollout_num_traj_samples: int = 1
    rollout_max_generation_length: int = 256

    # FSDP
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD / SHARD_GRAD_OP / NO_SHARD
    fsdp_auto_wrap_min_params: int = 1_000_000
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE / BACKWARD_POST
    fsdp_sync_module_states: bool = True


# Parameter names that should not receive weight decay
NO_DECAY_PATTERNS = {"bias", "LayerNorm.weight", "layernorm", "layer_norm", "embed"}


def _get_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in NO_DECAY_PATTERNS):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def _build_lr_lambda(
    warmup_steps: int,
    total_steps: int,
    scheduler_type: str,
):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        # linear
        return max(0.0, 1.0 - progress)

    return lr_lambda


def _calc_min_ade(
    gt_future_xyz: torch.Tensor, pred_xyz: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Compute minADE for a single sample.

    Args:
        gt_future_xyz: [1, 1, T, 3] ground-truth future trajectory.
        pred_xyz: [1, num_traj_sets, num_traj_samples, T, 3] predicted trajectories.

    Returns:
        Tuple of (scalar minADE, index of best trajectory sample).
    """
    gt_xy = gt_future_xyz[0, 0, :, :2]  # [T, 2]
    pred_xy = pred_xyz[0, 0, :, :, :2]  # [num_traj_samples, T, 2]
    diff = (pred_xy - gt_xy.unsqueeze(0)).norm(dim=-1)  # [num_traj_samples, T]
    ade = diff.mean(dim=-1)  # [num_traj_samples]
    best_idx = ade.argmin().item()
    return ade[best_idx], best_idx


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
    ):
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # Distributed / FSDP setup
        self.is_fsdp = config.use_fsdp
        if self.is_fsdp:
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = torch.device("cuda", self.local_rank)
            self.model = self._setup_fsdp(model)
        else:
            self.rank = 0
            self.local_rank = 0
            self.device = torch.device("cuda")
            self.model = model

        # Optimizer (must be created after FSDP wrapping)
        param_groups = _get_parameter_groups(self.model, config.weight_decay)
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
        )

        # Scheduler
        steps_per_epoch = math.ceil(
            len(train_dataloader) / config.gradient_accumulation_steps
        )
        self.total_steps = steps_per_epoch * config.num_epochs
        lr_lambda = _build_lr_lambda(
            config.warmup_steps, self.total_steps, config.lr_scheduler_type
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        self._wandb = None

    @property
    def _is_main_process(self) -> bool:
        return self.rank == 0

    def _setup_fsdp(self, model: torch.nn.Module) -> FSDP:
        sharding_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_map[self.config.fsdp_sharding_strategy]

        from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
        prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = prefetch_map[self.config.fsdp_backward_prefetch]

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=self.config.fsdp_auto_wrap_min_params,
        )

        wrapped = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=CPUOffload(offload_params=self.config.fsdp_cpu_offload),
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=backward_prefetch,
            mixed_precision=mp_policy,
            device_id=self.local_rank,
            sync_module_states=self.config.fsdp_sync_module_states,
        )
        logger.info(
            f"FSDP initialized: sharding={self.config.fsdp_sharding_strategy}, "
            f"rank={self.rank}, local_rank={self.local_rank}"
        )
        return wrapped

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        self._init_wandb()

        if self.config.resume_from_checkpoint:
            self._load_checkpoint(self.config.resume_from_checkpoint)

        start_epoch = self.epoch

        for epoch in range(start_epoch, self.config.num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch}")
            # Update sampler epoch for proper shuffling across ranks
            if self.is_fsdp and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
            self.model.train()

            accum_loss = 0.0
            accum_count = 0

            for batch_idx, batch in enumerate(self.train_dataloader):
                # When resuming, skip already-processed batches
                if self.config.resume_from_checkpoint and epoch == start_epoch:
                    completed_batches = (
                        self.global_step * self.config.gradient_accumulation_steps
                    )
                    if batch_idx < completed_batches:
                        continue

                loss = self._training_step(batch)
                accum_loss += loss
                accum_count += 1

                if accum_count % self.config.gradient_accumulation_steps == 0:
                    self._optimization_step()

                    if self.global_step % self.config.log_every_n_steps == 0:
                        avg_loss = accum_loss / accum_count
                        metrics = {
                            "train/loss": avg_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        }
                        self._log(metrics, self.global_step)
                        accum_loss = 0.0
                        accum_count = 0

                    if (
                        self.eval_dataloader is not None
                        and self.global_step % self.config.eval_every_n_steps == 0
                    ):
                        eval_metrics = self._evaluate()
                        self._log(eval_metrics, self.global_step)
                        eval_score = eval_metrics.get("eval/min_ade", float("inf"))
                        is_best = eval_score < self.best_eval_loss
                        if is_best:
                            self.best_eval_loss = eval_score
                        self._save_checkpoint(is_best=is_best)
                        self.model.train()

                    if self.global_step % self.config.save_every_n_steps == 0:
                        self._save_checkpoint(is_best=False)

            # End-of-epoch save
            self._save_checkpoint(is_best=False)

        if self._wandb is not None:
            self._wandb.finish()

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _training_step(self, batch: list[dict[str, Any]]) -> float:
        device = self.device

        # Reset streaming state before each batch
        self.model.reset_streaming_state()

        # Rollout: populate streaming kv cache (no grad)
        with torch.no_grad():
            for window in batch[:-1]:
                self.model.sample_trajectories_from_data_with_streaming_vlm_rollout(
                    data=helper.to_device(
                        {
                            "tokenized_data": {
                                "input_ids": window["input_ids"],
                                "attention_mask": window["attention_mask"],
                                "pixel_values": window["pixel_values"],
                                "image_grid_thw": window["image_grid_thw"],
                            },
                            "ego_history_xyz": window["ego_history_xyz"],
                            "ego_history_rot": window["ego_history_rot"],
                            "is_prefill": window["is_prefill"],
                        },
                        device,
                    ),
                    top_p=self.config.rollout_top_p,
                    temperature=self.config.rollout_temperature,
                    num_traj_samples=self.config.rollout_num_traj_samples,
                    max_generation_length=self.config.rollout_max_generation_length,
                )

        # Forward on the last window with autocast bf16
        last = batch[-1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.config.training_stage == "vlm":
                output = self.model._forward_vlm_training(
                    input_ids=last["input_ids"].to(device),
                    attention_mask=last["attention_mask"].to(device),
                    pixel_values=last["pixel_values"].to(device),
                    image_grid_thw=last["image_grid_thw"].to(device),
                    ego_history_xyz=last["ego_history_xyz"].to(device),
                    ego_history_rot=last["ego_history_rot"].to(device),
                    output_ids_range=last["output_ids_range"],
                    labels=last["labels"].to(device),
                )
                loss = output.loss
            else:  # expert
                loss = self.model._forward_expert_training(
                    input_ids=last["input_ids"].to(device),
                    attention_mask=last["attention_mask"].to(device),
                    pixel_values=last["pixel_values"].to(device),
                    image_grid_thw=last["image_grid_thw"].to(device),
                    max_new_tokens=self.config.rollout_max_generation_length,
                    temperature=self.config.rollout_temperature,
                    top_k=None,
                    top_p=self.config.rollout_top_p,
                    ego_history_xyz=last["ego_history_xyz"].to(device),
                    ego_history_rot=last["ego_history_rot"].to(device),
                    ego_future_xyz=last["ego_future_xyz"].to(device),
                    ego_future_rot=last["ego_future_rot"].to(device),
                )
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        return loss.detach().item()

    def _optimization_step(self):
        if self.config.max_grad_norm > 0:
            if self.is_fsdp:
                self.model.clip_grad_norm_(self.config.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        self.global_step += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self) -> dict[str, float]:
        """Run evaluation on rank 0 only. Other ranks wait at barrier."""
        metrics: dict[str, float] = {}

        if self._is_main_process:
            # Under FSDP, consolidate parameters to rank 0 for eval
            if self.is_fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.summon_full_params(self.model):
                    metrics = self._run_eval_loop()
            else:
                metrics = self._run_eval_loop()

        if self.is_fsdp:
            dist.barrier()

        return metrics

    def _run_eval_loop(self) -> dict[str, float]:
        self.model.eval()
        device = self.device
        total_min_ade = 0.0
        n_min_ade = 0
        all_cot_texts: list[str] = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                if (
                    self.config.eval_steps is not None
                    and batch_idx >= self.config.eval_steps
                ):
                    break

                # Reset streaming state before each eval batch
                self.model.reset_streaming_state()

                # Rollout all windows, compute minADE per window
                for window in batch:
                    pred_xyz, pred_rot, extra = self.model.sample_trajectories_from_data_with_streaming_vlm_rollout(
                        data=helper.to_device(
                            {
                                "tokenized_data": {
                                    "input_ids": window["input_ids"],
                                    "attention_mask": window["attention_mask"],
                                    "pixel_values": window["pixel_values"],
                                    "image_grid_thw": window["image_grid_thw"],
                                },
                                "ego_history_xyz": window["ego_history_xyz"],
                                "ego_history_rot": window["ego_history_rot"],
                                "is_prefill": window["is_prefill"],
                            },
                            device,
                        ),
                        top_p=self.config.rollout_top_p,
                        temperature=self.config.rollout_temperature,
                        num_traj_samples=self.config.rollout_num_traj_samples,
                        max_generation_length=self.config.rollout_max_generation_length,
                        return_extra=True,
                    )
                    min_ade, min_ade_idx = _calc_min_ade(
                        window["ego_future_xyz"].to(device), pred_xyz,
                    )
                    best_cot = extra["cot"][0][0][min_ade_idx]
                    total_min_ade += min_ade
                    n_min_ade += 1
                    all_cot_texts.append(best_cot)

        metrics: dict[str, float] = {}
        if n_min_ade > 0:
            metrics["eval/min_ade"] = total_min_ade / n_min_ade

        parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        logger.info(f"Eval: {', '.join(parts)}")

        if all_cot_texts:
            self._log_cot_table(all_cot_texts, self.global_step)

        return metrics
    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, is_best: bool = False):
        ckpt_dir = os.path.join(
            self.config.output_dir, f"checkpoint-{self.global_step}"
        )

        if self.is_fsdp:
            # Gather full state dict to rank 0
            full_sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_sd_cfg):
                model_state = self.model.state_dict()
                optim_state = FSDP.optim_state_dict(self.model, self.optimizer)

            if self._is_main_process:
                os.makedirs(ckpt_dir, exist_ok=True)
                # Save model weights
                torch.save(model_state, os.path.join(ckpt_dir, "model_state_dict.pt"))
                # Save optimizer / scheduler / training state
                state = {
                    "optimizer": optim_state,
                    "scheduler": self.scheduler.state_dict(),
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "best_eval_loss": self.best_eval_loss,
                }
                torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
                logger.info(f"Saved FSDP checkpoint to {ckpt_dir}")
                self._rotate_checkpoints()
                if is_best:
                    best_dir = os.path.join(self.config.output_dir, "best_model")
                    if os.path.exists(best_dir):
                        shutil.rmtree(best_dir)
                    shutil.copytree(ckpt_dir, best_dir)
                    logger.info(f"Saved best model to {best_dir}")
            dist.barrier()
        else:
            os.makedirs(ckpt_dir, exist_ok=True)
            self.model.save_pretrained(ckpt_dir)
            state = {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
            }
            torch.save(state, os.path.join(ckpt_dir, "training_state.pt"))
            logger.info(f"Saved checkpoint to {ckpt_dir}")
            self._rotate_checkpoints()
            if is_best:
                best_dir = os.path.join(self.config.output_dir, "best_model")
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                shutil.copytree(ckpt_dir, best_dir)
                logger.info(f"Saved best model to {best_dir}")

    def _rotate_checkpoints(self):
        if self.config.save_total_limit is None or self.config.save_total_limit <= 0:
            return

        # List all checkpoint-* dirs sorted by step
        ckpt_dirs = []
        for name in os.listdir(self.config.output_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(self.config.output_dir, name)
                if os.path.isdir(path):
                    try:
                        step = int(name.split("-")[1])
                        ckpt_dirs.append((step, path))
                    except (IndexError, ValueError):
                        continue
        ckpt_dirs.sort(key=lambda x: x[0])

        # Remove oldest if over limit
        while len(ckpt_dirs) > self.config.save_total_limit:
            _, path = ckpt_dirs.pop(0)
            shutil.rmtree(path)
            logger.info(f"Removed old checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        state_path = os.path.join(path, "training_state.pt")
        if not os.path.isfile(state_path):
            logger.warning(
                f"No training_state.pt found in {path}, starting from scratch"
            )
            return

        state = torch.load(state_path, map_location="cpu", weights_only=False)

        if self.is_fsdp:
            # Load model weights
            model_sd_path = os.path.join(path, "model_state_dict.pt")
            if os.path.isfile(model_sd_path):
                full_sd = torch.load(model_sd_path, map_location="cpu", weights_only=False)
                full_sd_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_sd_cfg):
                    self.model.load_state_dict(full_sd)
            # Load optimizer state
            optim_state = FSDP.optim_state_dict_to_load(
                self.model, self.optimizer, state["optimizer"]
            )
            self.optimizer.load_state_dict(optim_state)
        else:
            self.optimizer.load_state_dict(state["optimizer"])

        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_eval_loss = state.get("best_eval_loss", float("inf"))
        logger.info(
            f"Resumed from {path} at step {self.global_step}, epoch {self.epoch}"
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_wandb(self):
        if self.config.wandb_project is None or not self._is_main_process:
            return
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                entity=self.config.wandb_entity,
                config={
                    k: v
                    for k, v in self.config.__dict__.items()
                    if not k.startswith("_")
                },
            )
        except ImportError:
            logger.warning("wandb not installed, skipping wandb logging")

    def _log(self, metrics: dict[str, Any], step: int):
        if not self._is_main_process:
            return
        parts = [f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()]
        logger.info(f"[step {step}] {', '.join(parts)}")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)

    def _log_cot_table(self, cot_texts: list[str], step: int):
        """Log generated Chain-of-Thought texts to wandb as a table."""
        for i, cot in enumerate(cot_texts):
            logger.info(f"[step {step}] eval sample {i} CoT: {cot[:200]}...")
        if self._wandb is not None:
            table = self._wandb.Table(
                columns=["sample_idx", "cot"],
                data=[[i, cot] for i, cot in enumerate(cot_texts)],
            )
            self._wandb.log({"eval/cot": table}, step=step)
