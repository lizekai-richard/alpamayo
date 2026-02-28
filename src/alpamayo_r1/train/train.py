import logging
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from alpamayo_r1.train.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper
from alpamayo_r1.train.dataset import StreamingDataset, EvalStreamingDataset, collate_fn, batched_collate_fn, eval_collate_fn
from alpamayo_r1.train.trainer import Trainer, TrainerConfig

_rank = int(os.environ.get("RANK", 0))
logging.basicConfig(
    level=logging.INFO if _rank == 0 else logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _is_torchrun() -> bool:
    """Auto-detect distributed launch (torchrun sets WORLD_SIZE)."""
    return int(os.environ.get("WORLD_SIZE", 1)) > 1


def train(cfg):
    distributed = _is_torchrun()

    # Distributed setup
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    logger.info("Distributed setup done" if distributed else "Single-GPU mode")

    # Load model to CPU; the Trainer moves to the correct device per rank.
    logger.info("Loading model...")
    path = cfg.resume_from_checkpoint if cfg.resume_from_checkpoint else cfg.model_path
    model = AlpamayoR1.from_pretrained(path, dtype=torch.bfloat16)
    model.set_training_stage(cfg.training_stage)
    logger.info(f"Model loaded, training_stage={cfg.training_stage}")

    processor = helper.get_processor(model.tokenizer)

    # Train dataset / dataloader
    logger.info("Building train dataset...")
    train_dataset = StreamingDataset(cfg, processor=processor, training_stage=cfg.training_stage)
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    num_workers = getattr(cfg, "num_workers", 0)
    batch_size = getattr(cfg, "batch_size", 1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # dataset handles clip-grouped ordering internally
        collate_fn=batched_collate_fn,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    # Eval dataset / dataloader (optional)
    eval_dataloader = None
    if cfg.eval_data_dir is not None:
        eval_cfg = OmegaConf.merge(cfg, {"data_dir": cfg.eval_data_dir, "clip_list": cfg.eval_clip_list})
        eval_dataset = EvalStreamingDataset(eval_cfg, processor=processor)
        logger.info(f"Eval dataset: {len(eval_dataset)} samples")
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=eval_collate_fn,
        )

    # Build trainer config from merged cfg
    logger.info("Building trainer...")
    trainer_config = TrainerConfig(
        training_stage=cfg.training_stage,
        min_rollout_steps=cfg.min_rollout_steps,
        max_rollout_steps=cfg.max_rollout_steps,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        adam_epsilon=cfg.adam_epsilon,
        max_grad_norm=cfg.max_grad_norm,
        num_epochs=cfg.num_epochs,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_every_n_steps=cfg.eval_every_n_steps,
        eval_steps=cfg.eval_steps,
        output_dir=cfg.output_dir,
        save_every_n_steps=cfg.save_every_n_steps,
        save_total_limit=cfg.save_total_limit,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        log_every_n_steps=cfg.log_every_n_steps,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        wandb_entity=cfg.wandb_entity,
        rollout_top_p=cfg.rollout_top_p,
        rollout_temperature=cfg.rollout_temperature,
        rollout_num_traj_samples=cfg.rollout_num_traj_samples,
        rollout_max_generation_length=cfg.rollout_max_generation_length,
        distributed=distributed,
        use_deepspeed=getattr(cfg, "use_deepspeed", False),
        deepspeed=OmegaConf.to_container(cfg.deepspeed, resolve=True) if getattr(cfg, "deepspeed", None) else None,
    )

    trainer = Trainer(
        config=trainer_config,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    logger.info("Trainer ready, starting training...")
    trainer.train()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Usage: python train.py --config configs/train_vlm.yaml [key=value overrides...]
    # e.g.:  python train.py --config configs/train_vlm.yaml lr=2e-5 num_epochs=3
    if "--config" not in sys.argv:
        print("Usage: python train.py --config <path.yaml> [key=value ...]")
        sys.exit(1)

    config_idx = sys.argv.index("--config")
    config_path = sys.argv[config_idx + 1]
    cli_overrides = [a for a in sys.argv[1:] if "=" in a]

    file_cfg = OmegaConf.load(config_path)
    cli_cfg = OmegaConf.from_dotlist(cli_overrides)
    cfg = OmegaConf.merge(file_cfg, cli_cfg)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    train(cfg)
