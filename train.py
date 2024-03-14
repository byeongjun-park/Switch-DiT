# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
from tqdm import tqdm

from util.data_util import create_dataloader

torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

from models.create_model import create_model
from util.dist_util import cleanup
from util.util import create_logger

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import hydra

import os
from omegaconf import DictConfig, OmegaConf
import wandb
from dotenv import load_dotenv
from util import dist_util
from util.util import flatten_dict, check_conflicts, initialize_cluster, visualize_mask
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import matplotlib as mpl

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


cm_magma = mpl.colormaps.get_cmap("magma")


@torch.compile()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    with torch.no_grad():
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def save_checkpoint(model, ema, opt, config, checkpoint_dir, train_steps, logger, uw):
    if uw is None:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "config": config,
        }
    else:
        checkpoint = {
            "model": model.module.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "config": config,
            "uw": uw.state_dict(),
        }
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return


#################################################################################
#                                  Training Loop                                #
#################################################################################


@hydra.main(config_path="config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist_util.setup_dist(cfg.general)
    device = dist_util.device()
    check_conflicts(cfg)

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            cfg.logs.results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{cfg.logs.results_dir}/*"))
        model_string_name = cfg.models.name.replace(
            "/", "-"
        )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        results_dir = f"{cfg.logs.results_dir}/{model_string_name}"
        experiment_dir = f"{results_dir}/{experiment_index:03d}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        load_dotenv(".env")
        wandb.init(
            entity="",
            project=cfg.logs.project_name,
            name=f"{experiment_index:03d}-{results_dir}",
            dir=results_dir,
        )
    else:
        logger = create_logger(None)

    # Define Diffusion
    diffusion = create_diffusion(
        timestep_respacing="",
        noise_schedule=cfg.general.schedule_name,
        mse_loss_weight_type=cfg.general.loss_weight_type,
    )  # default: 1000 steps

    # Create model:
    cfg.models.param.latent_size = cfg.general.image_size // 8
    if dist.get_rank() == 0:
        wandb.config.update(flatten_dict(OmegaConf.to_container(cfg, resolve=True)))
    model = create_model(model_config=cfg.models, routing_config=cfg.routing)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(
        model.to(device),
        device_ids=[dist_util.device()],
        find_unused_parameters=False,
        bucket_cap_mb=300,
    )

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.general.vae}").to(device)
    model_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_parameters:,}")
    if dist.get_rank() == 0:
        wandb.config.update({"Model Parameters": model_parameters})

    if cfg.general.loss_weight_type == "uw":
        from models.UW import UncertaintyWeighting, sample_t_batch

        uw = DDP(UncertaintyWeighting(num_task=8).to(device), device_ids=[dist_util.device()])
        clusters = initialize_cluster(total_clusters=8, num_timesteps=diffusion.num_timesteps)
        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        opt = torch.optim.AdamW(
            [
                {"params": model.parameters()},
                {"params": uw.parameters(), "lr": 0.025, "weight_decay": 0.0},
            ],
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
        )

    else:
        uw = None
        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )

    # Setup data:
    loader, sampler = create_dataloader(cfg.general, logger)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    if cfg.general.loss_weight_type == "uw":
        uw.train()

    # Variables for monitoring/logging purposes:
    epoch = 0
    log_steps = 0
    running_loss = 0
    running_aux_loss = 0
    start_time = time()

    model = torch.compile(model)  ## not support python 3.11+, use python version <= 3.10
    logger.info(f"Training for {cfg.general.iterations} iterations...")

    @torch.compile()
    def vae_encode(x):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        return x

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.general.mixed_precision)
    for train_steps in tqdm(range(cfg.general.iterations), dynamic_ncols=True):
        try:
            x, y = next(batch_iterator)
        except:
            batch_iterator = iter(loader)
            sampler.set_epoch(epoch)
            logger.info(f"Beginning epoch {epoch}...")
            epoch += 1
            x, y = next(batch_iterator)
        x = x.to(device)
        y = y.to(device)
        if cfg.data.is_uncond == 1:
            y = torch.zeros_like(y)
        with torch.cuda.amp.autocast(enabled=cfg.general.mixed_precision):
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae_encode(x)
            if cfg.general.loss_weight_type == "uw":
                t = sample_t_batch(x.shape[0], clusters, device)
            else:
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

            model_kwargs = dict(y=y)
            loss_dict, aux_loss = diffusion.training_losses(model, x, t, uw, model_kwargs)
            mse_loss = loss_dict["loss"].mean()
            loss = mse_loss + aux_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        update_ema(ema, model.module)

        # Log loss values:
        running_loss += mse_loss.item()
        if cfg.routing.name == "DMoE":
            running_aux_loss += aux_loss.item()

        log_steps += 1
        train_steps += 1
        if train_steps % cfg.logs.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / dist.get_world_size()
            if cfg.routing.name == "DMoE":
                avg_aux_loss = torch.tensor(running_aux_loss / log_steps, device=device)
                dist.all_reduce(avg_aux_loss, op=dist.ReduceOp.SUM)
                avg_aux_loss = avg_aux_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, AUX_Loss: {avg_aux_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if dist.get_rank() == 0:
                    wandb.log(
                        {
                            "Train Loss": avg_loss,
                            "Train AUX Loss": avg_aux_loss,
                            "Train Steps_per_Sec": steps_per_sec,
                        },
                        step=train_steps,
                    )

                    if train_steps % (cfg.logs.log_every * 1) == 0:

                        model.eval()
                        full_task_expert = []

                        with torch.no_grad():
                            eval_t = torch.LongTensor(range(0, diffusion.num_timesteps)).to(model.device)
                            full_t_emb = model.module.t_embedder(eval_t)

                            for i, router in enumerate(model.module.routers):
                                logits = router.f_gate(full_t_emb)
                                top_k_logits, top_k_indices = logits.topk(cfg.routing.param.k, dim=1)
                                top_k_gates = torch.softmax(top_k_logits, dim=1)

                                zeros = torch.zeros((diffusion.num_timesteps, logits.size(1)), dtype=torch.float32, device=logits.device)
                                gates = zeros.scatter(1, top_k_indices, top_k_gates)
                                task_expert_map = (gates > 0).float()
                                full_task_expert.append(task_expert_map)

                            experts_map = torch.cat(full_task_expert, dim=1)
                            experts_map_ = experts_map.repeat_interleave(int(diffusion.num_timesteps / experts_map.size(-1)), dim=1)

                            sim_map = torch.mm(experts_map, experts_map.t())
                            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())

                            wandb.log({f"Task-Expert Affinity": wandb.Image(experts_map_)})
                            wandb.log({f"Task-Task Affinity": wandb.Image(cm_magma(sim_map.detach().cpu().numpy()))})

                        model.train()
            else:
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if dist.get_rank() == 0:
                    wandb.log(
                        {"Train Loss": avg_loss, "Train Steps_per_Sec": steps_per_sec},
                        step=train_steps,
                    )
            if cfg.general.loss_weight_type == "uw":
                logger.info(f"uw_weight: {uw.module.get_loss_weight().tolist()}")

            # Reset monitoring variables:
            running_loss = 0
            running_aux_loss = 0
            log_steps = 0
            start_time = time()

        # Save Checkpoint
        if train_steps % cfg.logs.ckpt_every == 0:  # and train_steps > 0:
            if dist.get_rank() == 0:
                logger.info("saving checkpoint")
                save_checkpoint(model, ema, opt, cfg, checkpoint_dir, train_steps, logger, uw)
            dist.barrier()

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    cleanup()

if __name__ == "__main__":
    main()
