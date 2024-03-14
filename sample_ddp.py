# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from omegaconf import DictConfig
import hydra
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math

from models.create_model import create_model
from util import dist_util
from util.util import check_conflicts


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


@hydra.main(config_path="config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = (
        True  # True: fast but may lead to some small numerical differences
    )
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist_util.setup_dist(cfg.general)
    device = dist_util.device()
    rank = dist.get_rank()

    check_conflicts(cfg, eval=True)

    # define diffusion
    diffusion = create_diffusion(
        str(cfg.eval.num_sampling_steps), noise_schedule=cfg.general.schedule_name
    )

    # Load model:
    latent_size = cfg.general.image_size // 8
    cfg.models.param.latent_size = latent_size

    model = create_model(model_config=cfg.models, routing_config=cfg.routing).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    model_string_name = cfg.models.name.replace(
        "/", "-"
    )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    ckpt_path = f"{cfg.logs.results_dir}/{model_string_name}/{cfg.eval.ckpt_path.version:03d}/checkpoints/{cfg.eval.ckpt_path.iterations:07d}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.general.vae}").to(device)

    if not cfg.data.is_uncond:
        folder_name = f"{cfg.eval.ckpt_path.iterations:07d}-size-{cfg.general.image_size}-vae-{cfg.general.vae}-samples-{cfg.eval.num_fid_samples}-cfg-{cfg.eval.cfg_scale}-seed-{cfg.general.global_seed}"
    else:
        folder_name = f"{cfg.eval.ckpt_path.iterations:07d}-size-{cfg.general.image_size}-vae-{cfg.general.vae}-samples-{cfg.eval.num_fid_samples}-seed-{cfg.general.global_seed}"
    sample_folder_dir = f"{cfg.eval.samples_dir}/{model_string_name}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = cfg.eval.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(cfg.eval.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = (
            torch.randint(0, cfg.data.num_classes, (n,), device=device)
            if not cfg.data.is_uncond
            else torch.zeros((n,), dtype=torch.int64, device=device)
        )

        # Setup classifier-free guidance:
        if not cfg.data.is_uncond:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([cfg.data.num_classes] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=cfg.eval.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        with torch.cuda.amp.autocast():
            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=device,
            )
            if not cfg.data.is_uncond:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
        samples = (
            torch.clamp(127.5 * samples + 128.0, 0, 255)
            .permute(0, 2, 3, 1)
            .to("cpu", dtype=torch.uint8)
            .numpy()
        )

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, cfg.eval.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
