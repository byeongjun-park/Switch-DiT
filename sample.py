# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch

from models.create_model import create_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from omegaconf import DictConfig
import hydra
from util.util import check_conflicts

@hydra.main(config_path="config", config_name="base_config.yaml")
def main(cfg: DictConfig):
    # Setup PyTorch:
    torch.manual_seed(cfg.general.global_seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    check_conflicts(cfg, eval=True)

    # Load model:
    latent_size = cfg.general.image_size // 8
    cfg.models.param.latent_size = latent_size
    model = create_model(model_config=cfg.models, routing_config=cfg.routing).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    model_string_name = cfg.models.name.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    ckpt_path = f'{cfg.logs.results_dir}/{model_string_name}/{cfg.eval.ckpt_path.version:03d}/checkpoints/{cfg.eval.ckpt_path.iterations:07d}.pt'
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(cfg.eval.num_sampling_steps), noise_schedule=cfg.general.schedule_name)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.general.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    if not cfg.data.is_uncond:
        class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    else:
        class_labels = [0] * 8

    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    if not cfg.data.is_uncond:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([1000] * n, device=device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=cfg.eval.cfg_scale)
        sample_fn = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        sample_fn = model.forward

    # Sample images:
    samples = diffusion.p_sample_loop(sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)

    if not cfg.data.is_uncond:
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    main()
