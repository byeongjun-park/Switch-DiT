import torch.distributed as dist
import logging
import sys
import torch
from omegaconf import DictConfig

if sys.version_info[:2] >= (3, 8):
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def flatten_dict(d, parent_key="", sep="_"):
    """
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def check_conflicts(cfg: DictConfig, eval=False):
    if not eval:
        assert cfg.general.data_path is not None, "Please set the data path"
    elif cfg.data.is_uncond:
        print(
            f"You have set cfg_scale to {cfg.eval.cfg_scale}, but this is not used to unconditional generation."
        )
    else:
        assert cfg.eval.cfg_scale >= 1.0, "Please set cfg_scale >= 1.0 for conditional generation"

    assert cfg.general.image_size in [256, 512], "Image Size has to be 256 or 512"
    assert cfg.general.vae in ["ema", "mse"], "vae has to be ema or mse"
    assert not (
        cfg.data.is_uncond and (cfg.data.num_classes != 1)
    ), "You have to set num classes to one for unconditional generation"


def initialize_cluster(total_clusters, num_timesteps=1000):
    clusters = []
    for cluster_ind in range(total_clusters):
        min_t = int(num_timesteps / total_clusters * cluster_ind)
        max_t = int(num_timesteps / total_clusters * (cluster_ind + 1))
        clusters.append((min_t, max_t))
    return clusters

def visualize_mask(dist, beta):
    # clamp dist to [0, 1] and apply cumsum
    dist = torch.clip(dist, min=0, max=1)
    dist_cumsum = torch.cumsum(dist, dim=0)

    # normalize dist_cumsum[-1] eqauls to 1-beta
    normalized_dist_cumsum = (1 - beta) * dist_cumsum / dist_cumsum[-1]
    normalized_dist_cumsum = torch.cat([torch.zeros((1,), device=dist.device), normalized_dist_cumsum], dim=0)

    mask = torch.zeros((1000, 1000), device=dist.device)
    for i in range(1000):
        start = int(1000 * normalized_dist_cumsum[i])
        end = int(1000 * (normalized_dist_cumsum[i] + beta))
        mask[i, start:end] = 1

    return mask