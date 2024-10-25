# coding: utf-8
"""
Collection of helper functions for SignLanguageTransformer project.
"""

import copy
import glob
import os
import errno
import shutil
import random
import logging
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
import yaml
from dtw import dtw


class ConfigurationError(Exception):
    """ Custom exception for configuration issues """


def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model, handling cases where directory exists or 
    continuing from a checkpoint.
    """
    if os.path.isdir(model_dir):
        if model_continue:
            return model_dir
        if not overwrite:
            raise FileExistsError("Model directory exists; overwriting disabled.")
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> logging.Logger:
    """
    Create a logger for tracking the training process.
    """
    logger = logging.getLogger("SLT_Logger")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(model_dir, log_file))
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def log_cfg(cfg: dict, logger: logging.Logger, prefix: str = "cfg") -> None:
    """
    Log configuration details recursively.
    """
    for key, value in cfg.items():
        full_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            log_cfg(value, logger, prefix=full_key)
        else:
            logger.info(f"{full_key:34s} : {value}")


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers for transformer models.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (for transformer decoder).
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Create an uneven subsequent mask for sequences of differing lengths.
    """
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path: str = "configs/default.yaml") -> dict:
    """
    Load and parse YAML configuration file.
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string: str) -> str:
    """
    Post-process BPE outputs by recombining BPE-split tokens.
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str, post_fix: str = "_every") -> Optional[str]:
    """
    Get the latest checkpoint (based on modification time) in a directory.
    """
    list_of_files = glob.glob(f"{ckpt_dir}/*{post_fix}.ckpt")
    return max(list_of_files, key=os.path.getctime) if list_of_files else None


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from a saved checkpoint file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint {path} not found")
    return torch.load(path, map_location='cuda' if use_cuda else 'cpu')


def freeze_params(module: nn.Module) -> None:
    """
    Freeze parameters of the module to prevent updating during training.
    """
    for param in module.parameters():
        param.requires_grad = False


def symlink_update(target: str, link_name: str) -> None:
    """
    Create or update a symbolic link.
    """
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def calculate_dtw(references: torch.Tensor, hypotheses: torch.Tensor) -> list:
    """
    Calculate DTW (Dynamic Time Warping) scores between references and hypotheses.
    """
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    dtw_scores = []

    # Remove the BOS frame from the hypothesis
    hypotheses = hypotheses[:, 1:]

    for i, ref in enumerate(references):
        # Trim reference to the maximum frame index
        _, ref_max_idx = torch.max(ref[:, -1], 0)
        ref_count = ref[:ref_max_idx, :-1].cpu().numpy()

        # Trim hypothesis to the maximum frame index
        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        hyp_count = hyp[:hyp_max_idx, :-1].cpu().numpy()

        # Calculate DTW and normalize by sequence length
        d, _, acc_cost_matrix, _ = dtw(ref_count, hyp_count, dist=euclidean_norm)
        dtw_scores.append(d / acc_cost_matrix.shape[0])

    return dtw_scores
