# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset
import yaml
from vocabulary import Vocabulary

from dtw import dtw


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """

def make_model_dir(model_dir: str, overwrite=False, model_continue=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :param model_continue: whether to continue from a checkpoint
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if model_continue:
            return model_dir
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        for file in os.listdir(model_dir):
            file_path = os.path.join(model_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        shutil.rmtree(model_dir, ignore_errors=True)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> (Logger, SummaryWriter):
    """
    Create a logger for logging the training process and initialize SummaryWriter for TensorBoard.
    
    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object and SummaryWriter object
    """
    logger = logging.getLogger("SLT_Logger")
    logger.setLevel(logging.DEBUG)
    
    # File handler for logging to a file
    fh = logging.FileHandler(os.path.join(model_dir, log_file))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Stream handler for console output
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
    # Initialize SummaryWriter for TensorBoard logs
    writer = SummaryWriter(log_dir=os.path.join(model_dir, "tensorboard_logs"))
    
    logger.info("Logger and SummaryWriter initialized.")
    
    return logger, writer


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def uneven_subsequent_mask(x_size: int, y_size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param x_size, y_size: dimensions of the mask
    :return: Tensor with 0s and 1s of shape (1, x_size, y_size)
    """
    mask = np.triu(np.ones((1, x_size, y_size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir, post_fix="_every" ) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory, of either every validation step or best

    :param ckpt_dir: directory of checkpoint
    :param post_fix: type of checkpoint, either "_every" or "_best"
    :return: latest checkpoint file
    """
    list_of_files = glob.glob(f"{ckpt_dir}/*{post_fix}.ckpt")
    return max(list_of_files, key=os.path.getctime) if list_of_files else None


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module, i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def calculate_dtw(references, hypotheses):
    """
    Calculate the DTW costs between a list of references and hypotheses

    :param references: list of reference sequences to compare against
    :param hypotheses: list of hypothesis sequences to fit onto the reference
    :return: dtw_scores: list of DTW costs
    """
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    dtw_scores = []
    hypotheses = hypotheses[:, 1:]

    for i, ref in enumerate(references):
        _, ref_max_idx = torch.max(ref[:, -1], 0)
        ref_max_idx = max(ref_max_idx, 1)
        ref_count = ref[:ref_max_idx, :-1].cpu().numpy()

        hyp = hypotheses[i]
        _, hyp_max_idx = torch.max(hyp[:, -1], 0)
        hyp_max_idx = max(hyp_max_idx, 1)
        hyp_count = hyp[:hyp_max_idx, :-1].cpu().numpy()

        d, cost_matrix, acc_cost_matrix, path = dtw(ref_count, hyp_count, dist=euclidean_norm)
        dtw_scores.append(d / acc_cost_matrix.shape[0])

    return dtw_scores
