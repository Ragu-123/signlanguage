# coding: utf-8
"""
Collection of helper functions for SignLanguageTransformer project
"""
import copy
import glob
import os
import shutil
import random
import logging
from typing import Optional, List, Dict
import numpy as np

import torch
from torch import nn, Tensor

import yaml
from dtw import dtw


class ConfigurationError(Exception):
    """Custom exception for configuration errors"""


def make_model_dir(model_dir: str, overwrite=False, continue_training=False) -> str:
    """
    Create or clean model directory for saving checkpoints and logs.

    :param model_dir: Path to model directory
    :param overwrite: Whether to overwrite an existing directory
    :param continue_training: Whether to continue from an existing model directory
    :return: model_dir
    """
    if os.path.isdir(model_dir):
        if continue_training:
            return model_dir
        if not overwrite:
            raise FileExistsError(f"Model directory {model_dir} exists and overwrite is disabled.")
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> logging.Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: Path to logging directory
    :param log_file: Name of the log file
    :return: Logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f"{model_dir}/{log_file}")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("SignLanguageTransformer Training Log")
    return logger


def log_cfg(cfg: dict, logger: logging.Logger, prefix: str = "cfg") -> None:
    """
    Log configuration settings.

    :param cfg: Configuration dictionary to log
    :param logger: Logger to write config details
    :param prefix: Prefix for configuration entries
    """
    for k, v in cfg.items():
        p = f"{prefix}.{k}"
        if isinstance(v, dict):
            log_cfg(v, logger, prefix=p)
        else:
            logger.info(f"{p:34s} : {v}")


def load_config(path="configs/base.yaml") -> dict:
    """
    Load and parse a YAML configuration file.

    :param path: Path to YAML configuration file
    :return: Configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility across torch, numpy, and random.

    :param seed: Seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Create N identical layers for Transformer models.

    :param module: Module to clone
    :param n: Number of clones
    :return: List of cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions to prevent attending to future tokens.

    :param size: Size of mask
    :return: Mask tensor
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0  # Returns True/False mask


def load_keypoints(sentence_name: str, keypoints_dir: str) -> np.ndarray:
    """
    Load OpenPose keypoints for a given sentence sequence.

    :param sentence_name: Name of the sentence in How2Sign dataset
    :param keypoints_dir: Path to the directory containing keypoints
    :return: Array of keypoints for each frame in the sequence
    """
    keypoint_files = sorted(glob.glob(os.path.join(keypoints_dir, f"{sentence_name}/*.json")))
    keypoints = [np.load(file)["pose_keypoints_2d"] for file in keypoint_files]
    return np.array(keypoints)


def calculate_dtw(references: List[np.ndarray], hypotheses: List[np.ndarray]) -> List[float]:
    """
    Calculate Dynamic Time Warping (DTW) scores between reference and hypothesis sequences.

    :param references: List of reference keypoint sequences
    :param hypotheses: List of hypothesis keypoint sequences
    :return: DTW scores
    """
    euclidean_norm = lambda x, y: np.sum(np.abs(x - y))
    dtw_scores = []
    for ref, hyp in zip(references, hypotheses):
        d, _, _, _ = dtw(ref, hyp, dist=euclidean_norm)
        dtw_scores.append(d / len(ref))
    return dtw_scores


# Removed the normalize_keypoints function since normalization is not needed.


def bpe_postprocess(sentence: str) -> str:
    """
    Remove BPE (Byte-Pair Encoding) artifacts from tokenized text.

    :param sentence: BPE tokenized string
    :return: Cleaned string
    """
    return sentence.replace("@@ ", "")


def save_checkpoint(state: Dict, is_best: bool, checkpoint_dir: str, filename: str = "checkpoint.pth.tar"):
    """
    Save the model's state as a checkpoint.

    :param state: Model state dictionary
    :param is_best: If True, saves an additional "best" checkpoint
    :param checkpoint_dir: Directory to save the checkpoint
    :param filename: Filename for the checkpoint
    """
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best_checkpoint.pth.tar'))
