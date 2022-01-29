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
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor, BoolTensor
from torchtext.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary, TextVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError("Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        fh.setFormatter(formatter)
        if platform == "linux":
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
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
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    logging_function: Callable[[str], None],
):
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param gls_vocab:
    :param txt_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain {:d},\n\tvalid {:d},\n\ttest {:d}".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    )

    logging_function(
        "First 10 words (gls): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )
    logging_function(
        "First 10 words (txt): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(txt_vocab.itos[:10]))
        )
    )

    logging_function("Number of unique glosses (types): {}".format(len(gls_vocab)))
    logging_function("Number of unique words (types): {}".format(len(txt_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

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


def wait_K_mask_generate(
    target_length: int,
    device: torch.device,
    K: int,
    batch_size: int = None,
    source_length: int = None,
    static_gls_len: int = 0,
) -> BoolTensor:

    mask_length = torch.as_tensor(
        [
            i * static_gls_len if i * static_gls_len <= source_length else source_length
            for i in range(K, K + target_length)
        ],
        device=device,
    )
    result = get_mask_from_sequence_lengths(
        mask_length, device=device, max_length=source_length
    )
    batch_result = result.expand(batch_size, result.shape[0], result.shape[1])
    return batch_result


def get_mask_from_sequence_lengths(
    sequence_lengths: torch.Tensor, device: torch.device = None, max_length: int = None
) -> torch.BoolTensor:
    """
    Given a variable of shape `(batch_size,)` that represents the sequence lengths of each batch
    element, this function returns a `(batch_size, max_length)` mask variable.  For example, if
    our input was `[2, 2, 3]`, with a `max_length` of 4, we'd return
    `[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]`.

    We require `max_length` here instead of just computing it from the input `sequence_lengths`
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    if max_length is None:
        max_length = int(sequence_lengths.max().item())
    if device is not None:
        ones = torch.ones((sequence_lengths.size(0), max_length), device=device)
    else:
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return sequence_lengths.unsqueeze(1) >= range_tensor


def memory_mask_generator(
    txt_length: int,
    device: torch.device,
    wait_k: int,
    weight_set: Tensor,
    gls_length: int = None,
    encoder_output: Tensor = None,
) -> BoolTensor:
    """generate memory mask

    Args:
        target_length (int): target length
        device (torch.device): tensor device
        weight_set (Tensor): weight set

    Returns:
        BoolTensor:
    """
    batch_size, source_length = weight_set.shape
    cum_weight = torch.cumsum(weight_set, dim=-1)
    x = torch.floor(cum_weight)
    right_x = torch.roll(x, 1, dims=-1)
    right_x[:, 0] = 0

    # generate memory mask
    expand_right_x = right_x.unsqueeze(1)
    expand_right_x = expand_right_x.expand(-1, txt_length, -1)
    mask_up_bound = (
        torch.arange(wait_k, txt_length + wait_k, device=device)
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    mask_up_bound = mask_up_bound.expand(batch_size, -1, source_length)
    wait_k_mask = (mask_up_bound - expand_right_x) > 0

    if gls_length is not None:
        # generate weight weighted encoder output
        pos = x - right_x
        start_weight = pos * (cum_weight - x)
        left_weight = weight_set - start_weight

        res = []
        _weight = left_weight
        for i in range(int(torch.max(x).item()) + 1):
            pos_mask = (x == i) | (right_x == i)
            tmp = (_weight * pos_mask.long()).unsqueeze(-1) * encoder_output
            _weight = _weight * (1 - pos_mask.long()) + start_weight * pos_mask.long()
            tmp = torch.sum(tmp, 1, keepdim=True)
            res.append(tmp)
        res = torch.cat(res, dim=1)
        if res.size(1) < gls_length:
            res = torch.cat(
                (
                    res,
                    torch.zeros(
                        (res.size(0), gls_length - res.size(1), res.size(2))
                    ).to(res.device),
                ),
                dim=1,
            )
        else:
            res = res[:, :gls_length, :]
        res_mask = torch.sum(res, dim=-1) != 0
    else:
        res = None
        res_mask = None

    return wait_k_mask, res, res_mask


def gen_wait_k_mask_from_boundary_possibility(
    boundary_possibility: Tensor,
    target_length: int,
    wait_k: int,
    threshold: float = 0.5,
) -> BoolTensor:
    """Generate wait k mask according to boundary probability

    Args:
        boundary_possibility (Tensor): Predicted boundary probability
        target_length (int): The maximum length of the target text
        wait_k (int): wait k
        threshold (float, optional): Threshold for judging the boundary. Defaults to 0.5.

    Returns:
        BoolTensor: wait k mask
    """
    batch_size, max_sign = boundary_possibility.shape
    wait_k_mask = boundary_possibility.new_ones(
        (batch_size, target_length, max_sign), dtype=torch.bool
    )
    boundary = boundary_possibility > threshold
    for batch_wait_k_mask, batch_boundary in zip(wait_k_mask, boundary):
        target_index = 0
        run_wait_k = wait_k
        for index, boundary_value in enumerate(batch_boundary):
            run_wait_k -= 1
            if boundary_value:
                if target_index < target_length and run_wait_k < 1:
                    batch_wait_k_mask[target_index, index + 1 :] = False
                    target_index += 1

    return wait_k_mask


def gen_reencoder_mask_from_wait_k_mask(wait_k_mask: Tensor) -> BoolTensor:
    """generate reencoder mask from wait k mask

    Args:
        wait_k_mask (Tensor): waitk mask,shape [batch_size,T,s]

    Returns:
        BoolTensor: reencoder mask ,shape [batch_size,s,s]
    """
    batch_size, t_len, s_len = wait_k_mask.shape
    reencoder_mask = wait_k_mask.new_ones((batch_size, s_len, s_len), dtype=torch.bool)
    # shape batch_size,t
    sum_len = wait_k_mask.sum(dim=-1)
    for batch_reencoder_mask, batch_sum_len in zip(reencoder_mask, sum_len):
        last_row = 0
        for sign_len in batch_sum_len:
            if sign_len == last_row:
                break
            batch_reencoder_mask[last_row:sign_len, sign_len:] = False
            last_row = sign_len
    return reencoder_mask
