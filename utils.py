# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import torch
from torch import Tensor
from typing import Optional, Union, List
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import random

from omegaconf import OmegaConf, DictConfig

CONST_log_range = 20
CONST_log_min = 1e-10
CONST_summary_rescale = 10
CONST_exp_range = 10
CONST_min_std_dev = math.exp(-CONST_exp_range)


def sandwich(x: Tensor):
    return x.reshape(x.size(0), -1, x.size(-1))


def safe_log(data: Tensor):
    return data.clamp(min=CONST_log_min).log()


def safe_exp(data: Tensor):
    return data.clamp(min=-CONST_exp_range, max=CONST_exp_range).exp()


def idx_to_float(idx: np.ndarray, num_bins: int):
    flt_zero_one = (idx + 0.5) / num_bins
    return (2.0 * flt_zero_one) - 1.0


def float_to_idx(flt: np.ndarray, num_bins: int):
    flt_zero_one = (flt / 2.0) + 0.5
    return torch.clamp(torch.floor(flt_zero_one * num_bins), min=0, max=num_bins - 1).long()


def quantize(flt, num_bins: int):
    return idx_to_float(float_to_idx(flt, num_bins), num_bins)


def pe_encode(sequence_length: int, embedding_size: int) -> Tensor:
    """Positional encoding as described in original attention is all you need paper"""

    pe = torch.zeros((sequence_length, embedding_size))
    pos = torch.arange(sequence_length).unsqueeze(1)
    pe[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )
    pe[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )

    return pe


def pe_encode_float(x: Tensor, max_freq: float, embedding_size: int) -> Tensor:
    pe = torch.zeros(list(x.shape) + [embedding_size], device=x.device)
    pos = (((x + 1) / 2) * max_freq).unsqueeze(-1)
    pe[..., 0::2] = torch.sin(
        pos
        / torch.pow(10000, torch.arange(0, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    pe[..., 1::2] = torch.cos(
        pos
        / torch.pow(10000, torch.arange(1, embedding_size, 2, dtype=torch.float32, device=x.device) / embedding_size)
    )
    return pe

TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")

def char_ids_to_str(char_ids: Union[list[int], np.array, torch.Tensor]) -> str:
    """Decode a 1D sequence of character IDs to a string."""
    return "".join([TEXT8_CHARS[i] for i in char_ids])


def batch_to_str(text_batch: Union[list[list], np.array, torch.Tensor]) -> list[str]:
    """Decode a batch of character IDs to a list of strings."""
    return [char_ids_to_str(row_char_ids) for row_char_ids in text_batch]


def batch_to_images(image_batch: torch.Tensor, ncols: int = None) -> plt.Figure:
    if ncols is None:
        ncols = math.ceil(math.sqrt(len(image_batch)))
    if image_batch.size(-1) == 3:  # for color images (CIFAR-10)
        image_batch = (image_batch + 1) / 2
    grid = make_grid(image_batch.permute(0, 3, 1, 2), ncols, pad_value=1).permute(1, 2, 0)
    fig = plt.figure(figsize=(grid.size(1) / 30, grid.size(0) / 30))
    plt.imshow(grid.cpu().clip(min=0, max=1), interpolation="nearest")
    plt.grid(False)
    plt.axis("off")
    return fig

def seed_everything(seed: Optional[int]):
    assert seed is not None
    seed += torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


default_train_config = {
    "meta": {
        "neptune": None,
        "debug": False,
        "root_dir": ".",
    },
    "data": {
        "dataset": "",
        "data_dir": "./data",
    },
    "train_loader": {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
        "drop_last": False,
    },
    "training": {
        "accumulate": 1,
        "checkpoint_dir": "./checkpoints",
        "checkpoint_interval": None,
        "ema_decay": -1,
        "grad_clip_norm": -1,
        "log_interval": 50,
        "max_val_batches": -1,
        "seed": 666,
        "start_step": 1,
        "val_repeats": 1,
    },
}


def make_config(cfg_file: str):
    cli_conf = OmegaConf.load(cfg_file)
    # Start with default config
    cfg = OmegaConf.create(default_train_config)
    # Merge into default config
    cfg = OmegaConf.merge(cfg, cli_conf)
    return cfg

def get_nnet(name, **kwargs):
    data_adapters = {
        "input_adapter": get_adapters(**kwargs['input_adapter']) if kwargs['input_adapter'] else None,
        "output_adapter": get_adapters(**kwargs['output_adapter']) if kwargs['output_adapter'] else None,
    }
    if name == "GPT":
        from networks.transformer import GPT
        return GPT(data_adapters=data_adapters, **kwargs['backbone'])
    elif name  == "UNetModel":
        from networks.unet_improved import UNetModel
        return UNetModel(data_adapters=data_adapters, **kwargs['backbone'])
    elif name == "UNetVDM":
        from networks.unet_vdm import UNetVDM
        return UNetVDM(data_adapters=data_adapters, **kwargs['backbone'])
    else:
        raise NotImplementedError(name)

    
def get_adapters(name, **kwargs):
    if name == "TextInputAdapter":
        from networks.adapters import TextInputAdapter
        return TextInputAdapter(**kwargs)
    elif name == "FourierImageInputAdapter":
        from networks.adapters import FourierImageInputAdapter
        return FourierImageInputAdapter(**kwargs)
    elif name == "OutputAdapter":
        from networks.adapters import OutputAdapter
        return OutputAdapter(**kwargs)
    else:
        raise NotImplementedError(name)
