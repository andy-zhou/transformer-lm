import torch
from .encoder import Encoder


def make_splits(data: str, encoder: Encoder) -> tuple[torch.Tensor, torch.Tensor]:
    split_point = int(0.9 * len(data))
    return encoder.encode(data[:split_point]), encoder.encode(data[split_point:])
