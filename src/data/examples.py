import torch


def make_examples(
    data: torch.Tensor,
    batch_size=4,
    block_size=8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ixs = torch.randint(
        data.shape[0] - block_size, (batch_size, 1), generator=generator
    )

    Xs = data[ixs + torch.arange(block_size)]
    Ys = data[ixs + torch.arange(block_size) + 1]
    return Xs, Ys
