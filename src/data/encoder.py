import torch


class Encoder:
    def __init__(self, chars: str):
        self.chars = sorted(list(set(chars)))
        self.vocab_size = len(self.chars)
        self.stoi = {v: i for i, v in enumerate(self.chars)}

    def encode(self, chars: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in chars], dtype=torch.long)

    def decode(self, encoded: torch.Tensor) -> str:
        return "".join(self.chars[i] for i in encoded.tolist())