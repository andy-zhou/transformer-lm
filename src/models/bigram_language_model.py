import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import Encoder, make_examples


# Bigram model for baseline
class BigramLanguageModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        n_embed: int = 32,
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(encoder.vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, encoder.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,  # B, T
        targets: torch.Tensor | None = None,  # B, T
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        emb: torch.Tensor = self.token_embedding_table(idx)  # (B, T, C)
        logits: torch.Tensor = self.lm_head(emb)  # (B, T, logits)

        if targets is None:
            loss = None
        else:
            logit_view = logits.movedim(logits.dim() - 1, 1)  # (B, C, T)
            loss = F.cross_entropy(logit_view, targets)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor | None = None,  # (B, idx)
        num_characters=100,
        generator: torch.Generator | None = None,
        device: str = "cpu",
    ):
        idx = (
            torch.zeros((1, 1), dtype=torch.long, device=device)
            if context is None
            else context
        )  # (B, idx)

        for _ in range(num_characters):
            last_char = idx[:, -1]  # (B, 1)
            logits: torch.Tensor
            logits, _ = self(last_char)  # (B, 1, logits)
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_char = torch.multinomial(
                probs, num_samples=1, generator=generator
            )  # (B, idx)
            idx = torch.cat([idx, next_char], dim=-1)

        return idx[0, 1:] if context is None else idx


@torch.no_grad()
def eval_bigram_model(
    model: BigramLanguageModel,
    data: torch.Tensor,
    batch_size=32,
    block_size=8,
) -> torch.Tensor:
    Xb, Yb = make_examples(data, batch_size=batch_size, block_size=block_size)
    _, loss = model(Xb, targets=Yb)
    return loss.mean()


def train_bigram_model(
    training_set: torch.Tensor,
    validation_set: torch.Tensor,
    encoder: Encoder,
    batch_size=32,
    block_size=8,
    num_iters: int = 10000,
    log_every_n: int = 2000,
    eval_batches=6400,
    device: str = "cpu",
) -> BigramLanguageModel:
    model = BigramLanguageModel(encoder)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for i in range(num_iters):
        if i % log_every_n == 0:
            model.eval()
            train_loss = eval_bigram_model(model, training_set, eval_batches).item()
            val_loss = eval_bigram_model(model, validation_set, eval_batches).item()
            print(f"Iteration {i:>6d}/{num_iters}: {train_loss=:.4f}, {val_loss=:.4f}")

        Xb, Yb = make_examples(
            training_set, batch_size=batch_size, block_size=block_size
        )
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        loss: torch.Tensor
        _, loss = model(Xb, targets=Yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
