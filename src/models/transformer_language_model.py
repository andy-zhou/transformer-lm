import torch
import torch.nn as nn
import torch.nn.functional as F
from ..data import Encoder, make_examples

N_KEY = 16
N_VALUE = 16
BLOCK_SIZE = 32
ATTENTION_HEADS = 4
N_EMBED = 64
N_HIDDEN = 128


class MultiHeadAttention(nn.Module):
    attention_mask: torch.Tensor

    def __init__(
        self,
        n_embed=N_EMBED,
        n_key=N_KEY,
        n_value=N_VALUE,
        block_size=BLOCK_SIZE,
        attention_heads=ATTENTION_HEADS,
    ):
        super().__init__()
        self.query_projection = nn.Linear(n_embed, n_key * attention_heads, bias=False)
        self.key_projection = nn.Linear(n_embed, n_key * attention_heads, bias=False)
        self.value_projection = nn.Linear(
            n_embed, n_value * attention_heads, bias=False
        )
        self.embed_projection = nn.Linear(
            n_value * attention_heads, n_embed, bias=False
        )

        self.attention_heads = attention_heads
        self.n_key = n_key
        self.register_buffer(
            "attention_mask", torch.tril(torch.ones((block_size, block_size))) == 0
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        H = self.attention_heads

        query = self.query_projection(x).view(B, T, H, -1)
        key = self.key_projection(x).view(B, T, H, -1)
        value = self.value_projection(x).view(B, T, H, -1)

        assert isinstance(query, torch.Tensor)
        assert isinstance(key, torch.Tensor)
        assert isinstance(value, torch.Tensor)

        attention_weights = torch.einsum("bihc,bjhc->bhij", query, key)
        attention_weights = attention_weights * self.n_key**-0.5

        attention_weights = attention_weights.masked_fill(
            self.attention_mask[:T, :T], float("-inf")
        )
        attention_weights = F.softmax(attention_weights, dim=-1)

        weighted_values = torch.einsum(
            "bihc,bhti->bthc", value, attention_weights
        ).contiguous()

        weighted_values = weighted_values.view(B, T, -1)
        out = self.embed_projection(weighted_values)

        return out


class FeedForward(nn.Module):
    def __init__(self, n_hidden=N_HIDDEN, n_embed=N_EMBED):
        super().__init__()
        self.inner = nn.Linear(n_embed, n_hidden)
        self.outer = nn.Linear(n_hidden, n_embed)

    def forward(self, x: torch.Tensor):
        return self.outer(F.relu(self.inner(x)))


class AttentionBlock(nn.Module):
    def __init__(
        self,
        n_embed=N_EMBED,
        n_key=N_KEY,
        n_value=N_VALUE,
        n_hidden=N_HIDDEN,
        attention_heads=ATTENTION_HEADS,
        block_size=BLOCK_SIZE,
    ):
        super().__init__()
        self.attention_ln = nn.LayerNorm(n_embed)
        self.multi_head_attention = MultiHeadAttention(
            n_embed=n_embed,
            n_key=n_key,
            n_value=n_value,
            attention_heads=attention_heads,
            block_size=block_size,
        )
        self.feed_forward_ln = nn.LayerNorm(n_embed)
        self.feed_forward = FeedForward(n_hidden=n_hidden, n_embed=n_embed)

    def forward(self, x: torch.Tensor):
        # A deviation from Attention is All You Need - LayerNorm is now typically done
        # outside of the residual block (https://arxiv.org/pdf/2002.04745)
        x = x + self.multi_head_attention(self.attention_ln(x))
        x = x + self.feed_forward(self.feed_forward_ln(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        n_embed=N_EMBED,
        n_key=N_KEY,
        n_value=N_VALUE,
        n_hidden=N_HIDDEN,
        block_size=BLOCK_SIZE,
        attention_heads=ATTENTION_HEADS,
        layers=2,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(encoder.vocab_size, n_embed)
        self.positional_embeddings = nn.Embedding(block_size, n_embed)
        self.attention_layers = nn.Sequential(
            *[
                AttentionBlock(
                    n_embed=n_embed,
                    n_key=n_key,
                    n_value=n_value,
                    n_hidden=n_hidden,
                    attention_heads=attention_heads,
                    block_size=block_size,
                )
                for _ in range(layers)
            ]
        )
        self.final_ln = nn.LayerNorm(n_embed)
        self.logit_projection = nn.Linear(n_embed, encoder.vocab_size)
        self.block_size = block_size

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape

        tok_emb = self.token_embeddings(x)
        pos_emb = self.positional_embeddings(torch.arange(T))
        x = tok_emb + pos_emb

        x = self.attention_layers(x)
        logits = self.logit_projection(self.final_ln(x))
        assert isinstance(logits, torch.Tensor)
        loss = (
            None if targets is None else F.cross_entropy(logits.movedim(-1, 1), targets)
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor | None = None,
        num_characters=100,
        generator: torch.Generator | None = None,
        device="cpu",
    ) -> torch.Tensor:
        idx = (
            torch.zeros((1, 1), dtype=torch.long, device=device)
            if context is None
            else context
        )

        for _ in range(num_characters):
            block = idx[:, -self.block_size :]
            logits, _ = self(block)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_char = torch.multinomial(
                probs, num_samples=1, generator=generator
            )  # (B, idx)
            idx = torch.cat([idx, next_char], dim=-1)

        return idx[0, 1:] if context is None else idx


@torch.no_grad()
def eval_transformer_model(
    model: TransformerLanguageModel,
    data: torch.Tensor,
    batch_size=32,
    block_size=BLOCK_SIZE,
) -> torch.Tensor:
    Xb, Yb = make_examples(data, batch_size=batch_size, block_size=block_size)
    _, loss = model(Xb, targets=Yb)
    return loss.mean()


def train_transformer_model(
    training_set: torch.Tensor,
    validation_set: torch.Tensor,
    encoder: Encoder,
    batch_size=32,
    n_embed=N_EMBED,
    n_key=N_KEY,
    n_value=N_VALUE,
    n_hidden=N_HIDDEN,
    block_size=BLOCK_SIZE,
    attention_heads=ATTENTION_HEADS,
    layers=2,
    num_iters: int = 10000,
    log_every_n: int = 2000,
    lr=1e-3,
    eval_batches=6400,
    device: str = "cpu",
) -> TransformerLanguageModel:
    model = TransformerLanguageModel(
        encoder,
        n_embed=n_embed,
        n_key=n_key,
        n_value=n_value,
        n_hidden=n_hidden,
        attention_heads=attention_heads,
        layers=layers,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for i in range(num_iters):
        if i % log_every_n == 0:
            model.eval()
            train_loss = eval_transformer_model(
                model, training_set, eval_batches, BLOCK_SIZE
            ).item()
            val_loss = eval_transformer_model(
                model, validation_set, eval_batches, BLOCK_SIZE
            ).item()
            print(f"Iteration {i:>6d}/{num_iters}: {train_loss=:.4f}, {val_loss=:.4f}")

        Xb, Yb = make_examples(
            training_set, batch_size=batch_size, block_size=block_size
        )
        Xb = Xb.to(device)
        Yb = Yb.to(device)
        _, loss = model(Xb, targets=Yb)
        assert isinstance(loss, torch.Tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
