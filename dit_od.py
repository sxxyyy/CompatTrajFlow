from torch import Tensor, nn

from dit import DiT


class ODConditionedDiT(DiT):
    """DiT conditioned only on an always-present origin-destination pair."""

    def __init__(self, *args, emb_dim: int = 128, **kwargs):
        super().__init__(*args, emb_dim=emb_dim, **kwargs)
        self.emb_dim = emb_dim
        self.cond_embedder = nn.Sequential(
            nn.Linear(2 * emb_dim, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        nn.init.normal_(self.cond_embedder[0].weight, std=0.02)
        nn.init.zeros_(self.cond_embedder[0].bias)
        nn.init.normal_(self.cond_embedder[2].weight, std=0.02)
        nn.init.zeros_(self.cond_embedder[2].bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        if cond.size(-1) != 2 * self.emb_dim:
            raise ValueError(
                f"ODConditionedDiT expects OD dimension {2 * self.emb_dim}, "
                f"received {cond.size(-1)}"
            )
        return super().forward(x, t, cond)
