from nazuna.models._base import BasicBaseModel
from nazuna.models.common import BatchSeriesNorm, MultiheadAttention
import torch
import torch.nn.functional as F


class _DispatcherAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_aw=0.1):
        super().__init__()
        self.aggregate = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.distribute = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)

    def forward(self, x, dispatcher):
        # x: (B, N, d_model), dispatcher: (B, k, d_model)
        d, _ = self.aggregate(dispatcher, x, x)  # (B, k, d_model)
        out, _ = self.distribute(x, d, d)  # (B, N, d_model)
        return out


class _UniTSTBlock(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ff, dropout_aw=0.1, dropout_sa=0.1,
        dropout_ff=(0.0, 0.1),
    ):
        super().__init__()
        self.attn = _DispatcherAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.dropout_sa = torch.nn.Dropout(dropout_sa)
        self.norm_0 = BatchSeriesNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_ff[0]),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout_ff[1]),
        )
        self.norm_1 = BatchSeriesNorm(d_model)

    def forward(self, x, dispatcher):
        x_save = x
        x = self.attn(x, dispatcher)
        x = self.dropout_sa(x)
        x = x_save + x
        x = self.norm_0(x)
        x_save = x
        x = self.ff(x)
        x = x_save + x
        x = self.norm_1(x)
        return x


class UniTSTLike(BasicBaseModel):
    """
    !!! info "Original Research"
        This model is based on the following research:
        > Juncheng Liu, Chenghao Liu, Gerald Woo, Yiwei Wang, Bryan Hooi,
          Caiming Xiong, and Doyen Sahoo.
          "UniTST: Effectively Modeling Inter-Series and Intra-Series Dependencies
          for Multivariate Time Series Forecasting."
          Transactions on Machine Learning Research (TMLR), 2025.
          [Paper](https://openreview.net/forum?id=p3y5q4cvzV) |
          [arXiv](https://arxiv.org/abs/2406.04975)

        The official source code was not publicly available at the time of writing,
        so this implementation follows the description in the paper. Activation
        function and dropout placement are not specified in the paper and follow
        choices common in PatchTST-style models.
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 16, stride: int = 8, padding_patch: str | None = 'end',
        d_model: int = 128, n_heads: int = 8, d_ff: int = 256, e_layers: int = 2,
        n_dispatchers: int = 8,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2),
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        assert seq_len >= patch_len, 'seq_len >= patch_len'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.n_patches = (seq_len - self.patch_len) // self.stride + 1
        if self.padding_patch == 'end':
            self.n_patches += 1

        self.patch_proj = torch.nn.Linear(self.patch_len, d_model)
        # Learnable 2D positional encoding shared across the batch.
        self.pos_enc = torch.nn.Parameter(
            torch.empty(c_in, self.n_patches, d_model)
        )
        torch.nn.init.uniform_(self.pos_enc, -0.02, 0.02)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        # Learnable dispatcher embeddings shared across the batch.
        self.dispatcher = torch.nn.Parameter(torch.empty(n_dispatchers, d_model))
        torch.nn.init.uniform_(self.dispatcher, -0.02, 0.02)

        self.blocks = torch.nn.ModuleList([
            _UniTSTBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout_aw=dropout_aw, dropout_sa=dropout_sa, dropout_ff=dropout_ff,
            )
            for _ in range(e_layers)
        ])

        self.out_proj = torch.nn.Linear(d_model * self.n_patches, self.pred_len)

    def _patchify(self, x):
        x = x.transpose(1, 2)  # (B, L_in, C) -> (B, C, L_in)
        if self.padding_patch == 'end':
            x = F.pad(x, (0, self.stride), mode='replicate')
        # x: (B, C, L_in + stride) -> (B, C, P, patch_len)
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        B, L, C = x.shape
        patches = self._patchify(x)  # (B, C, P, patch_len)
        P = patches.size(2)
        z = self.patch_proj(patches)  # (B, C, P, d_model)
        z = z + self.pos_enc.unsqueeze(0)  # broadcast (1, C, P, d_model)
        z = self.dropout_emb(z)
        z = z.reshape(B, C * P, -1)  # (B, C*P, d_model)

        dispatcher = self.dispatcher.unsqueeze(0).expand(B, -1, -1)
        for block in self.blocks:
            z = block(z, dispatcher)

        z = z.reshape(B, C, P, -1)  # (B, C, P, d_model)
        z = z.reshape(B, C, -1)  # (B, C, P * d_model)
        y = self.out_proj(z)  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}
