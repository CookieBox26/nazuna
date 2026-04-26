from nazuna.models._base import BasicBaseModel
import math
import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.pe = torch.nn.Parameter(
            torch.empty(max_len, d_model)
        )  # [max_len, d_model]
        torch.nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):  # x: [B, S, D]
        s = x.size(1)
        return self.dropout(x + self.pe[:s].unsqueeze(0))


class _MultiheadAttention(torch.nn.Module):
    def __init__(
        self, d_model, n_heads,
        attn_dropout=0., proj_dropout=0., res_attention=True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.res_attention = res_attention

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.Dropout(proj_dropout),
        )
        self.attn_dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, x, prev=None):  # x: [B, L, D]
        B, L, _ = x.shape
        H, dh = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, dh).transpose(1, 2)  # [B, H, L, dh]
        k = self.k_proj(x).view(B, L, H, dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, dh).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dh)
        if prev is not None:
            attn_scores = attn_scores + prev

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)  # [B, H, L, dh]
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.out_proj(output)

        if self.res_attention:
            return output, attn_scores
        return output


class TSTEncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ff,
        attn_dropout=0., dropout=0.1, res_attention=True,
    ):
        super().__init__()
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention,
        )
        self.dropout_attn = torch.nn.Dropout(dropout)
        self.norm_attn = torch.nn.BatchNorm1d(d_model)

        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )
        self.norm_ff = torch.nn.BatchNorm1d(d_model)

    def forward(self, x, prev=None):  # x: [B, S, D]
        if self.res_attention:
            attn_out, scores = self.self_attn(x, prev=prev)
        else:
            attn_out = self.self_attn(x)
        x = x + self.dropout_attn(attn_out)
        # BatchNorm1d expects [B, D, S]
        x = self.norm_attn(x.transpose(1, 2)).transpose(1, 2)

        x = x + self.ff(x)
        x = self.norm_ff(x.transpose(1, 2)).transpose(1, 2)
        if self.res_attention:
            return x, scores
        return x


class PatchTST(BasicBaseModel):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.
          "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
          In International Conference on Learning Representations (ICLR), 2023.
          [Paper](https://openreview.net/forum?id=Jbdc0vTOcol) |
          [arXiv](https://arxiv.org/abs/2211.14730) |
          [GitHub](https://github.com/yuqinie98/PatchTST)
    """
    def _get_seq_len_for_model(self, seq_len):
        return seq_len

    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 16, stride: int = 8, padding_patch: str | None = 'end',
        d_model: int = 128, n_heads: int = 16, n_layers: int = 3,
        d_ff: int = 256, dropout: float = 0.2,
        attn_dropout: float = 0.0, head_dropout: float = 0.0, res_attention: bool = True,
        revin: bool = True, revin_affine: bool = True, revin_eps: float = 1e-5,
        scaler_cls: type | None = None,
        scaler_params: dict | None = None,
    ) -> None:
        super()._setup(seq_len, pred_len, scaler_cls, scaler_params)
        seq_len_for_model = self._get_seq_len_for_model(seq_len)

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.head_dropout = head_dropout
        self.res_attention = res_attention
        self.use_revin = revin
        self.revin_affine = revin_affine
        self.revin_eps = revin_eps

        if self.use_revin and self.revin_affine:
            self.revin_affine_weight = torch.nn.Parameter(torch.ones(c_in))
            self.revin_affine_bias = torch.nn.Parameter(torch.zeros(c_in))

        assert seq_len_for_model >= self.patch_len, 'seq_len >= patch_len'
        self.n_patches = (seq_len_for_model - self.patch_len) // self.stride + 1
        if self.padding_patch == 'end':
            self.n_patches += 1

        self.patch_proj = torch.nn.Linear(self.patch_len, self.d_model)

        self.pos = PositionalEncoding(
            self.d_model,
            max_len=self.n_patches,
            dropout=self.dropout,
        )

        self.encoder = torch.nn.ModuleList([
            TSTEncoderLayer(
                self.d_model, self.n_heads,
                self.d_ff,
                attn_dropout=self.attn_dropout,
                dropout=self.dropout,
                res_attention=self.res_attention,
            )
            for _ in range(self.n_layers)
        ])
        self.head = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=-2),
            torch.nn.Linear(self.d_model * self.n_patches, self.pred_len),
            torch.nn.Dropout(self.head_dropout),
        )

    def _patchify(self, x):  # x: [B, L, C] -> [B, C, P, patch_len]
        x = x.transpose(1, 2)  # [B, C, L]
        if self.padding_patch == 'end':
            x = F.pad(x, (0, self.stride), mode='replicate')
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        B, L, C = x.shape

        # RevIN: instance normalization (per-sample, per-channel)
        if self.use_revin:
            ri_mean = x.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
            ri_std = torch.sqrt(
                x.var(dim=1, keepdim=True, unbiased=False) + self.revin_eps
            ).detach()  # [B, 1, C]
            x = (x - ri_mean) / ri_std
            if self.revin_affine:
                x = x * self.revin_affine_weight + self.revin_affine_bias

        patches = self._patchify(x)  # [B, C, P, pl]
        P = patches.size(2)
        z = patches.reshape(B * C, P, self.patch_len)  # [B*C, P, pl]
        z = self.patch_proj(z)  # [B*C, P, D]
        z = self.pos(z)  # [B*C, P, D]
        if self.res_attention:
            scores = None
            for layer in self.encoder:
                z, scores = layer(z, prev=scores)
        else:
            for layer in self.encoder:
                z = layer(z)

        z = z.view(B, C, P, self.d_model).transpose(2, 3)  # [B, C, D, P]
        yhat = self.head(z)  # [B, C, pred_len]
        yhat = yhat.transpose(1, 2)  # [B, H, C]

        # RevIN: de-normalize
        if self.use_revin:
            if self.revin_affine:
                yhat = (yhat - self.revin_affine_bias) / (
                    self.revin_affine_weight + self.revin_eps ** 2
                )
            yhat = yhat * ri_std + ri_mean

        return yhat, {}


class DiffPatchTST(PatchTST):
    def _get_seq_len_for_model(self, seq_len):
        return seq_len - 1

    def forward(self, x):  # x: [B, seq_len, C] (scaled)
        last_val = x[:, -1:, :]  # [B, 1, C]
        dx = x[:, 1:, :] - x[:, :-1, :]  # [B, seq_len-1, C]
        pred_dx, info = super().forward(dx)  # [B, pred_len, C]
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
