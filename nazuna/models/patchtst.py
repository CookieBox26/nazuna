from nazuna.models._base import BasicBaseModel
from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer, BatchSeriesNorm
import torch
import torch.nn.functional as F


class PatchTST(BasicBaseModel):
    """
    !!! info "Original Research"
        This model is based on the following research:
        > Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam.
          "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
          In International Conference on Learning Representations (ICLR), 2023.
          [Paper](https://openreview.net/forum?id=Jbdc0vTOcol) |
          [arXiv](https://arxiv.org/abs/2211.14730) |
          [GitHub](https://github.com/yuqinie98/PatchTST)

    !!! tip "Example parameter configurations"
        ```toml
        [definitions.PatchTST]
        cls_path = "nazuna.models.patchtst.PatchTST"
        [definitions.PatchTST.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        c_in = 10  # task-dependent
        patch_len = 16
        stride = 8
        padding_patch = "end"
        d_model = 128
        n_heads = 16
        d_ff = 256
        e_layers = 2
        dropout_emb = 0.1
        dropout_aw = 0.1
        dropout_sa = 0.1
        dropout_ff = [ 0.0, 0.2,]
        res_attention = true
        scaler_cls_path = ""
        scaler_params = {}
        prep_type = "none"
        use_revin = true
        revin_affine = false
        revin_eps = 1e-5
        ```
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 16, stride: int = 8, padding_patch: str | None = 'end',
        d_model: int = 128, n_heads: int = 16, d_ff: int = 256, e_layers: int = 2,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2), res_attention: bool = True,
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
    ) -> None:
        assert seq_len >= patch_len, 'seq_len >= patch_len'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
        )

        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.n_patches = (seq_len - self.patch_len) // self.stride + 1
        if self.padding_patch == 'end':
            self.n_patches += 1
        self.patch_proj = torch.nn.Linear(self.patch_len, d_model)
        self.pos_enc = torch.nn.Parameter(torch.empty(self.n_patches, d_model))
        torch.nn.init.uniform_(self.pos_enc, -0.02, 0.02)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.encoder_layers = torch.nn.ModuleList([
            TransformerEncoderLayer(
                MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw),
                d_model=d_model, d_ff=d_ff,
                norm_0=BatchSeriesNorm(d_model),
                norm_1=BatchSeriesNorm(d_model),
                activation=torch.nn.GELU(),
                dropout_sa=dropout_sa, dropout_ff=dropout_ff,
            )
            for _ in range(e_layers)
        ])
        self.res_attention = res_attention

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
        z = patches.reshape(B * C, P, self.patch_len)  # (B*C, P, patch_len)
        z = self.patch_proj(z)  # (B*C, P, d_model)
        z = z + self.pos_enc.unsqueeze(0)
        z = self.dropout_emb(z)

        scores = None
        for layer in self.encoder_layers:
            z, scores = layer(z, (scores if self.res_attention else None))

        z = z.reshape(B, C, P, -1)  # (B, C, P, d_model)
        z = z.transpose(2, 3)  # (B, C, d_model, P)
        z = z.reshape(B, C, -1)  # (B, C, d_model * P)
        y = self.out_proj(z)  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}
