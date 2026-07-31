from nazuna.models._base import BasicBaseModel
from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer, BatchSeriesNorm, Patchifier
import torch


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
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 16, stride: int = 8, padding_patch: str | None = 'end',
        d_model: int = 128, n_heads: int = 16, d_ff: int = 256, e_layers: int = 2,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2), res_attention: bool = True,
        norm_first: bool = False, norm_out: bool = False,
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        assert seq_len >= patch_len >= stride, 'Expected seq_len >= patch_len >= stride'
        assert d_model % n_heads == 0, 'Expected d_model to be divisible by n_heads'
        assert d_model // n_heads >= 4, 'Expected head_dim >= 4'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.patch_len = patch_len
        self.patchifier = Patchifier(patch_len, stride, padding_patch)
        self.n_patches = self.patchifier.num_patches(seq_len)
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
                norm_first=norm_first,
            )
            for _ in range(e_layers)
        ])
        self.res_attention = res_attention

        self.out_proj = torch.nn.Linear(d_model * self.n_patches, self.pred_len)

        self.norm_out = \
            BatchSeriesNorm(d_model) if norm_first and norm_out else None

    def forward(self, x):
        B, L, C = x.shape
        patches = self.patchifier(x)  # (B, C, P, patch_len)
        P = patches.size(2)
        z = patches.reshape(B * C, P, self.patch_len)  # (B*C, P, patch_len)
        z = self.patch_proj(z)  # (B*C, P, d_model)
        z = z + self.pos_enc.unsqueeze(0)
        z = self.dropout_emb(z)

        scores = None
        for i, layer in enumerate(self.encoder_layers):
            z, scores, x_f1_debug = layer(z, (scores if self.res_attention else None))

            if self.training and i == len(self.encoder_layers) - 1:
                z_f1 = x_f1_debug.reshape(B, C * P, -1)  # (B, C*P, d_ff)
                f1_norm = torch.linalg.vector_norm(z_f1, dim=(1, 2)).mean().item()
                self._debug_if_initial_stage(f'x_f1_shape = {tuple(z_f1.shape)}')
                self._debug(f'x_f1_norm = {f1_norm}')

        if self.norm_out is not None:
            z = self.norm_out(z)

        if self.training:
            z_out = z.reshape(B, C * P, -1)  # (B, C*P, d_model)
            out_norm = torch.linalg.vector_norm(z_out, dim=(1, 2)).mean().item()
            self._debug_if_initial_stage(f'x_out_shape = {tuple(z_out.shape)}')
            self._debug(f'x_out_norm = {out_norm}')
            self._finish_initial_debug_stage()

        z = z.reshape(B, C, P, -1)  # (B, C, P, d_model)
        z = z.transpose(2, 3)  # (B, C, d_model, P)
        z = z.reshape(B, C, -1)  # (B, C, d_model * P)
        y = self.out_proj(z)  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}
