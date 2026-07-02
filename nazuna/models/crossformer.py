from nazuna.models._base import BasicBaseModel
from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer, BatchSeriesNorm, Patchifier
from math import ceil
import torch


class _TwoStageAttentionLayer(torch.nn.Module):
    def __init__(
        self, seg_num, factor, d_model, n_heads, d_ff,
        dropout_aw, dropout_sa, dropout_ff, norm_first,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.time_layer = TransformerEncoderLayer(
            MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw),
            d_model=d_model, d_ff=d_ff,
            norm_0=BatchSeriesNorm(d_model),
            norm_1=BatchSeriesNorm(d_model),
            activation=torch.nn.GELU(),
            dropout_sa=dropout_sa, dropout_ff=dropout_ff,
            norm_first=norm_first,
        )
        self.dim_sender = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.dim_receiver = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.router = torch.nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = torch.nn.Dropout(dropout_sa)
        self.norm_2 = BatchSeriesNorm(d_model)
        self.norm_3 = BatchSeriesNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_ff[0]),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout_ff[1]),
        )

    def forward(self, x):
        B, C, S, d = x.shape  # (B, C, seg_num, d_model)
        factor = self.router.size(1)
        time_in = x.reshape(B * C, S, d)
        dim_in, _ = self.time_layer(time_in)  # (B*C, S, d_model)

        dim_send = dim_in.reshape(B, C, S, d).permute(0, 2, 1, 3)
        dim_send = dim_send.reshape(B * S, C, d)  # (B*S, C, d_model)
        router = self.router.unsqueeze(0).expand(B, -1, -1, -1)
        router = router.reshape(B * S, factor, d)  # (B*S, factor, d_model)
        if not self.norm_first:
            dim_buffer, _ = self.dim_sender(router, dim_send, dim_send)
            dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
            dim_enc = self.norm_2(dim_send + self.dropout(dim_receive))
            dim_enc = self.norm_3(dim_enc + self.dropout(self.mlp(dim_enc)))
        else:
            h = self.norm_2(dim_send)
            dim_buffer, _ = self.dim_sender(router, h, h)
            dim_receive, _ = self.dim_receiver(h, dim_buffer, dim_buffer)
            dim_enc = dim_send + self.dropout(dim_receive)
            dim_enc = dim_enc + self.dropout(self.mlp(self.norm_3(dim_enc)))

        out = dim_enc.reshape(B, S, C, d).permute(0, 2, 1, 3)  # (B, C, S, d_model)
        return out


class _SegMerging(torch.nn.Module):
    def __init__(self, d_model, win_size):
        super().__init__()
        self.win_size = win_size
        self.norm = torch.nn.LayerNorm(win_size * d_model)
        self.linear_trans = torch.nn.Linear(win_size * d_model, d_model)

    def forward(self, x):
        B, C, S, d = x.shape  # (B, C, seg_num, d_model)
        pad_num = S % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat([x, x[:, :, -1:, :].repeat(1, 1, pad_num, 1)], dim=-2)
        seg = [x[:, :, i::self.win_size, :] for i in range(self.win_size)]
        x = torch.cat(seg, dim=-1)  # (B, C, ceil(S/win), win*d_model)
        x = self.norm(x)
        x = self.linear_trans(x)
        return x


class _ScaleBlock(torch.nn.Module):
    def __init__(
        self, win_size, seg_num, factor, d_model, n_heads, d_ff,
        dropout_aw, dropout_sa, dropout_ff, norm_first,
    ):
        super().__init__()
        self.merge = _SegMerging(d_model, win_size) if win_size > 1 else None
        self.attn = _TwoStageAttentionLayer(
            seg_num, factor, d_model, n_heads, d_ff,
            dropout_aw, dropout_sa, dropout_ff, norm_first,
        )

    def forward(self, x):
        if self.merge is not None:
            x = self.merge(x)
        return self.attn(x)


class _Encoder(torch.nn.Module):
    def __init__(
        self, e_layers, win_size, in_seg_num, factor, d_model, n_heads, d_ff,
        dropout_aw, dropout_sa, dropout_ff, norm_first,
    ):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(_ScaleBlock(
            1, in_seg_num, factor, d_model, n_heads, d_ff,
            dropout_aw, dropout_sa, dropout_ff, norm_first,
        ))
        seg_num = in_seg_num
        for _ in range(1, e_layers):
            seg_num = ceil(seg_num / win_size)
            self.blocks.append(_ScaleBlock(
                win_size, seg_num, factor, d_model, n_heads, d_ff,
                dropout_aw, dropout_sa, dropout_ff, norm_first,
            ))

    def forward(self, x):
        outs = [x]
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        return outs


class _DecoderLayer(torch.nn.Module):
    def __init__(
        self, patch_len, out_seg_num, factor, d_model, n_heads, d_ff,
        dropout_aw, dropout_sa, dropout_ff, norm_first,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.self_attn = _TwoStageAttentionLayer(
            out_seg_num, factor, d_model, n_heads, d_ff,
            dropout_aw, dropout_sa, dropout_ff, norm_first,
        )
        self.cross_attn = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.dropout = torch.nn.Dropout(dropout_sa)
        self.norm_0 = BatchSeriesNorm(d_model)
        self.norm_1 = BatchSeriesNorm(d_model)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.linear_pred = torch.nn.Linear(d_model, patch_len)

    def forward(self, x, cross):
        B, C, S, d = x.shape  # x: (B, C, out_seg_num, d_model)
        x = self.self_attn(x)
        x = x.reshape(B * C, S, d)
        cross = cross.reshape(B * C, cross.size(2), d)
        if not self.norm_first:
            tmp, _ = self.cross_attn(x, cross, cross)
            x = self.norm_0(x + self.dropout(tmp))
            x = self.norm_1(x + self.mlp(x))
        else:
            tmp, _ = self.cross_attn(self.norm_0(x), cross, cross)
            x = x + self.dropout(tmp)
            x = x + self.mlp(self.norm_1(x))
        dec = x.reshape(B, C, S, d)
        pred = self.linear_pred(x).reshape(B, C, S, -1)  # (B, C, out_seg_num, patch_len)
        return dec, pred


class _Decoder(torch.nn.Module):
    def __init__(
        self, n_layers, patch_len, pred_len, out_seg_num, factor,
        d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff, norm_first,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.layers = torch.nn.ModuleList([
            _DecoderLayer(
                patch_len, out_seg_num, factor, d_model, n_heads, d_ff,
                dropout_aw, dropout_sa, dropout_ff, norm_first,
            )
            for _ in range(n_layers)
        ])

    def forward(self, x, cross_list):
        final = None
        for layer, cross in zip(self.layers, cross_list):
            x, pred = layer(x, cross)
            final = pred if final is None else final + pred
        B, C, S, pl = final.shape  # (B, C, out_seg_num, patch_len)
        final = final.permute(0, 2, 3, 1).reshape(B, S * pl, C)
        return final[:, :self.pred_len, :]


class Crossformer(BasicBaseModel):
    """
    !!! info "Original Research"
        This model is based on the following research:
        > Yunhao Zhang and Junchi Yan.
          "Crossformer: Transformer Utilizing Cross-Dimension Dependency
          for Multivariate Time Series Forecasting."
          In International Conference on Learning Representations (ICLR), 2023.
          [Paper](https://openreview.net/forum?id=vSVLM2j9eie) |
          [GitHub](https://github.com/Thinklab-SJTU/Crossformer)
    """
    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 12, stride: int | None = None,
        padding_patch: str | None = 'end',
        win_size: int = 4, factor: int = 10,
        d_model: int = 256, n_heads: int = 4, d_ff: int = 512, e_layers: int = 3,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2), norm_first: bool = False,
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        if stride is None:
            stride = patch_len
        assert d_model % n_heads == 0, 'Expected d_model to be divisible by n_heads'
        assert d_model // n_heads >= 4, 'Expected head_dim >= 4'
        assert patch_len >= stride >= 1, 'Expected patch_len >= stride >= 1'
        assert win_size >= 1, 'Expected win_size >= 1'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.patchifier = Patchifier(patch_len, stride, padding_patch)
        self.in_seg_num = self.patchifier.num_patches(seq_len)
        self.out_seg_num = ceil(pred_len / patch_len)

        self.value_embed = torch.nn.Linear(patch_len, d_model)
        self.enc_pos = torch.nn.Parameter(
            torch.randn(1, c_in, self.in_seg_num, d_model)
        )
        self.pre_norm = torch.nn.LayerNorm(d_model)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.encoder = _Encoder(
            e_layers, win_size, self.in_seg_num, factor, d_model, n_heads, d_ff,
            dropout_aw, dropout_sa, dropout_ff, norm_first,
        )
        self.dec_pos = torch.nn.Parameter(
            torch.randn(1, c_in, self.out_seg_num, d_model)
        )
        self.decoder = _Decoder(
            e_layers + 1, patch_len, pred_len, self.out_seg_num, factor,
            d_model, n_heads, d_ff, dropout_aw, dropout_sa, dropout_ff, norm_first,
        )

    def forward(self, x):
        B, L, C = x.shape
        patches = self.patchifier(x)  # (B, C, in_seg_num, patch_len)
        z = self.value_embed(patches)  # (B, C, in_seg_num, d_model)
        z = z + self.enc_pos
        z = self.pre_norm(z)
        z = self.dropout_emb(z)

        enc_out = self.encoder(z)  # list of (B, C, S_i, d_model)
        dec_in = self.dec_pos.expand(B, -1, -1, -1)  # (B, C, out_seg_num, d_model)
        y = self.decoder(dec_in, enc_out)  # (B, pred_len, C)
        return y, {}
