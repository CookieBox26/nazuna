from nazuna.models._base import BasicBaseModel
from nazuna.models.common import IqrScaler, TimeFeatureEmbedding, SeriesDecomp
import numpy as np
import torch
import math


class TokenNormTimeDemean(torch.nn.Module):
    """
    Normalize each token, then remove mean over time
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.norm(x)
        return x - x.mean(dim=1, keepdim=True)


class ConvEmb(torch.nn.Module):
    def __init__(
        self, c_in, c_out, kernel_size, bias=False,
        keep_len=True, padding_mode='circular',
    ):
        padding = 0
        if keep_len:
            assert kernel_size % 2 == 1, 'Kernel size must be odd.'
            padding = (kernel_size - 1) // 2
        super().__init__()
        self.conv1d = torch.nn.Conv1d(
            in_channels=c_in, out_channels=c_out, kernel_size=kernel_size,
            bias=bias, padding=padding, padding_mode=padding_mode,
        )
    def forward(self, x):  # x: [B, L, C_in]
        x = x.permute(0, 2, 1)  # [B, C_in, L]
        x = self.conv1d(x)  # [B, C_out, L']
        return x.transpose(2, 1)  # [B, L', C_out]


class AutoCorrelationLayer(torch.nn.Module):
    def __init__(
        self, d_model: int, n_heads: int,
        topk_factor: float = 1.0, dropout: float = 0.1,
        independent_heads: bool = False,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.topk_factor = topk_factor
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.independent_heads = independent_heads

    def _lagged_aggregation_approx(self, b, mean_corr, topk, v):
        # mean_corr: [B, L] or [B, H, L]
        _, indices = torch.topk(
            mean_corr.mean(dim=0, keepdim=True), topk, dim=-1,
        )  # [1, (H,) topk]
        indices = indices.expand(*(mean_corr.shape[:-1] + (-1,))) # [B, (H,) topk]
        w = torch.gather(mean_corr, dim=-1, index=indices)  # [B, (H,) topk]
        w = torch.softmax(w, dim=-1)
        w = self.dropout(w)
        agg = torch.zeros_like(v)
        if self.independent_heads:
            for h in range(self.n_heads):
                for i in range(topk):
                    shifted = \
                        torch.roll(v[:, h, :, :], -int(indices[0, h, i]), dims=1)
                    agg[:, h, :, :] = \
                        agg[:, h, :, :] + shifted * w[:, h, i].view(b, 1, 1)
        else:
            for i in range(topk):
                shifted = torch.roll(v, -int(indices[0, i]), dims=2)
                agg = agg + shifted * w[:, i].view(b, 1, 1, 1)
        return agg

    def _lagged_aggregation(self, b, mean_corr, topk, v, l):
        # mean_corr: [B, L] or [B, H, L]
        # w, indices: [B, topk] or [B, H, topk]
        w, indices = torch.topk(mean_corr, topk, dim=-1)
        w = torch.softmax(w, dim=-1)
        w = self.dropout(w)
        agg = torch.zeros_like(v)
        v_rep = torch.cat([v, v], dim=2)  # [B, H, 2L, d_head]
        init_index = (
            torch.arange(l, device=v.device).view(1, 1, l, 1)
            .expand(b, self.n_heads, l, self.d_head)
        )
        for i in range(topk):
            idx = init_index + indices[..., i].view(b, -1, 1, 1).expand(
                b, self.n_heads, l, self.d_head
            )
            shifted = v_rep.gather(dim=2, index=idx)
            agg = agg + shifted * w[..., i].view(b, -1, 1, 1)
        return agg

    def _autocorrelation(self, q, k, v, l):
        b = q.shape[0]
        qf = torch.fft.rfft(q, dim=2)
        kf = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=l, dim=2)
        topk = max(1, int(self.topk_factor * math.log(l + 1)))

        if self.independent_heads:
            mean_corr = corr.mean(dim=-1)  # [B, H, L]
            if self.training:
                agg = self._lagged_aggregation_approx(b, mean_corr, topk, v)
            else:
                agg = self._lagged_aggregation(b, mean_corr, topk, v, l)
        else:
            mean_corr = corr.mean(dim=1).mean(dim=-1)  # [B, L]
            if self.training:
                agg = self._lagged_aggregation_approx(b, mean_corr, topk, v)
            else:
                agg = self._lagged_aggregation(b, mean_corr, topk, v, l)
        return agg

    def _reshape_heads(self, x):
        # [B, L, D] -> [B, H, L, d_head]
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q_in, k_in, v_in):
        # q_in: [B, Lq, D]; k_in, v_in: [B, Lk, D].
        # When Lq != Lk, align lengths by zero-padding or truncating k/v to Lq
        # (official Autoformer behavior for cross-correlation).
        b, lq, _ = q_in.shape
        lk = k_in.shape[1]
        if lk < lq:
            pad = torch.zeros(
                b, lq - lk, self.d_model,
                device=k_in.device, dtype=k_in.dtype,
            )
            k_in = torch.cat([k_in, pad], dim=1)
            v_in = torch.cat([v_in, pad], dim=1)
        elif lk > lq:
            k_in = k_in[:, :lq, :]
            v_in = v_in[:, :lq, :]
        l = lq
        q = self._reshape_heads(self.q_proj(q_in))
        k = self._reshape_heads(self.k_proj(k_in))
        v = self._reshape_heads(self.v_proj(v_in))
        agg = self._autocorrelation(q, k, v, l)
        agg = agg.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.out_proj(agg)


class EncoderLayer(torch.nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int,
        decomp_kernel: int, n_moving_avg: int = 1, topk_factor: float = 1.0,
        dropout_aw: float = 0.1, dropout_ac: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.3),
        independent_heads: bool = False,
    ):
        super().__init__()
        self.ac = AutoCorrelationLayer(
            d_model, n_heads, topk_factor=topk_factor, dropout=dropout_aw,
            independent_heads=independent_heads,
        )
        self.dropout_ac = torch.nn.Dropout(dropout_ac)
        self.decomp1 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff, bias=False),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_ff[0]),
            torch.nn.Linear(d_ff, d_model, bias=False),
            torch.nn.Dropout(dropout_ff[1]),
        )
        self.decomp2 = SeriesDecomp(decomp_kernel, n_moving_avg)

    def forward(self, x):
        y = self.dropout_ac(self.ac(x, x, x))
        s, _ = self.decomp1(x + y)
        y = self.ff(s)
        x, _ = self.decomp2(s + y)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, c_out: int,
        decomp_kernel: int, n_moving_avg: int = 1, topk_factor: float = 1.0,
        dropout_aw: float = 0.1, dropout_ac: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.3),
        independent_heads: bool = False,
    ):
        super().__init__()
        self.self_ac = AutoCorrelationLayer(
            d_model, n_heads, topk_factor=topk_factor, dropout=dropout_aw,
            independent_heads=independent_heads,
        )
        self.cross_ac = AutoCorrelationLayer(
            d_model, n_heads, topk_factor=topk_factor, dropout=dropout_aw,
            independent_heads=independent_heads,
        )
        self.dropout_ac = torch.nn.Dropout(dropout_ac)
        self.decomp1 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.decomp2 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.decomp3 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff, bias=False),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_ff[0]),
            torch.nn.Linear(d_ff, d_model, bias=False),
            torch.nn.Dropout(dropout_ff[1]),
        )
        self.out_proj = ConvEmb(d_model, c_out, kernel_size=3)

    def forward(self, x, cross):
        y = self.dropout_ac(self.self_ac(x, x, x))
        x, trend1 = self.decomp1(x + y)
        y = self.dropout_ac(self.cross_ac(x, cross, cross))
        x, trend2 = self.decomp2(x + y)
        y = self.ff(x)
        x, trend3 = self.decomp3(x + y)
        return x, self.out_proj(trend1 + trend2 + trend3)


class Autoformer(BasicBaseModel):
    """
    !!! note "Original Research"
        This model is based on the following research:
        > Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long.
          "Autoformer: Decomposition Transformers with Auto-Correlation
          for Long-Term Series Forecasting."
          In Advances in Neural Information Processing Systems
          (NeurIPS 2021), vol. 34, 2021.
          [Paper](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html) |
          [arXiv](https://arxiv.org/abs/2106.13008) |
          [GitHub](https://github.com/thuml/Autoformer)
    """
    def _get_seq_len_for_model(self, seq_len):
        return seq_len

    def _setup(
        self, seq_len: int, pred_len: int, c_in: int, label_len: int = None,
        freq: str = 'Hour', e_layers: int = 2, d_layers: int = 1,
        d_model: int = 64, n_heads: int = 4, d_ff: int = 256,
        decomp_kernel: int = 25, n_moving_avg: int = 1,
        topk_factor: float = 1.0, dropout_emb: float = 0.05,
        dropout_aw: float = 0.1, dropout_ac: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.3),
        independent_heads: bool = False,
        quantile_mode_train: str | None = None,
        quantile_mode_eval: str | None = None,
    ) -> None:
        super()._setup(seq_len, pred_len)
        seq_len_for_model = self._get_seq_len_for_model(seq_len)

        self.c_in = c_in
        self.c_out = c_in
        self.d_model = d_model
        self.freq = freq
        self.label_len = label_len
        if self.label_len is None:
            self.label_len = seq_len_for_model // 2

        self.enc_in_proj = ConvEmb(c_in, d_model, kernel_size=3)
        self.dec_in_proj = ConvEmb(c_in, d_model, kernel_size=3)
        self.enc_tfe = TimeFeatureEmbedding(d_model, freq=freq)
        self.dec_tfe = TimeFeatureEmbedding(d_model, freq=freq)
        self.decomp = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.encoder_layers = torch.nn.ModuleList([
            EncoderLayer(
                d_model, n_heads, d_ff,
                decomp_kernel, n_moving_avg, topk_factor,
                dropout_aw, dropout_ac, dropout_ff, independent_heads,
            ) for _ in range(e_layers)
        ])
        self.decoder_layers = torch.nn.ModuleList([
            DecoderLayer(
                d_model, n_heads, d_ff, self.c_out,
                decomp_kernel, n_moving_avg, topk_factor,
                dropout_aw, dropout_ac, dropout_ff, independent_heads,
            ) for _ in range(d_layers)
        ])
        self.enc_norm = TokenNormTimeDemean(d_model)
        self.dec_norm = TokenNormTimeDemean(d_model)
        self.out_proj = torch.nn.Linear(d_model, self.c_out)

        if quantile_mode_train and quantile_mode_eval:
            self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)

    def _build_marks(self, batch, device):
        # Build encoder/decoder time-feature tensors from batch timestamps.
        tsta = np.asarray(batch.tsta[:, -self.seq_len:])
        tsta_dec = np.concatenate([
            tsta[:, -self.label_len:],
            np.asarray(batch.tsta_future[:, :self.pred_len]),
        ], axis=1)
        x_mark_enc = self.enc_tfe.get_feats(tsta)
        x_mark_dec = self.dec_tfe.get_feats(tsta_dec)
        return (
             torch.tensor(x_mark_enc, dtype=torch.float32, device=device),
             torch.tensor(x_mark_dec, dtype=torch.float32, device=device),
        )

    def _extract_input(self, batch):
        x = super()._extract_input(batch)
        x_mark_enc, x_mark_dec = self._build_marks(batch, x.device)
        return x, x_mark_enc, x_mark_dec

    def forward(self, input_):
        x_enc, x_mark_enc, x_mark_dec = input_
        B, L, C = x_enc.shape
        dtype = x_enc.dtype

        # Encoder
        enc_h = self.enc_in_proj(x_enc) + self.enc_tfe(x_mark_enc)
        enc_h = self.dropout_emb(enc_h)
        for layer in self.encoder_layers:
            enc_h = layer(enc_h)
        enc_h = self.enc_norm(enc_h)

        # Decoder input initialization
        seasonal_init, trend_init = self.decomp(x_enc)
        seasonal_init = torch.cat([
            seasonal_init[:, -self.label_len:, :],
            torch.zeros(B, self.pred_len, C, device=self.device, dtype=dtype),
        ], dim=1)
        trend_init = torch.cat([
            trend_init[:, -self.label_len:, :],
            x_enc.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1),
        ], dim=1)

        # Decoder
        dec_h = self.dec_in_proj(seasonal_init) + self.dec_tfe(x_mark_dec)
        dec_h = self.dropout_emb(dec_h)
        trend = trend_init
        for layer in self.decoder_layers:
            dec_h, residual_trend = layer(dec_h, enc_h)
            trend = trend + residual_trend
        dec_h = self.dec_norm(dec_h)

        seasonal_part = self.out_proj(dec_h)
        out = seasonal_part + trend
        return out[:, -self.pred_len:, :], {}


class DiffAutoformer(Autoformer):
    def _get_seq_len_for_model(self, seq_len):
        return seq_len - 1

    def forward(self, input_):
        x, x_mark_enc, x_mark_dec = input_
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx_mark_enc = x_mark_enc[:, 1:, :]
        last_val = x[:, -1:, :]
        pred_dx, info = super().forward((dx, dx_mark_enc, x_mark_dec))
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
