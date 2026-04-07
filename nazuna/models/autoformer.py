import math
from nazuna.models.base import BasicBaseModel
from nazuna.scaler import IqrScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TimeFeatureEmbedding(nn.Module):
    # Number of time features per `freq` setting.
    # 'h': hour, weekday, day, dayofyear
    # 't': minute, hour, weekday, day, dayofyear
    FREQ_MAP = {'h': 4, 't': 5}

    def __init__(self, d_model: int, freq: str = 'h'):
        super().__init__()
        if freq not in self.FREQ_MAP:
            raise ValueError(f"Unsupported freq '{freq}'. Use one of {list(self.FREQ_MAP)}.")
        self.freq = freq
        self.embed = nn.Linear(self.FREQ_MAP[freq], d_model, bias=False)

    def forward(self, x_mark):
        return self.embed(x_mark)

    @staticmethod
    def build_features(tsta: np.ndarray, freq: str, device) -> torch.Tensor:
        # tsta: numpy array of timestamps with shape [B, L]
        # Returns float tensor [B, L, d_inp] normalized to [-0.5, 0.5].
        flat = pd.DatetimeIndex(np.asarray(tsta).reshape(-1))
        hour = flat.hour.values.astype(np.float32) / 23.0 - 0.5
        weekday = flat.dayofweek.values.astype(np.float32) / 6.0 - 0.5
        day = (flat.day.values.astype(np.float32) - 1.0) / 30.0 - 0.5
        dayofyear = (flat.dayofyear.values.astype(np.float32) - 1.0) / 364.0 - 0.5
        feats = [hour, weekday, day, dayofyear]
        if freq == 't':
            minute = flat.minute.values.astype(np.float32) / 59.0 - 0.5
            feats = [minute] + feats
        arr = np.stack(feats, axis=-1)
        b, l = np.asarray(tsta).shape
        arr = arr.reshape(b, l, -1)
        return torch.tensor(arr, dtype=torch.float32, device=device)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size, n_moving_avg=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_moving_avg = n_moving_avg
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=1, padding=0
        )

    def moving_avg(self, x):
        pad_front = self.kernel_size // 2
        pad_end = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_front, 1)
        end = x[:, -1:, :].repeat(1, pad_end, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x):
        trend = x
        for _ in range(self.n_moving_avg):
            trend = self.moving_avg(trend)
        seasonal = x - trend
        return seasonal, trend


class AutoCorrelation(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        topk_factor: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.topk_factor = topk_factor

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x):
        # [B, L, D] -> [B, H, L, d_head]
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q_in, k_in, v_in, causal: bool = False):
        # q_in: [B, Lq, D]; k_in, v_in: [B, Lk, D].
        # When Lq != Lk, align lengths by zero-padding or truncating k/v to Lq
        # (official Autoformer behavior for cross-correlation).
        b, lq, _ = q_in.shape
        lk = k_in.shape[1]
        if lk < lq:
            pad = torch.zeros(b, lq - lk, self.d_model, device=k_in.device, dtype=k_in.dtype)
            k_in = torch.cat([k_in, pad], dim=1)
            v_in = torch.cat([v_in, pad], dim=1)
        elif lk > lq:
            k_in = k_in[:, :lq, :]
            v_in = v_in[:, :lq, :]
        l = lq

        q = self._reshape_heads(self.q_proj(q_in))
        k = self._reshape_heads(self.k_proj(k_in))
        v = self._reshape_heads(self.v_proj(v_in))

        qf = torch.fft.rfft(q, dim=2)
        kf = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=l, dim=2)
        corr = corr.mean(dim=-1)  # [B, H, L]

        if causal:
            # Disallow positive lags that would reach into future positions.
            # lag index i corresponds to a circular shift by i steps; in the
            # decoder self-attention we mask out lags that mix future into past.
            mask = torch.full_like(corr, float('-inf'))
            # Keep lag 0 only (each position attends to itself) for strict causality
            # at the correlation-selection stage. The top-k then degenerates to lag 0.
            mask[..., 0] = 0.0
            corr = corr + mask

        topk = max(1, int(self.topk_factor * math.log(l + 1)))
        vals, lags = torch.topk(corr, k=topk, dim=-1)
        w = torch.softmax(vals, dim=-1)

        agg = torch.zeros_like(v)
        for i in range(topk):
            lag = lags[..., i]  # [B, H]
            idx = (
                torch.arange(l, device=q_in.device)[None, None, :]
                - lag[..., None]
            ) % l
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
            shifted = v.gather(dim=2, index=idx)
            agg = agg + shifted * w[..., i].unsqueeze(-1).unsqueeze(-1)

        agg = self.dropout(agg)
        agg = agg.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.out_proj(agg)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        decomp_kernel: int,
        n_moving_avg: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ac = AutoCorrelation(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decomp1 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.decomp2 = SeriesDecomp(decomp_kernel, n_moving_avg)

    def forward(self, x):
        y = self.dropout(self.ac(x, x, x))
        s, _ = self.decomp1(x + y)
        y = self.ff(s)
        x, _ = self.decomp2(s + y)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        c_out: int,
        decomp_kernel: int,
        n_moving_avg: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_ac = AutoCorrelation(d_model, n_heads, dropout=dropout)
        self.cross_ac = AutoCorrelation(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decomp1 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.decomp2 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.decomp3 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.trend_proj = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False,
        )

    def forward(self, x, cross):
        y = self.dropout(self.self_ac(x, x, x, causal=True))
        x, trend1 = self.decomp1(x + y)
        y = self.dropout(self.cross_ac(x, cross, cross))
        x, trend2 = self.decomp2(x + y)
        y = self.ff(x)
        x, trend3 = self.decomp3(x + y)
        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.trend_proj(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class ZeroMeanLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm(x)
        bias = x.mean(dim=1, keepdim=True)
        return x - bias


class Autoformer(BasicBaseModel):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode_train: str,
        quantile_mode_eval: str,
        c_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        e_layers: int = 3,
        d_layers: int = 1,
        label_len: int = None,
        decomp_kernel: int = 25,
        n_moving_avg: int = 1,
        dropout: float = 0.1,
        freq: str = 'h',
    ) -> None:
        super()._setup(seq_len, pred_len)

        self.c_in = c_in
        self.c_out = c_in
        self.d_model = d_model
        self.freq = freq
        self.label_len = label_len if label_len is not None else seq_len // 2

        self.enc_in_proj = nn.Linear(c_in, d_model)
        self.dec_in_proj = nn.Linear(c_in, d_model)
        self.enc_time_embed = TimeFeatureEmbedding(d_model, freq=freq)
        self.dec_time_embed = TimeFeatureEmbedding(d_model, freq=freq)

        self.enc_decomp = SeriesDecomp(decomp_kernel, n_moving_avg)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, decomp_kernel, n_moving_avg, dropout)
            for _ in range(e_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                d_model, n_heads, d_ff, self.c_out,
                decomp_kernel, n_moving_avg, dropout,
            )
            for _ in range(d_layers)
        ])
        self.final_norm = ZeroMeanLayerNorm(d_model)
        self.seasonal_proj = nn.Linear(d_model, self.c_out)
        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)

    def _build_marks(self, batch, device):
        # Build encoder/decoder time-feature tensors from batch timestamps.
        tsta_past = np.asarray(batch.tsta[:, -self.seq_len:])
        b, _ = tsta_past.shape
        if batch.tsta_future is not None:
            tsta_future = np.asarray(batch.tsta_future[:, :self.pred_len])
        else:
            # Extrapolate future timestamps assuming uniform spacing.
            last = tsta_past[:, -1:]
            step = tsta_past[:, -1:] - tsta_past[:, -2:-1]
            offsets = np.arange(1, self.pred_len + 1).reshape(1, -1)
            tsta_future = last + step * offsets
        tsta_dec = np.concatenate(
            [tsta_past[:, -self.label_len:], tsta_future], axis=1
        )
        x_mark_enc = TimeFeatureEmbedding.build_features(tsta_past, self.freq, device)
        x_mark_dec = TimeFeatureEmbedding.build_features(tsta_dec, self.freq, device)
        return x_mark_enc, x_mark_dec

    def _extract_input(self, batch):
        x = super()._extract_input(batch)
        x_mark_enc, x_mark_dec = self._build_marks(batch, x.device)
        return x, x_mark_enc, x_mark_dec

    def forward(self, input_):
        x_enc, x_mark_enc, x_mark_dec = input_
        B, L, C = x_enc.shape

        # Decoder input initialization (in raw-channel space for trend).
        mean = x_enc.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            B, self.pred_len, C, device=x_enc.device, dtype=x_enc.dtype
        )
        seasonal_init, trend_init = self.enc_decomp(x_enc)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1
        )
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1
        )

        enc_h = self.enc_in_proj(x_enc) + self.enc_time_embed(x_mark_enc)
        for layer in self.encoder_layers:
            enc_h = layer(enc_h)

        dec_h = self.dec_in_proj(seasonal_init) + self.dec_time_embed(x_mark_dec)
        trend = trend_init
        for layer in self.decoder_layers:
            dec_h, residual_trend = layer(dec_h, enc_h)
            trend = trend + residual_trend

        dec_h = self.final_norm(dec_h)
        seasonal_part = self.seasonal_proj(dec_h)
        out = seasonal_part + trend
        return out[:, -self.pred_len:, :], {}


class DiffAutoformer(Autoformer):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode_train: str,
        quantile_mode_eval: str,
        c_in: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        e_layers: int = 3,
        d_layers: int = 1,
        label_len: int = None,
        decomp_kernel: int = 25,
        n_moving_avg: int = 1,
        dropout: float = 0.1,
        freq: str = 'h',
    ) -> None:
        diff_seq_len = seq_len - 1
        diff_label_len = (
            label_len if label_len is not None else diff_seq_len // 2
        )
        super()._setup(
            diff_seq_len, pred_len,
            quantile_mode_train, quantile_mode_eval,
            c_in, d_model, n_heads, d_ff,
            e_layers, d_layers, diff_label_len,
            decomp_kernel, n_moving_avg, dropout, freq,
        )
        # Restore original seq_len for _extract_input slicing.
        self.seq_len = seq_len

    def forward(self, input_):
        x, x_mark_enc, x_mark_dec = input_
        dx = x[:, 1:, :] - x[:, :-1, :]
        dx_mark_enc = x_mark_enc[:, 1:, :]
        last_val = x[:, -1:, :]
        pred_dx, info = super().forward((dx, dx_mark_enc, x_mark_dec))
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
