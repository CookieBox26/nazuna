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
        # x_mark: [B, L, d_inp]
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
        arr = np.stack(feats, axis=-1)  # [B*L, d_inp]
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
        # x: [B, L, D] -> [B, H, L, d_head]
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, x):
        b, l, _ = x.shape
        q = self._reshape_heads(self.q_proj(x))  # [B, H, L, d_head]
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))

        # Auto-correlation via FFT
        qf = torch.fft.rfft(q, dim=2)
        kf = torch.fft.rfft(k, dim=2)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=l, dim=2)
        corr = corr.mean(dim=-1)  # [B, H, L]

        topk = max(1, int(self.topk_factor * math.log(l + 1)))
        vals, lags = torch.topk(corr, k=topk, dim=-1)  # [B, H, topk]
        w = torch.softmax(vals, dim=-1)

        agg = torch.zeros_like(v)
        for i in range(topk):
            lag = lags[..., i]  # [B, H]
            idx = (
                torch.arange(l, device=x.device)[None, None, :]
                - lag[..., None]
            ) % l
            # idx: [B, H, L] -> [B, H, L, d_head]
            idx = idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head)
            shifted = v.gather(dim=2, index=idx)
            agg = agg + shifted * w[..., i].unsqueeze(-1).unsqueeze(-1)

        agg = self.dropout(agg)
        agg = agg.transpose(1, 2).contiguous().view(b, l, self.d_model)
        return self.out_proj(agg)


class AutoformerBlock(nn.Module):
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
        self.decomp1 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.ac = AutoCorrelation(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.decomp2 = SeriesDecomp(decomp_kernel, n_moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        s, t = self.decomp1(x)
        s2 = self.ac(s)
        x = s + self.dropout(s2) + t
        x = self.norm1(x)

        s, t = self.decomp2(x)
        s2 = self.ff(s)
        x = s + self.dropout(s2) + t
        x = self.norm2(x)
        return x


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
        decomp_kernel: int = 25,
        n_moving_avg: int = 1,
        dropout: float = 0.1,
        freq: str = 'h',
    ) -> None:
        super()._setup(seq_len, pred_len)

        self.c_in = c_in
        self.d_model = d_model
        self.freq = freq
        self.in_proj = nn.Linear(c_in, d_model)
        self.time_embed = TimeFeatureEmbedding(d_model, freq=freq)
        self.blocks = nn.ModuleList([
            AutoformerBlock(
                d_model, n_heads, d_ff,
                decomp_kernel, n_moving_avg, dropout,
            )
            for _ in range(e_layers)
        ])
        self.head = nn.Linear(d_model, pred_len * c_in)
        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)

    def _extract_input(self, batch):
        x = super()._extract_input(batch)
        tsta = batch.tsta[:, -self.seq_len:]
        x_mark = TimeFeatureEmbedding.build_features(tsta, self.freq, x.device)
        return x, x_mark

    def forward(self, input_):
        x, x_mark = input_
        B, L, C = x.shape
        h = self.in_proj(x)
        h = h + self.time_embed(x_mark)
        for blk in self.blocks:
            h = blk(h)
        token = h[:, -1, :]  # [B, d_model]
        yhat = self.head(token)  # [B, pred_len * C]
        yhat = yhat.view(B, self.pred_len, C)
        return yhat, {}


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
        decomp_kernel: int = 25,
        n_moving_avg: int = 1,
        dropout: float = 0.1,
        freq: str = 'h',
    ) -> None:
        # After first-order differencing, length becomes seq_len - 1.
        diff_seq_len = seq_len - 1
        super()._setup(
            diff_seq_len, pred_len,
            quantile_mode_train, quantile_mode_eval,
            c_in, d_model, n_heads, d_ff,
            e_layers, decomp_kernel, n_moving_avg, dropout,
            freq,
        )
        # Restore original seq_len for _extract_input slicing.
        self.seq_len = seq_len

    def forward(self, input_):
        x, x_mark = input_
        dx_mark = x_mark[:, 1:, :]
        sub_input = (x[:, 1:, :] - x[:, :-1, :], dx_mark)
        last_val = x[:, -1:, :]  # [B, 1, C]
        pred_dx, info = super().forward(sub_input)
        pred = last_val + torch.cumsum(pred_dx, dim=1)
        return pred, info
