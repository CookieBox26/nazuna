from enum import Enum
import numpy as np
import pandas as pd
import torch


class TimeFeatureEmbedding(torch.nn.Module):
    Freq = Enum('Freq', ['Hour', 'Day'])

    to_hour = classmethod(lambda cls, arr: arr.hour.values.astype(np.float32))
    to_dow = classmethod(lambda cls, arr: arr.dayofweek.values.astype(np.float32))
    to_day = classmethod(lambda cls, arr: arr.day.values.astype(np.float32) - 1.0)
    to_doy = classmethod(lambda cls, arr: arr.dayofyear.values.astype(np.float32) - 1.0)

    to_hour_feat = classmethod(lambda cls, arr: cls.to_hour(arr) / 23.0 - 0.5)
    to_dow_feat = classmethod(lambda cls, arr: cls.to_dow(arr) / 6.0 - 0.5)
    to_day_feat = classmethod(lambda cls, arr: cls.to_day(arr) / 30.0 - 0.5)
    to_doy_feat = classmethod(lambda cls, arr: cls.to_doy(arr) / 365.0 - 0.5)

    def _get_extractors(self):
        cls = type(self)
        if self.freq == cls.Freq.Hour:
            return [cls.to_hour_feat, cls.to_dow_feat, cls.to_day_feat, cls.to_doy_feat]
        if self.freq == cls.Freq.Day:
            return [cls.to_dow_feat, cls.to_day_feat, cls.to_doy_feat]
        raise NotImplementedError(self.freq)

    def __init__(self, d_model: int, freq: str = 'Hour'):
        super().__init__()
        self.freq = type(self).Freq[freq]
        self.n_feat = len(self._get_extractors())
        self.embed = torch.nn.Linear(self.n_feat, d_model, bias=False)

    def forward(self, x_mark):
        return self.embed(x_mark)

    def get_feats(self, tsta: np.ndarray) -> torch.Tensor:
        # tsta: numpy array of timestamps with shape [B, L]
        # Returns float tensor [B, L, d_inp] normalized to [-0.5, 0.5].
        B, L = np.asarray(tsta).shape
        flatten = pd.DatetimeIndex(np.asarray(tsta).reshape(-1))
        feats = []
        for extractor in self._get_extractors():
            feats.append(extractor(flatten))
        return np.stack(feats, axis=-1).reshape(B, L, -1)
