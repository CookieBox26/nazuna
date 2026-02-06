import torch
from abc import ABC, abstractmethod


class BaseScaler(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def scale(self, x, batch):
        pass

    @abstractmethod
    def rescale(self, z, batch):
        pass


class IqrScaler(BaseScaler):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.register_buffer('q1s', None)
        self.register_buffer('q2s', None)
        self.register_buffer('q3s', None)

    def prepare_load_state_dict(self, state_dict):
        self.q1s = torch.empty_like(state_dict['scaler.q1s'])
        self.q2s = torch.empty_like(state_dict['scaler.q2s'])
        self.q3s = torch.empty_like(state_dict['scaler.q3s'])

    def _get_quantiles(self, batch):
        if self.mode == 'saved':
            if self.q1s.shape[0] != batch.data.shape[0]:
                self.q1s = self.q1s[:1].expand(batch.data.shape[0], -1, -1)
                self.q2s = self.q2s[:1].expand(batch.data.shape[0], -1, -1)
                self.q3s = self.q3s[:1].expand(batch.data.shape[0], -1, -1)
        else:
            self.q1s = batch.quantiles[self.mode][:, 0, :]
            self.q2s = batch.quantiles[self.mode][:, 1, :]
            self.q3s = batch.quantiles[self.mode][:, 2, :]
        return self.q1s, self.q2s, self.q3s

    def scale(self, x, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return (x - means) / stds

    def rescale(self, z, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return z * stds + means
