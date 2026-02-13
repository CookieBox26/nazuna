import torch
from abc import ABC, abstractmethod


class BaseScaler(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.training = False

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
            if self.q1s is None:
                raise ValueError('Saved quartiles not found')
            if self.q1s.shape[0] == batch.data.shape[0]:
                return self.q1s, self.q2s, self.q3s
            return (
                self.q1s[:1].expand(batch.data.shape[0], -1, -1),
                self.q2s[:1].expand(batch.data.shape[0], -1, -1),
                self.q3s[:1].expand(batch.data.shape[0], -1, -1),
            )

        # batch_size, 3, 1, n_channel -> batch_size, 1, n_channel
        q1s_ = batch.quantiles[self.mode][:, 0]
        q2s_ = batch.quantiles[self.mode][:, 1]
        q3s_ = batch.quantiles[self.mode][:, 2]
        if self.training:
            self.q1s, self.q2s, self.q3s = q1s_, q2s_, q3s_
        return q1s_, q2s_, q3s_

    def scale(self, x, batch):
        # x: batch_size, seq_len, n_channel
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return (x - means) / stds

    def rescale(self, z, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return z * stds + means
