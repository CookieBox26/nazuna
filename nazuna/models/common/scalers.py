from abc import ABC, abstractmethod
import torch


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
    def __init__(self, stat_types: tuple[str, str]):  # train, eval
        super().__init__()
        self.stat_type_train = stat_types[0]
        self.stat_type_eval = stat_types[1]
        self.register_buffer('q1s', None)
        self.register_buffer('q2s', None)
        self.register_buffer('q3s', None)

    def prepare_load_state_dict(self, state_dict):
        self.q1s = torch.empty_like(state_dict['scaler.q1s'])
        self.q2s = torch.empty_like(state_dict['scaler.q2s'])
        self.q3s = torch.empty_like(state_dict['scaler.q3s'])

    def _get_stats(self, batch):
        stat_type = self.stat_type_train if self.training else self.stat_type_eval
        if stat_type == 'saved':
            if self.q1s is None:
                raise ValueError('Saved quartiles not found')
            return self.q1s, self.q2s, self.q3s

        # 1, 3, 1, n_channel -> 1, 1, n_channel
        q1s_ = batch.stats[stat_type][:, 0]
        q2s_ = batch.stats[stat_type][:, 1]
        q3s_ = batch.stats[stat_type][:, 2]
        if self.training:
            self.q1s, self.q2s, self.q3s = q1s_, q2s_, q3s_
        return q1s_, q2s_, q3s_

    def scale(self, x, batch):
        # x: batch_size, seq_len, n_channel
        q1s_, q2s_, q3s_ = self._get_stats(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return (x - means) / stds

    def rescale(self, z, batch):
        q1s_, q2s_, q3s_ = self._get_stats(batch)
        means = q2s_
        stds = q3s_ - q1s_
        return z * stds + means
