import torch
from abc import ABC


class BaseLoss(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def setup(self):
        pass

    @classmethod
    def create(cls, device, **kwargs):
        criterion = cls(**kwargs)
        criterion.device = device
        criterion.setup()
        return criterion


class MSELoss(BaseLoss):
    """Mean Squared Error loss with channel and sequence weighting."""

    def __init__(
        self,
        n_channel: int,
        pred_len: int,
        decay_rate: float | None = None,
    ):
        """
        Args:
            n_channel: Number of channels in the prediction.
            pred_len: Length of the prediction sequence.
            decay_rate: Exponential decay rate for sequence weighting.
                If None or 1, uniform weights are used.
        """
        super().__init__()
        self.n_channel = n_channel
        self.pred_len = pred_len
        self.decay_rate = decay_rate

    def set_w_channel(self):
        self.w_channel = torch.ones(self.n_channel, dtype=torch.float, device=self.device)
        self.w_channel /= self.w_channel.sum()

    def set_w_seq(self):
        r = self.decay_rate
        if r is None or r == 1:
            w = torch.ones(self.pred_len, dtype=torch.float, device=self.device)
        else:
            idx = torch.arange(self.pred_len, dtype=torch.float, device=self.device)
            w = torch.pow(torch.tensor(r, dtype=torch.float, device=self.device), idx)  # [1, r, r^2, ...]
        self.w_seq = w / w.sum()

    def setup(self):
        self.set_w_channel()
        self.set_w_seq()

    def calc_loss(self, pred, true):
        return (pred[:, :self.pred_len, :] - true[:, :self.pred_len, :]) ** 2

    def forward(self, pred, true):
        loss = self.calc_loss(pred, true)  # batch_size, pred_len, n_channel
        me_of_each_sample_channel = torch.einsum('j,ijk->ik', (self.w_seq, loss))
        me_of_each_sample = torch.einsum('k,ik->i', (self.w_channel, me_of_each_sample_channel))
        return (
            me_of_each_sample.mean(),  # (scalar)
            me_of_each_sample,  # batch_size
            me_of_each_sample_channel,  # batch_size, n_channel
        )


class MAELoss(MSELoss):
    def calc_loss(self, pred, true):
        return torch.abs(pred[:, :self.pred_len, :] - true[:, :self.pred_len, :])
