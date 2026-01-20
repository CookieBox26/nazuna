from nazuna.models.base import BaseModel
import torch


class BaseSimpleAverage(BaseModel):
    def _setup(self, seq_len, pred_len, period_len, **kwargs):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        assert seq_len % self.period_len == 0

    def extract_input(self, batch):
        return batch.data[:, -self.seq_len:, :]

    def extract_target(self, batch):
        return batch.data_future[:, :self.pred_len]

    def predict(self, batch):
        input_ = self.extract_input(batch)
        output = self(input_)
        return output

    def get_loss(self, batch, criterion):
        input_ = self.extract_input(batch)
        target = self.extract_target(batch)
        output = self(input_)
        loss = criterion(output[0], target)
        return loss, {}


class SimpleAverage(BaseSimpleAverage):
    def _setup(self, seq_len: int, pred_len: int, period_len: int, decay_rate: float = 1.0, max_n_period: int = 10) -> None:
        """
        Args:
            seq_len: Input sequence length (must be divisible by `period_len`)
            pred_len: Prediction length
            period_len: Length of one period (e.g., 24 for hourly data with daily periodicity)
            decay_rate: Weight decay rate for older periods (default: 1.0, meaning equal weights)
            max_n_period: Maximum number of periods to consider
        """
        super()._setup(seq_len, pred_len, period_len)
        self.decay_rate = decay_rate
        self.max_n_period = max_n_period
        self.w_base = torch.tensor(
            [self.decay_rate**j for j in reversed(range(self.max_n_period))],
            dtype=torch.float, device=self.device,
        )

    def forward(self, x):
        batch_size, _, n_channel = x.shape  # batch_size, seq_len, n_channel
        x_view = x.view(batch_size, -1, self.period_len, n_channel)  # batch_size, n_period, period_len, n_channel
        n_period = x_view.shape[1]
        w = self.w_base[(- n_period):]
        w = w / w.sum()
        return torch.einsum('j,ijkl->ikl', (w, x_view)), {}
