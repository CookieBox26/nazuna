from nazuna.models.base import BaseModel
import torch


class BaseSimpleAverage(BaseModel):
    def _setup(self, seq_len, pred_len, period_len, **kwargs):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len
        assert self.seq_len % self.period_len == 0
        self.n_period = int(self.seq_len / self.period_len)

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
        return loss


class SimpleAverage(BaseSimpleAverage):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        decay_rate: float = 1.0,
    ) -> None:
        """
        Args:
            seq_len: Input sequence length (must be divisible by `period_len`)
            pred_len: Prediction length
            period_len: Length of one period (e.g., 24 for hourly data with daily periodicity)
            decay_rate: Weight decay rate for older periods (default: 1.0, meaning equal weights)
        """
        super()._setup(seq_len, pred_len, period_len)
        self.decay_rate = decay_rate
        w = torch.tensor(
            [self.decay_rate**j for j in reversed(range(self.n_period))],
            dtype=torch.float, device=self.device,
        )  # Ex. [0.1250, 0.2500, 0.5000, 1.0000] (n_period = 4, decay_rate = 0.5)
        self.weight = w / w.sum()

    def forward(self, x):
        batch_size, _, n_channel = x.shape  # batch_size, seq_len, n_channel
        x_view = x.view(batch_size, self.n_period, self.period_len, n_channel)
        return torch.einsum('j,ijkl->ikl', (self.weight, x_view)), {}


class SimpleAverageVariableDecay(BaseSimpleAverage):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
    ) -> None:
        super()._setup(seq_len, pred_len, period_len)
        self.decay_rate = torch.nn.Parameter(torch.tensor(0.7))  # Initial decay rate

    def forward(self, x):
        batch_size, _, n_channel = x.shape  # batch_size, seq_len, n_channel
        x_view = x.view(batch_size, self.n_period, self.period_len, n_channel)
        j = torch.arange(
            self.n_period - 1, -1, -1, dtype=torch.float32, device=self.device,
        )  # Ex. [3., 2., 1., 0.] (n_period = 4)
        w = self.decay_rate ** j
        w = w / w.sum()
        return torch.einsum('j,ijkl->ikl', (w, x_view)), {}


class SimpleAverageVariableDecayChannelwise(BaseSimpleAverage):
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        n_channel: int,
    ) -> None:
        super()._setup(seq_len, pred_len, period_len)
        self.n_channel = n_channel
        self.decay_rate = torch.nn.Parameter(torch.full((n_channel,), 0.7))  # Initial decay rate per channel

    def forward(self, x):
        batch_size, _, n_channel = x.shape  # batch_size, seq_len, n_channel
        x_view = x.view(batch_size, self.n_period, self.period_len, n_channel)
        j = torch.arange(
            self.n_period - 1, -1, -1, dtype=torch.float32, device=self.device,
        )  # Ex. [3., 2., 1., 0.] (n_period = 4)
        # decay_rate: (n_channel,), j: (n_period,) -> w: (n_channel, n_period)
        w = self.decay_rate.unsqueeze(1) ** j.unsqueeze(0)
        w = w / w.sum(dim=1, keepdim=True)
        # x_view: (batch_size, n_period, period_len, n_channel)
        # w: (n_channel, n_period) -> einsum: 'lj,ijkl->ikl'
        return torch.einsum('lj,ijkl->ikl', (w, x_view)), {}