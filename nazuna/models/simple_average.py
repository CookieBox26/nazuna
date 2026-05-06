from nazuna.models._base import BasicBaseModel
import torch


class BaseSimpleAverage(BasicBaseModel):
    def _setup(self, seq_len, pred_len, period_len, prep_type: str = 'none'):
        super()._setup(seq_len, pred_len, prep_type=prep_type)
        self.period_len = period_len
        assert self.seq_len % self.period_len == 0
        self.n_period = int(self.seq_len / self.period_len)
        assert self.pred_len % self.period_len == 0
        self.n_repeat = int(self.pred_len / self.period_len)


class SimpleAverage(BaseSimpleAverage):
    """
    !!! tip "Example parameter configurations"
        ```toml
        [definitions.SimpleAverage]
        cls_path = "nazuna.models.simple_average.SimpleAverage"
        [definitions.SimpleAverage.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        period_len = 24  # task-dependent
        decay_rate = 1.0
        prep_type = "none"
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        decay_rate: float = 1.0,
        prep_type: str = 'none',
    ) -> None:
        super()._setup(seq_len, pred_len, period_len, prep_type=prep_type)
        self.decay_rate = decay_rate
        w = torch.tensor(
            [self.decay_rate**j for j in reversed(range(self.n_period))],
            dtype=torch.float, device=self.device,
        )  # Ex. [0.1250, 0.2500, 0.5000, 1.0000] (n_period = 4, decay_rate = 0.5)
        self.weight = w / w.sum()

    def forward(self, x):
        batch_size, _, n_channel = x.shape  # batch_size, seq_len, n_channel
        x_view = x.view(batch_size, self.n_period, self.period_len, n_channel)
        y = torch.einsum('j,ijkl->ikl', (self.weight, x_view))
        return y.repeat(1, self.n_repeat, 1), {}


class SimpleAverageVariableDecay(BaseSimpleAverage):
    """
    !!! tip "Example parameter configurations"
        ```toml
        [definitions.SimpleAverageVariableDecay]
        cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecay"
        [definitions.SimpleAverageVariableDecay.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        period_len = 24  # task-dependent
        prep_type = "none"
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        prep_type: str = 'none',
    ) -> None:
        super()._setup(seq_len, pred_len, period_len, prep_type=prep_type)
        self.decay_rate = torch.nn.Parameter(torch.tensor(0.7))  # Initial decay rate

    def forward(self, x):
        batch_size, _, n_channel = x.shape  # batch_size, seq_len, n_channel
        x_view = x.view(batch_size, self.n_period, self.period_len, n_channel)
        j = torch.arange(
            self.n_period - 1, -1, -1, dtype=torch.float32, device=self.device,
        )  # Ex. [3., 2., 1., 0.] (n_period = 4)
        w = self.decay_rate ** j
        w = w / w.sum()
        y = torch.einsum('j,ijkl->ikl', (w, x_view))
        return y.repeat(1, self.n_repeat, 1), {}


class SimpleAverageVariableDecayChannelwise(BaseSimpleAverage):
    """
    !!! tip "Example parameter configurations"
        ```toml
        [definitions.SimpleAverageVariableDecayChannelwise]
        cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecayChannelwise"
        [definitions.SimpleAverageVariableDecayChannelwise.params]
        seq_len = 96  # task-dependent
        pred_len = 24  # task-dependent
        period_len = 24  # task-dependent
        n_channel = 7  # task-dependent
        prep_type = "none"
        ```
    """
    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        period_len: int,
        n_channel: int,
        prep_type: str = 'none',
    ) -> None:
        super()._setup(seq_len, pred_len, period_len, prep_type=prep_type)
        self.n_channel = n_channel
        self.decay_rate = torch.nn.Parameter(torch.full((n_channel,), 0.7))  # Initial decay rate

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
        y = torch.einsum('lj,ijkl->ikl', (w, x_view))
        return y.repeat(1, self.n_repeat, 1), {}
