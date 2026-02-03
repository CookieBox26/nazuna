from abc import ABC, abstractmethod
import dataclasses
import torch


@dataclasses.dataclass
class TimeSeriesError:
    batch_size: int
    batch_mean: torch.Tensor
    each_sample: torch.Tensor
    each_sample_channel: torch.Tensor | None = None
    info: dict = dataclasses.field(default_factory=dict)

    def batch_sum(self) -> float:
        return self.each_sample.shape[0] * self.batch_mean.item()


class BaseError(torch.nn.Module, ABC):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.device = device
        self._setup(**kwargs)
        self.to(device)

    @abstractmethod
    def _setup(self, **kwargs):
        pass

    @classmethod
    def create(cls, device, **kwargs):
        return cls(device=device, **kwargs)


class BaseImprovement(BaseError, ABC):
    """
    Abstract base class for metrics that compare predictions against a baseline.
    Subclasses should implement forward(baseline, pred, true) instead of forward(pred, true).
    """
    pass


class MSE(BaseError):
    """
    Mean Squared Error with channel and sequence weighting.
    """
    def _setup(
        self,
        n_channel: int,
        pred_len: int,
        decay_rate: float | None = None,
        tolerance: float = 0,
    ):
        """
        Args:
            n_channel: Number of channels in the prediction.
            pred_len: Length of the prediction sequence.
            decay_rate: Exponential decay rate for sequence weighting.
                If None or 1, uniform weights are used.
            tolerance: Errors below this threshold are treated as zero.
                Default is 0 (no tolerance).

        Note:
            The weight for step i is decay_rate^i (before normalization).
            For example, if decay_rate=0.9, step 0 has weight 1.0, step 1 has 0.9,
            step 2 has 0.81, and so on.
        """
        self.n_channel = n_channel
        self.pred_len = pred_len
        self.decay_rate = decay_rate
        self.tolerance = tolerance
        self._set_w_channel()
        self._set_w_seq()

    def _set_w_channel(self):
        self.w_channel = torch.ones(self.n_channel, dtype=torch.float, device=self.device)
        self.w_channel /= self.w_channel.sum()

    def _set_w_seq(self):
        r = self.decay_rate
        if r is None or r == 1:
            w = torch.ones(self.pred_len, dtype=torch.float, device=self.device)
        else:
            idx = torch.arange(self.pred_len, dtype=torch.float, device=self.device)
            w = torch.pow(torch.tensor(r, dtype=torch.float, device=self.device), idx)  # [1, r, r^2, ...]
        self.w_seq = w / w.sum()

    def get_error(self, pred, true):
        error = pred[:, :self.pred_len, :] - true[:, :self.pred_len, :]
        if self.tolerance > 0:
            error = torch.where(torch.abs(error) < self.tolerance, torch.zeros_like(error), error)
        return error ** 2

    def forward(self, pred, true):
        error = self.get_error(pred, true)  # batch_size, pred_len, n_channel
        error_of_each_sample_channel = torch.einsum('j,ijk->ik', (self.w_seq, error))
        error_of_each_sample = torch.einsum('k,ik->i', (self.w_channel, error_of_each_sample_channel))
        return TimeSeriesError(
            true.shape[0],
            error_of_each_sample.mean(),  # (scalar)
            error_of_each_sample,  # batch_size
            error_of_each_sample_channel,  # batch_size, n_channel
        )


class MAE(MSE):
    def get_error(self, pred, true):
        error = pred[:, :self.pred_len, :] - true[:, :self.pred_len, :]
        if self.tolerance > 0:
            error = torch.where(torch.abs(error) < self.tolerance, torch.zeros_like(error), error)
        return torch.abs(error)


class ImprovementRate(BaseImprovement):
    """
    Computes improvement rate over a baseline prediction.

    Improvement rate = 1 - (pred_error / baseline_error)
    - Positive values indicate improvement over baseline
    - Zero indicates equal performance
    - Negative values indicate worse performance than baseline
    """
    def _setup(
        self,
        n_channel: int,
        pred_len: int,
        decay_rate: float | None = None,
        tolerance: float = 0,
        epsilon: float = 1e-8,
        error_type: str = "mse",
    ):
        """
        Args:
            n_channel: Number of channels in the prediction.
            pred_len: Length of the prediction sequence.
            decay_rate: Exponential decay rate for sequence weighting.
                If None or 1, uniform weights are used.
            tolerance: Errors below this threshold are treated as zero.
                Default is 0 (no tolerance).
            epsilon: Small value to avoid division by zero.
            error_type: Type of error metric ("mse" or "mae").
        """
        self.n_channel = n_channel
        self.pred_len = pred_len
        self.decay_rate = decay_rate
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.error_type = error_type
        self._set_w_channel()
        self._set_w_seq()

    def _set_w_channel(self):
        self.w_channel = torch.ones(self.n_channel, dtype=torch.float, device=self.device)
        self.w_channel /= self.w_channel.sum()

    def _set_w_seq(self):
        r = self.decay_rate
        if r is None or r == 1:
            w = torch.ones(self.pred_len, dtype=torch.float, device=self.device)
        else:
            idx = torch.arange(self.pred_len, dtype=torch.float, device=self.device)
            w = torch.pow(torch.tensor(r, dtype=torch.float, device=self.device), idx)
        self.w_seq = w / w.sum()

    def get_raw_error(self, pred, true):
        error = pred[:, :self.pred_len, :] - true[:, :self.pred_len, :]
        if self.tolerance > 0:
            error = torch.where(torch.abs(error) < self.tolerance, torch.zeros_like(error), error)
        if self.error_type == "mse":
            return error ** 2
        else:
            return torch.abs(error)

    def forward(self, baseline, pred, true):
        """
        Args:
            baseline: Baseline model prediction (batch_size, pred_len, n_channel)
            pred: Model prediction to evaluate (batch_size, pred_len, n_channel)
            true: Ground truth (batch_size, pred_len, n_channel)

        Returns:
            TimeSeriesError: Aggregated improvement rates
        """
        baseline_error = self.get_raw_error(baseline, true)  # (batch_size, pred_len, n_channel)
        pred_error = self.get_raw_error(pred, true)  # (batch_size, pred_len, n_channel)

        # (batch_size, n_channel)
        error_of_each_sample_channel_baseline = torch.einsum('j,ijk->ik', self.w_seq, baseline_error)
        error_of_each_sample_channel = torch.einsum('j,ijk->ik', self.w_seq, pred_error)

        improvement_each_sample_channel = 1 - (
            error_of_each_sample_channel / (error_of_each_sample_channel_baseline + self.epsilon)
        )

        # (batch_size,)
        improvement_each_sample = torch.einsum('k,ik->i', self.w_channel, improvement_each_sample_channel)

        improvement_batch_mean = improvement_each_sample.mean()

        return TimeSeriesError(
            batch_size=true.shape[0],
            batch_mean=improvement_batch_mean,
            each_sample=improvement_each_sample,
            each_sample_channel=improvement_each_sample_channel,
            info={
                "error_baseline": error_of_each_sample_channel_baseline,
                "error_pred": error_of_each_sample_channel,
            },
        )
