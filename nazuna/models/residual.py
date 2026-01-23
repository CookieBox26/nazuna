from nazuna.models.base import BaseModel
from nazuna.scaler import IqrScaler
from nazuna import load_class


def _make_concrete(cls):
    """Create a concrete class from an abstract base class by providing dummy implementations."""
    class ConcreteModel(cls):
        def extract_input(self, batch):
            raise NotImplementedError('Not used in ResidualModel')

        def extract_target(self, batch):
            raise NotImplementedError('Not used in ResidualModel')

        def predict(self, batch):
            raise NotImplementedError('Not used in ResidualModel')
    return ConcreteModel


class ResidualModel(BaseModel):
    """
    Residual learning framework that combines a naive model and a neural model.

    The final prediction is: naive_output + neural_output

    Both sub-models receive the same scaled input and their outputs are summed.
    """

    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode: str,
        naive_model_cls_path: str,
        naive_model_params: dict,
        neural_model_cls_path: str,
        neural_model_params: dict,
    ) -> None:
        """
        Args:
            seq_len: Input sequence length
            pred_len: Prediction length
            quantile_mode: Source of quantiles for scaling ('full', 'cum', or 'rolling')
            naive_model_cls_path: Class path for the naive model (e.g., 'nazuna.models.simple_average.BaseSimpleAverage')
            naive_model_params: Parameters for the naive model
            neural_model_cls_path: Class path for the neural model (e.g., 'nazuna.models.dlinear.BaseDLinear')
            neural_model_params: Parameters for the neural model
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.quantile_mode = quantile_mode
        self.scaler = IqrScaler()

        naive_model_cls = _make_concrete(load_class(naive_model_cls_path))
        self.naive_model = naive_model_cls(device=self.device, **naive_model_params)

        neural_model_cls = _make_concrete(load_class(neural_model_cls_path))
        self.neural_model = neural_model_cls(device=self.device, **neural_model_params)

    def forward(self, x):
        naive_out = self.naive_model(x)
        if isinstance(naive_out, tuple):
            naive_out = naive_out[0]

        neural_out = self.neural_model(x)
        if isinstance(neural_out, tuple):
            neural_out = neural_out[0]

        return naive_out + neural_out

    def _get_quantiles(self, batch):
        if self.quantile_mode == 'full':
            quantiles = batch.quantiles_full
        elif self.quantile_mode == 'cum':
            quantiles = batch.quantiles_cum
        elif self.quantile_mode == 'rolling':
            quantiles = batch.quantiles_rolling
        else:
            raise ValueError(f'Unknown quantile_mode: {self.quantile_mode}')
        q1s_ = quantiles[:, 0, :]
        q2s_ = quantiles[:, 1, :]
        q3s_ = quantiles[:, 2, :]
        return q1s_, q2s_, q3s_

    def extract_input(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        return self.scaler.scale(
            batch.data[:, -self.seq_len:, :], q1s_=q1s_, q2s_=q2s_, q3s_=q3s_
        )

    def extract_target(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        return self.scaler.scale(
            batch.data_future[:, :self.pred_len], q1s_=q1s_, q2s_=q2s_, q3s_=q3s_
        )

    def predict(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        input_ = self.extract_input(batch)
        output = self(input_)
        return self.scaler.rescale(output, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)

    def get_loss(self, batch, criterion):
        input_ = self.extract_input(batch)
        target = self.extract_target(batch)
        output = self(input_)
        return criterion(output, target)
