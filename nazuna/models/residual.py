from nazuna.models.base import BasicBaseModel
from nazuna.scaler import IqrScaler
from nazuna import load_class


def _make_concrete(cls):
    """Create a concrete class from an abstract base class by providing dummy implementations."""
    class ConcreteModel(cls):
        def predict(self, batch):
            raise NotImplementedError('Not used in ResidualModel')
    return ConcreteModel


class ResidualModel(BasicBaseModel):
    """
    Residual learning framework that combines a naive model and a neural model.

    The final prediction is: naive_output + neural_output

    Both sub-models receive the same scaled input and their outputs are summed.
    """

    def _setup(
        self,
        seq_len: int,
        pred_len: int,
        quantile_mode_train: str,
        quantile_mode_eval: str,
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
        super()._setup(seq_len, pred_len)
        self.scaler = IqrScaler(quantile_mode_train, quantile_mode_eval)

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

        return naive_out + neural_out, {}

    def predict(self, batch):
        input_ = self._extract_input(batch)
        output, info = self(input_)
        output = self.scaler.rescale(output, batch)
        return output, info
