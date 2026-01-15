from abc import ABC, abstractmethod
import torch
from nazuna.data_manager import TimeSeriesDataManager
from nazuna import load_class


class BaseTaskRunner(ABC):
    def __init__(self, dm: TimeSeriesDataManager, device: str = '', **kwargs):
        self.dm = dm
        self.device = device
        if not self.device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.configure(**kwargs)

    @abstractmethod
    def configure(self, **kwargs):
        pass

    @abstractmethod
    def run(self):
        pass


class EvalTaskRunner(BaseTaskRunner):
    """
    データの指定期間でモデルを評価する
    """
    def configure(self, n_channel: int, seq_len: int, pred_len: int):
        assert self.dm.n_channel == n_channel, f'{self.dm.n_channel=}, {n_channel=}'
        assert self.dm.seq_len >= seq_len
        assert self.dm.pred_len >= pred_len
        self.n_channel = n_channel
        self.seq_len = seq_len
        self.pred_len = pred_len

    def set_data_loader_eval(self):
        self.data_loader_eval = self.dm.get_data_loader(  # TODO: conf からパラメータを指定
            data_range=(0.8, 1.0),
            batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
            batch_sampler_kwargs={'batch_size': 16},
            offset=0, rolling_window=0, device=self.device,
        )

    def eval(self):
        data_loader = self.data_loader_eval
        loss_total = 0.0
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                loss, _ = self.model.get_loss(batch, self.criterion)
                loss_total += batch.tsta_future.shape[0] * loss[0].item()
        return loss_total / data_loader.dataset.n_sample

    def run(self):
        self.set_data_loader_eval()
        self.criterion = load_class('nazuna.criteria.MAELoss').create(  # TODO: conf から指定
            self.device,
            n_channel=self.n_channel,
            pred_len=self.pred_len,
            decay_rate=None,
        )
        self.model = load_class('nazuna.models.simple_average.SimpleAverage').create(
            self.device,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            period_len=24,
        )
        loss = self.eval()
        return loss


class TrainTaskRunner(EvalTaskRunner):
    """
    データの指定期間でモデルを訓練する
    """
    def set_data_loader_train(self):
        self.data_loader_train = self.dm.get_data_loader(  # TODO: conf からパラメータを指定
            data_range=(0.0, 0.8),
            batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
            batch_sampler_kwargs={'batch_size': 16},
            offset=0, rolling_window=28, device=self.device,
        )

    def __init__(self, dm: TimeSeriesDataManager, conf):
        super().__init__(dm, conf)
        self.set_data_loader_train()
        self.data_loader_eval = None
        if True:  # 評価期間をとって Early Stopping する場合 (TODO: conf で制御)
            self.set_data_loader_eval()

    def run(self):
        pass


class OptunaTaskRunner(BaseTaskRunner):
    """
    ハイパーパラメータ探索する
    """
    def run(self):
        pass


class DiagnosticsTaskRunner(BaseTaskRunner):
    """
    データを診断する (内容未定)
    """
    def run(self):
        pass
