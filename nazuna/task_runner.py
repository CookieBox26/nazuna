from abc import ABC, abstractmethod
import torch
from nazuna.data_manager import TimeSeriesDataManager
from nazuna import load_class
from types import SimpleNamespace


class BaseTaskRunner(ABC):
    def __init__(self, dm: TimeSeriesDataManager, conf):
        # TODO: conf に device の指定があれば優先
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dm = dm
        self.conf = SimpleNamespace(**conf)

    @abstractmethod
    def run(self):
        pass


class EvalTaskRunner(BaseTaskRunner):
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

    def __init__(self, dm: TimeSeriesDataManager, conf):
        super().__init__(dm, conf)
        assert self.dm.n_channel == self.conf.n_channel
        assert self.dm.seq_len >= self.conf.seq_len
        assert self.dm.pred_len >= self.conf.pred_len

        self.set_data_loader_eval()
        self.criterion = load_class('nazuna.criteria.MSELoss').create(  # TODO: conf から指定
            self.device,
            n_channel=self.conf.n_channel,
            pred_len=self.conf.pred_len,
            decay_rate=None,
        )
        self.model = load_class('nazuna.models.simple_average.SimpleAverage').create(
            self.device,
            seq_len=self.conf.seq_len,
            pred_len=self.conf.pred_len,
            period_len=7,
        )

    def run(self):
        loss = self.eval()
        print(loss)


class TrainTaskRunner(EvalTaskRunner):
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
        if True:  # TODO: conf で制御
            self.set_data_loader_eval()

    def run(self):
        pass
