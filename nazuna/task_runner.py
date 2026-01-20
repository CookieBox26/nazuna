from abc import ABC, abstractmethod
import dataclasses
import collections
import toml
import math
from pathlib import Path
from datetime import datetime
import torch
from nazuna.data_manager import TimeSeriesDataManager
from nazuna import load_class


class BaseTaskRunner(ABC):
    def __init__(
        self,
        dm: TimeSeriesDataManager,
        device: str = '',
        out_dir: str = '',
        exist_ok: bool = False,
        **kwargs,
    ):
        self.dm = dm
        self.device = device
        if not self.device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not out_dir:
            self.out_path = Path('out') / datetime.now().strftime('%Y%m%d-%H%M%S') / 'task_0'
        else:
            self.out_path = Path(out_dir).expanduser()
        if (not exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')
        self.out_path.mkdir(parents=True, exist_ok=exist_ok)
        self._setup(**kwargs)

    @abstractmethod
    def _setup(self, **kwargs):
        pass

    @abstractmethod
    def _run(self):
        pass

    def run(self):
        self._run()


class EvalTaskRunner(BaseTaskRunner):
    """
    Evaluate a model on a specified data range.
    """
    def _setup(self, n_channel: int, seq_len: int, pred_len: int, period_len: int = 24):
        assert self.dm.n_channel == n_channel, f'{self.dm.n_channel=}, {n_channel=}'
        assert self.dm.seq_len >= seq_len
        assert self.dm.pred_len >= pred_len
        self.n_channel = n_channel
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period_len = period_len

    def set_data_loader_eval(self):
        self.data_loader_eval = self.dm.get_data_loader(  # TODO: Specify parameters from conf
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

    def _run(self):
        self.set_data_loader_eval()
        self.criterion = load_class('nazuna.criteria.MAELoss').create(  # TODO: Specify from conf
            self.device,
            n_channel=self.n_channel,
            pred_len=self.pred_len,
            decay_rate=None,
        )
        self.model = load_class('nazuna.models.simple_average.SimpleAverage').create(
            self.device,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            period_len=self.period_len,
        )
        loss = self.eval()
        return loss


class TrainTaskRunner(EvalTaskRunner):
    """
    Train a model on a specified data range.
    """
    def set_data_loader_train(self):
        self.data_loader_train = self.dm.get_data_loader(  # TODO: Specify parameters from conf
            data_range=(0.0, 0.8),
            batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
            batch_sampler_kwargs={'batch_size': 16},
            offset=0, rolling_window=28, device=self.device,
        )

    def __init__(self, dm: TimeSeriesDataManager, conf):
        super().__init__(dm, conf)
        self.set_data_loader_train()
        self.data_loader_eval = None
        if True:  # If using early stopping with validation data (TODO: Control from conf)
            self.set_data_loader_eval()

    def _run(self):
        pass


class OptunaTaskRunner(BaseTaskRunner):
    """
    Search for optimal hyperparameters.
    """
    def _run(self):
        pass


class DiagnosticsTaskRunner(BaseTaskRunner):
    """
    Diagnose data (details TBD).
    """
    def _run(self):
        pass


@dataclasses.dataclass
class Config:
    out_dir: str | Path
    exist_ok: bool
    data: dict
    tasks: list[dict]

    def __post_init__(self):
        if not self.out_dir:
            self.out_path = Path('out') / datetime.now().strftime('%Y%m%d-%H%M%S')
        else:
            self.out_path = Path(self.out_dir).expanduser()
        if (not self.exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')
        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        self.to_toml()
        for key in ['seq_len', 'pred_len']:
            self.data[key] = Config.prod_if_list(self.data[key])

    @staticmethod
    def prod_if_list(x):
        return math.prod(x) if isinstance(x, list) else x

    @classmethod
    def from_toml(cls, toml_path: str | Path):
        d = toml.loads(Path(toml_path).read_text(encoding='utf8'))
        return Config(**d)

    def to_toml(self):
        od = collections.OrderedDict()
        od['out_dir'] = self.out_dir
        od['exist_ok'] = self.exist_ok
        od['data'] = self.data
        od['tasks'] = self.tasks
        toml_path = (self.out_path / 'config.toml')
        toml_path.write_text(toml.dumps(od), newline='\n', encoding='utf8')


def run_tasks(conf: Config):
    dm = TimeSeriesDataManager(**conf.data)
