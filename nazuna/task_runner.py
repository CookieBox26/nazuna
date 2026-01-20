from abc import ABC, abstractmethod
from enum import Enum
import dataclasses
import toml
import copy
from pathlib import Path
import datetime
import torch
from nazuna.datasets import get_path
from nazuna.data_manager import TimeSeriesDataManager
from nazuna import load_class, measure_time


def _get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


@dataclasses.dataclass
class BaseTaskRunner(ABC):
    dm: TimeSeriesDataManager
    device: str = ''
    out_dir: str | Path = ''
    exist_ok: bool = False

    def __post_init__(self):
        self.device = self.device or str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.out_path = Path(self.out_dir or f'out/{_get_timestamp()}/task_0/').expanduser()
        if (not self.exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')
        self.result = {}

    @abstractmethod
    def _run(self):
        pass

    def run(self):
        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        with measure_time(self.result):
            self._run()
        self.result_path = self.out_path / 'result.toml'
        self.result_path.write_text(toml.dumps(self.result), newline='\n', encoding='utf8')


@dataclasses.dataclass
class EvalTaskRunner(BaseTaskRunner):
    """
    Evaluate a model on a specified data range.
    """
    data_range: tuple[int, int] = None
    data_offset: int = 0
    data_rolling_window: int = 4
    batch_size: int = 32

    criterion_cls_path: str = 'nazuna.criteria.MAELoss'
    criterion_params: dict = None

    model_cls_path: str = 'nazuna.models.simple_average.SimpleAverage'
    model_params: dict = None

    n_channel: int = -1
    seq_len: int = -1
    pred_len: int = -1
    period_len: int = -1

    def __post_init__(self):
        super().__post_init__()
        assert self.data_range is not None
        # assert self.dm.n_channel == self.n_channel, f'{self.dm.n_channel=}, {self.n_channel=}'
        # assert self.dm.seq_len >= self.seq_len
        # assert self.dm.pred_len >= self.pred_len
        self.criterion_cls = load_class(self.criterion_cls_path)
        assert self.criterion_params is not None
        self.model_cls = load_class(self.model_cls_path)
        assert self.model_params is not None

    def set_data_loader_eval(self):
        self.data_loader_eval = self.dm.get_data_loader(  # TODO: Specify parameters from conf
            data_range=self.data_range,
            batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
            batch_sampler_params={'batch_size': self.batch_size},
            offset=self.data_offset, rolling_window=self.data_rolling_window, device=self.device,
        )

    def eval(self):
        data_loader = self.data_loader_eval
        loss = 0.0
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                loss_, _ = self.model.get_loss(batch, self.criterion)
                loss += batch.tsta_future.shape[0] * loss_[0].item()
        self.result['n_sample'] = data_loader.dataset.n_sample
        self.result['loss'] = loss
        self.result['loss_per_sample'] = loss / data_loader.dataset.n_sample

    def _run(self):
        self.set_data_loader_eval()
        self.criterion = self.criterion_cls.create(self.device, **self.criterion_params)
        self.model = self.model_cls.create(self.device, **self.model_params)
        self.eval()


@dataclasses.dataclass
class TrainTaskRunner(EvalTaskRunner):
    """
    Train a model on a specified data range.
    """
    def set_data_loader_train(self):
        self.data_loader_train = self.dm.get_data_loader(  # TODO: Specify parameters from conf
            data_range=(0.0, 0.8),
            batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
            batch_sampler_params={'batch_size': 16},
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


@dataclasses.dataclass
class OptunaTaskRunner(BaseTaskRunner):
    """
    Search for optimal hyperparameters.
    """
    def _run(self):
        pass


@dataclasses.dataclass
class DiagnosticsTaskRunner(BaseTaskRunner):
    """
    Diagnose data (details TBD).
    """
    def _run(self):
        pass


class TaskType(Enum):
    eval = EvalTaskRunner
    train = TrainTaskRunner


@dataclasses.dataclass
class Config:
    out_dir: str | Path = ''
    exist_ok: bool = False
    data: dict = None
    device: str = ''
    tasks: list[dict] = None

    def __post_init__(self):
        self.out_dir = self.out_dir or f'out/{_get_timestamp()}/'
        self.out_path = Path(self.out_dir).expanduser()
        if (not self.exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')
        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        assert self.data is not None
        self.device = self.device or str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        assert self.tasks is not None
        self.to_toml()

    def get_data_param(self):
        param = copy.deepcopy(self.data)
        if isinstance(param['path'], (list, tuple)):
            param['path'] = get_path(*param['path'])
        return param

    def get_task_runner(self, i_task):
        params = copy.deepcopy(self.tasks[i_task])
        self.task_type = params.pop('task_type')
        task_runner_cls = TaskType[self.task_type].value
        params.setdefault('device', self.device)
        params.setdefault('out_dir', self.out_path / f'task_{i_task}')
        params.setdefault('exist_ok', self.exist_ok)
        return task_runner_cls, params

    @classmethod
    def from_toml_str(cls, toml_str: str | Path):
        d = toml.loads(toml_str)
        return cls(**d)

    @classmethod
    def from_toml_path(cls, toml_path: str | Path):
        return cls.from_toml_str(Path(toml_path).read_text(encoding='utf8'))

    @classmethod
    def create(cls, source):
        if type(source) is cls:
            return source
        if isinstance(source, dict):
            return cls(**source)
        if Path(source).is_file():
            return cls.from_toml_path(source)
        if isinstance(source, str):
            return cls.from_toml_str(source)
        raise ValueError('Cannot cast to Config')

    def to_toml(self):
        toml_str = toml.dumps({
            'out_dir': self.out_dir,
            'exist_ok': self.exist_ok,
            'data': self.data,
            'device': self.device,
        })
        toml_str += '\n'
        toml_str += toml.dumps({'tasks': self.tasks})
        self.conf_path = self.out_path / 'config.toml'
        self.conf_path.write_text(toml_str, newline='\n', encoding='utf8')


def run_tasks(conf_: Config | dict | Path | str):
    conf = Config.create(conf_)

    dm = TimeSeriesDataManager(**conf.get_data_param())
    runners = []
    for i_task, _ in enumerate(conf.tasks):
        cls_, params_ = conf.get_task_runner(i_task)
        runners.append(cls_(dm=dm, **params_))

    for runner in runners:
        runner.run()

    report_path = conf.out_path / 'report.md'
    report_path.write_text('hello', newline='\n', encoding='utf8')
