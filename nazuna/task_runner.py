from abc import ABC, abstractmethod
from enum import Enum
import dataclasses
import toml
import copy
from pathlib import Path
import datetime
import inspect
import torch
from nazuna.datasets import get_path
from nazuna.data_manager import TimeSeriesDataManager
from nazuna import load_class, measure_time


def _to_snake_case(s):
    return '_'.join(s.lower().split())


def _get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def _get_params(func):
    sig = inspect.signature(func)
    params_required = []
    params_optional = []
    for p in sig.parameters.values():
        if p.name == 'self':
            continue
        if p.default is inspect._empty:
            params_required.append(p.name)
        else:
            params_optional.append(p.name)
    return params_required, params_optional


def _validate_params(func, params):
    params_required, params_optional = _get_params(func)
    for p in params_required:
        assert p in params, p
    for p in params:
        assert (p in params_required) or (p in params_optional), p


@dataclasses.dataclass
class BaseTaskRunner(ABC):
    dm: TimeSeriesDataManager
    device: str = ''
    name: str = ''
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
        if self.result:
            raise RuntimeError('Running TaskRunner more than once is not supported.')
        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        with measure_time(self.result):
            self._run()
        self.result_path = self.out_path / 'result.toml'
        self.result_path.write_text(toml.dumps(self.result), newline='\n', encoding='utf8')
        elapsed = self.result['elapsed']
        print(f'[Task] Finished task: {self.result_path.as_posix()} ({elapsed})')


@dataclasses.dataclass
class EvalTaskRunner(BaseTaskRunner):
    """
    Evaluate a model on a specified data range.
    """
    data_range_eval: tuple[float, float] | None = None
    data_offset_eval: int = 0
    data_rolling_window_eval: int = 4
    batch_size_eval: int = 32

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

        if type(self) is EvalTaskRunner:
            assert self.data_range_eval is not None

        self.criterion_cls = load_class(self.criterion_cls_path)
        _validate_params(self.criterion_cls._setup, self.criterion_params)
        self.model_cls = load_class(self.model_cls_path)
        _validate_params(self.model_cls._setup, self.model_params)

        criterion_n_channel = self.criterion_params.get('n_channel', None)
        assert (criterion_n_channel is None) or (criterion_n_channel == self.dm.n_channel)
        criterion_pred_len = self.criterion_params.get('pred_len', None)
        assert (criterion_pred_len is None) or (criterion_pred_len <= self.dm.pred_len)

        model_seq_len = self.model_params.get('seq_len', None)
        assert (model_seq_len is None) or (model_seq_len <= self.dm.seq_len)
        model_pred_len = self.model_params.get('pred_len', None)
        assert (model_pred_len is None) or (model_pred_len <= self.dm.pred_len)

    def set_data_loader_eval(self):
        self.data_loader_eval = self.dm.get_data_loader(
            data_range=self.data_range_eval,
            batch_sampler_cls=load_class('nazuna.batch_sampler.BatchSampler'),
            batch_sampler_params={'batch_size': self.batch_size_eval},
            offset=self.data_offset_eval,
            rolling_window=self.data_rolling_window_eval,
            device=self.device,
        )

    def eval(self):
        data_loader = self.data_loader_eval
        loss_total = 0.0
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                loss = self.model.get_loss(batch, self.criterion)
                loss_total += loss.batch_sum()
        return {
            'n_sample': data_loader.dataset.n_sample,
            'loss_total': loss_total,
            'loss_per_sample': loss_total / data_loader.dataset.n_sample,
        }

    def _run(self):
        self.set_data_loader_eval()
        self.criterion = self.criterion_cls.create(self.device, **self.criterion_params)
        self.model = self.model_cls.create(self.device, **self.model_params)
        loss_eval = self.eval()
        self.result.update(loss_eval)


@dataclasses.dataclass
class TrainTaskRunner(EvalTaskRunner):
    """
    Train a model on a specified data range.
    """
    data_range_train: tuple[int, int] = None
    data_offset_train: int = 0
    data_rolling_window_train: int = 4

    batch_sampler_cls_path: str = ''
    batch_sampler_params: dict = None

    optimizer_cls_path: str = ''
    optimizer_params: dict = None

    lr_scheduler_cls_path: str = ''
    lr_scheduler_params: dict = None

    n_epoch: int = 0
    early_stop: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.data_range_train is not None
        self.batch_sampler_cls = load_class(self.batch_sampler_cls_path)
        self.optimizer_cls = load_class(self.optimizer_cls_path)
        self.lr_scheduler_cls = None
        if self.lr_scheduler_cls_path:
            self.lr_scheduler_cls = load_class(self.lr_scheduler_cls_path)
        assert self.n_epoch > 0

    def set_data_loader_train(self):
        self.data_loader_train = self.dm.get_data_loader(
            data_range=self.data_range_train,
            batch_sampler_cls=self.batch_sampler_cls,
            batch_sampler_params=self.batch_sampler_params,
            offset=self.data_offset_train,
            rolling_window=self.data_rolling_window_train,
            device=self.device,
        )

    def train(self):
        data_loader = self.data_loader_train
        loss_total = 0.0
        for i_batch, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            loss = self.model.get_loss_and_backward(batch, self.criterion)
            loss_total += loss.batch_sum()
            self.optimizer.step()
        return {
            'n_sample': data_loader.dataset.n_sample,
            'loss_total': loss_total,
            'loss_per_sample': loss_total / data_loader.dataset.n_sample,
        }

    def _run(self):
        self.set_data_loader_train()
        if self.data_range_eval is not None:
            self.set_data_loader_eval()

        self.criterion = self.criterion_cls.create(self.device, **self.criterion_params)
        self.model = self.model_cls.create(self.device, **self.model_params)
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_params)
        self.lr_scheduler = None
        if self.lr_scheduler_cls:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_params)

        loss_per_sample_eval_best = float('inf')
        early_stop_counter = 0
        stop = False

        self.result['epochs'] = []
        for i_epoch in range(self.n_epoch):
            self.result['epochs'].append({})

            loss_train = self.train()
            self.result['epochs'][-1]['train'] = loss_train

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.data_range_eval is None:
                continue

            loss_eval = self.eval()
            self.result['epochs'][-1]['eval'] = loss_eval
            loss_per_sample_eval = loss_eval['loss_per_sample']

            if loss_per_sample_eval < loss_per_sample_eval_best:
                # TODO: save model state
                loss_per_sample_eval_best = loss_per_sample_eval
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if (self.early_stop) and (early_stop_counter >= 5):
                stop = True
            if stop:
                break


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
        self.to_toml_path()

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
        params.setdefault('name', f'Task {i_task}')
        params.setdefault('out_dir', self.out_path / _to_snake_case(params['name']))
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

    def to_toml_str(self):
        toml_str = toml.dumps({
            'out_dir': self.out_dir,
            'exist_ok': self.exist_ok,
            'data': self.data,
            'device': self.device,
        })
        toml_str += '\n'
        toml_str += toml.dumps({'tasks': self.tasks})
        toml_str = toml_str.replace('[[tasks]]', '\n\n[[tasks]]')
        return toml_str

    def to_toml_path(self):
        self.conf_path = self.out_path / 'config.toml'
        self.conf_path.write_text(self.to_toml_str(), newline='\n', encoding='utf8')


def run_tasks(conf_: Config | dict | Path | str):
    conf = Config.create(conf_)

    dm = TimeSeriesDataManager(**conf.get_data_param())
    task_runners = []
    for i_task, _ in enumerate(conf.tasks):
        cls_, params_ = conf.get_task_runner(i_task)
        task_runners.append(cls_(dm=dm, **params_))

    result = {}
    with measure_time(result):
        for task_runner in task_runners:
            task_runner.run()

    report_path = conf.out_path / 'report.md'
    with report_path.open('w', newline='\n', encoding='utf8') as f:
        f.write('### Configuration\n')
        f.write('```toml\n')
        f.write(conf.to_toml_str())
        f.write('```\n')
        f.write('\n')
        f.write('### Result\n')
        for task_runner in task_runners:
            f.write(f'#### {task_runner.name}\n')
            f.write('```toml\n')
            f.write(toml.dumps(task_runner.result))
            f.write('```\n')
    elapsed = result['elapsed']
    print(f'Finished all tasks: {report_path.as_posix()} ({elapsed})')
