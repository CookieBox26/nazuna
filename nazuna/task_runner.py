from abc import ABC, abstractmethod
from typing import IO
import os
from enum import Enum
import dataclasses
import toml
import copy
from pathlib import Path
import datetime
import inspect
import pickle
import numpy as np
import optuna
import torch
from nazuna.datasets import get_path
from nazuna.data_manager import TimeSeriesDataManager
from nazuna.criteria import BaseImprovement
from nazuna import fix_seed, load_class, measure_time
from nazuna.utils.optuna_helper import OptunaHelper
from nazuna.utils.diagnoser import Diagnoser
from nazuna.utils.report import report
try:
    from nazuna.utils.inspector import Inspector
except ImportError:
    Inspector = None


def _to_snake_case(s):
    s = s.translate(str.maketrans('()=', '___'))
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
    """
    Base class for task runners.  
    Subclasses must follow these rules:  

    - A TaskRunner should be runnable on its own with just a `TimeSeriesDataManager` instance.
    - Calling `run()` writes `result.toml` (and other artifacts if any) to `out_dir`.

    Attributes:
        dm (TimeSeriesDataManager): TimeSeriesDataManager instance **(required)**.
        device (str = ''): Device name for computation (Ex. 'cpu', 'cuda').
            If not specified, it will be automatically detected from your environment.
        name (str = ''): Name of this task. Not used when running a task standalone.
        out_dir (str | Path = ''): Output path for this task's artifacts.
            Defaults to 'out/YYYYmmdd-HHMMSS/task_0/' if not specified.
        exist_ok (bool = False): Whether to allow the output path to already exist.
    """
    dm: TimeSeriesDataManager
    device: str = None
    name: str = None
    out_dir: str | Path = None
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

    Attributes:
        data_range_eval (tuple[float, float] | None = None): Data range for evaluation
            as (start, end) ratios **(required for EvalTaskRunner; optional for TrainTaskRunner)**.
        data_offset_eval (int = 0): Offset for evaluation data.
        data_rolling_window_eval (int = 4): Rolling window size for computing quartiles for scaling
            (unused if quartile-based rolling-window scaling is disabled).
        batch_size_eval (int = 32): Batch size for evaluation.
        criterion_cls_path (str = None): Class path for the criterion (Ex. 'nazuna.criteria.MAELoss')
            **(required)**.
        criterion_params (dict = None): Parameters for the criterion **(required)**.
        baseline_model_cls_path (str = None): Class path for the baseline model.
            Required only when the criterion requires a baseline.
        baseline_model_params (dict = None): Parameters for the baseline model.
        model_cls_path (str = None): Class path for the model
            (Ex. 'nazuna.models.simple_average.SimpleAverage') **(required)**.
        model_params (dict = None): Parameters for the model **(required)**.
        model_state_path (str | PathLike | IO[bytes] = None): Path to the model state file.
    """
    data_range_eval: tuple[float, float] | None = None
    data_offset_eval: int = 0
    data_rolling_window_eval: int = 4
    batch_size_eval: int = 32

    criterion_cls_path: str = None
    criterion_params: dict = None

    baseline_model_cls_path: str = None
    baseline_model_params: dict = None

    model_cls_path: str = None
    model_params: dict = None
    model_state_path: str | os.PathLike[str] | IO[bytes] = None

    def __post_init__(self):
        super().__post_init__()

        if type(self) is EvalTaskRunner:
            assert self.data_range_eval is not None

        self.criterion_cls = load_class(self.criterion_cls_path)
        _validate_params(self.criterion_cls._setup, self.criterion_params)

        self.eval_improvement = issubclass(self.criterion_cls, BaseImprovement)
        if self.eval_improvement:
            self.baseline_model_cls = load_class(self.baseline_model_cls_path)
            _validate_params(self.baseline_model_cls._setup, self.baseline_model_params)

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

    def eval(
        self,
        output_loss_per_channel=True,
        output_scaled_loss=True,
    ):
        data_loader = self.data_loader_eval
        loss_total = 0.0
        loss_scaled_total = 0.0
        loss_per_channel_total = None
        loss_scaled_per_channel_total = None
        sample_saved = False
        self.model.eval()
        with torch.no_grad():
            for i_batch, batch in enumerate(data_loader):
                true = self.model.extract_true(batch)
                pred, _ = self.model.predict(batch)
                if self.eval_improvement:
                    baseline, _ = self.baseline_model.predict(batch)
                    loss = self.criterion(baseline, pred, true)
                else:
                    baseline = None
                    loss = self.criterion(pred, true)
                loss_total += loss.batch_sum()
                if output_loss_per_channel and loss.each_sample_channel is not None:
                    batch_channel_sum = loss.each_sample_channel.sum(dim=0)
                    if loss_per_channel_total is None:
                        loss_per_channel_total = batch_channel_sum
                    else:
                        loss_per_channel_total += batch_channel_sum

                if output_scaled_loss:
                    # batch.data: (batch_size, seq_len, n_channel)
                    data = batch.data
                    q2 = data.median(dim=1, keepdim=True).values
                    q1 = data.quantile(0.25, dim=1, keepdim=True)
                    q3 = data.quantile(0.75, dim=1, keepdim=True)
                    iqr = q3 - q1
                    pred_s = (pred - q2) / iqr
                    true_s = (true - q2) / iqr
                    if self.eval_improvement:
                        baseline_s = (baseline - q2) / iqr
                        loss_s = self.criterion(baseline_s, pred_s, true_s)
                    else:
                        loss_s = self.criterion(pred_s, true_s)
                    loss_scaled_total += loss_s.batch_sum()
                    if (output_loss_per_channel
                            and loss_s.each_sample_channel is not None):
                        batch_channel_sum_s = \
                            loss_s.each_sample_channel.sum(dim=0)
                        if loss_scaled_per_channel_total is None:
                            loss_scaled_per_channel_total = batch_channel_sum_s
                        else:
                            loss_scaled_per_channel_total += batch_channel_sum_s

                if not sample_saved:
                    save_data = {
                        'pred': pred[0].cpu().numpy(),
                        'data': batch.data[0].cpu().numpy(),
                        'data_future': batch.data_future[0].cpu().numpy(),
                    }
                    if baseline is not None:
                        save_data['baseline'] = baseline[0].cpu().numpy()
                    np.savez(self.out_path / 'pred_0_0.npz', **save_data)
                    sample_saved = True

        n_sample = data_loader.dataset.n_sample
        result = {
            'n_sample': n_sample,
            'loss_total': loss_total,
            'loss_per_sample': loss_total / n_sample,
        }

        if output_scaled_loss:
            result['loss_scaled_total'] = loss_scaled_total
            result['loss_scaled_per_sample'] = loss_scaled_total / n_sample
            if loss_scaled_per_channel_total is not None:
                loss_scaled_per_channel = (
                    loss_scaled_per_channel_total / n_sample
                ).cpu().tolist()
                result['loss_scaled_per_channel'] = dict(
                    zip(self.dm.cols, loss_scaled_per_channel),
                )

        if loss_per_channel_total is not None:
            loss_per_channel = (loss_per_channel_total / n_sample).cpu().tolist()
            result['loss_per_channel'] = dict(zip(self.dm.cols, loss_per_channel))

        return result

    def _run(self):
        self.set_data_loader_eval()
        self.criterion = self.criterion_cls.create(self.device, **self.criterion_params)
        if self.eval_improvement:
            self.baseline_model = self.baseline_model_cls.create(
                self.device, **self.baseline_model_params,
            )
        self.model = self.model_cls.create(self.device, self.model_state_path, **self.model_params)
        loss_eval = self.eval()
        torch.save(self.model.state_dict(), self.out_path / 'model_state.pth')  # for debugging

        self.result['cols_org'] = dict(zip(self.dm.cols, self.dm.cols_org))
        self.result['data_range_eval'] = self.data_loader_eval.dataset.info
        self.result.update(loss_eval)


@dataclasses.dataclass
class TrainTaskRunner(EvalTaskRunner):
    """
    Train a model on a specified data range.

    Attributes:
        data_range_train (tuple[float, float] | None = None): Data range for training
            as (start, end) ratios **(required)**.
        data_offset_train (int = 0): Offset for training data.
        data_rolling_window_train (int = 4): Rolling window size for computing quartiles for scaling
            (unused if quartile-based rolling-window scaling is disabled).
        batch_sampler_cls_path (str = ''): Class path for the batch sampler
            (Ex. 'nazuna.batch_sampler.BatchSamplerShuffle') **(required)**.
        batch_sampler_params (dict = None): Parameters for the batch sampler **(required)**.
        optimizer_cls_path (str = ''): Class path for the optimizer
            (Ex. 'torch.optim.Adam') **(required)**.
        optimizer_params (dict = None): Parameters for the optimizer **(required)**.
        lr_scheduler_cls_path (str = ''): Class path for the learning rate scheduler
            (Ex. 'torch.optim.lr_scheduler.CosineAnnealingLR'). Optional.
        lr_scheduler_params (dict = None): Parameters for the learning rate scheduler.
        n_epoch (int = 0): Number of training epochs **(required)**.
        early_stop (bool = False): Whether to enable early stopping.
            Stops training if evaluation loss does not improve for 5 consecutive epochs.
    """
    data_range_train: tuple[int, int] = None
    data_offset_train: int = 0
    data_rolling_window_train: int = 4

    batch_sampler_cls_path: str = None
    batch_sampler_params: dict = None

    optimizer_cls_path: str = None
    optimizer_params: dict = None

    lr_scheduler_cls_path: str = None
    lr_scheduler_params: dict = None

    n_epoch: int = 0
    n_epoch_path: str | Path = None
    n_epoch_path_defer: bool = False
    early_stop: bool = False
    patience: int = 5

    def __post_init__(self):
        super().__post_init__()
        assert self.data_range_train is not None
        self.batch_sampler_cls = load_class(self.batch_sampler_cls_path)
        self.optimizer_cls = load_class(self.optimizer_cls_path)
        self.lr_scheduler_cls = None
        if self.lr_scheduler_cls_path:
            self.lr_scheduler_cls = load_class(self.lr_scheduler_cls_path)
        if self.n_epoch_path is None:
            assert self.n_epoch > 0
        else:
            assert self.n_epoch == 0
            if type(self.n_epoch_path) is str:
                self.n_epoch_path = Path(self.n_epoch_path)
            if not self.n_epoch_path_defer:
                assert self.n_epoch_path.is_file()

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
        self.model.train()
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
        if self.n_epoch_path is not None:
            self.n_epoch = toml.loads(
                Path(self.n_epoch_path).read_text(encoding='utf8')
            )['i_epoch_best'] + 1

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

        self.result['cols_org'] = dict(zip(self.dm.cols, self.dm.cols_org))
        self.result['data_range_train'] = self.data_loader_train.dataset.info

        loss_history = []
        for i_epoch in range(self.n_epoch):
            print(f'----- Epoch {i_epoch} -----')
            epoch_record = {'i_epoch': i_epoch}

            loss_train = self.train()
            epoch_record['train'] = loss_train

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.data_range_eval is None:
                loss_history.append(epoch_record)
                continue

            loss_eval = self.eval(output_loss_per_channel=False, output_scaled_loss=False)
            epoch_record['eval'] = loss_eval
            loss_history.append(epoch_record)
            loss_per_sample_eval = loss_eval['loss_per_sample']

            if loss_per_sample_eval < loss_per_sample_eval_best:
                loss_per_sample_eval_best = loss_per_sample_eval
                early_stop_counter = 0
                self.result['i_epoch_best'] = i_epoch
                self.result['n_sample_eval'] = self.data_loader_eval.dataset.n_sample
                self.result['loss_per_sample_eval_best'] = loss_per_sample_eval_best
                torch.save(self.model.state_dict(), self.out_path / 'model_state.pth')
            else:
                early_stop_counter += 1
            if (self.early_stop) and (early_stop_counter >= self.patience):
                stop = True
            if stop:
                break

        history_path = self.out_path / 'train_loss_history.toml'
        history_path.write_text(
            toml.dumps({'epochs': loss_history}),
            newline='\n', encoding='utf8',
        )

        if self.data_range_eval is None:
            torch.save(self.model.state_dict(), self.out_path / 'model_state.pth')


@dataclasses.dataclass
class DiagnosticsTaskRunner(BaseTaskRunner):
    """
    Diagnose data characteristics such as seasonality.

    Attributes:
        data_range_diag (tuple[float, float] | None = None): Data range for diagnostics
            as (start, end) ratios **(required)**.
        period (int | None = None): Seasonal period for STL decomposition **(required)**.
    """
    data_range_diag: tuple[float, float] | None = None
    period: int | None = None

    def _run(self):
        diagnoser = Diagnoser(self.dm, self.data_range_diag)
        diagnostics, data = diagnoser.run(period=self.period)
        self.result.update(diagnostics)
        np.savez(
            self.out_path / 'sample.npz',
            values=data['values'],
            columns=data['columns'],
            timestamps=data['timestamps'],
        )


@dataclasses.dataclass
class OptunaTaskRunner(BaseTaskRunner):
    """
    Search for optimal hyperparameters using Optuna.

    Attributes:
        n_trials (int = 10): Number of Optuna trials to run.
        direction (str = 'minimize'): Optimization direction ('minimize' or 'maximize').
        search_space (dict = None): Hyperparameter search space definition **(required)**.
            Keys are parameter names, values are lists like ['log_uniform', low, high],
            ['uniform', low, high], ['int', low, high], or ['categorical', choices].
        data_ranges (list = None): List of data range dicts for cross-validation **(required)**.
            Each dict should have 'train' and 'eval' keys with (start, end) ratio tuples.
        data_offset_train (int = 0): Offset for training data.
        data_rolling_window_train (int = 4): Rolling window size for training data.
        data_offset_eval (int = 0): Offset for evaluation data.
        data_rolling_window_eval (int = 4): Rolling window size for evaluation data.
        batch_size_eval (int = 32): Batch size for evaluation.
        criterion_cls_path (str): Class path for the criterion **(required)**.
        criterion_params (dict = None): Parameters for the criterion.
        model_cls_path (str = ''): Class path for the model **(required)**.
        model_params (dict = None): Base parameters for the model.
            Search space parameters will be merged into this.
        batch_sampler_cls_path (str = ''): Class path for the batch sampler **(required)**.
        batch_sampler_params (dict = None): Base parameters for the batch sampler.
        optimizer_cls_path (str = ''): Class path for the optimizer **(required)**.
        optimizer_params (dict = None): Base parameters for the optimizer.
        lr_scheduler_cls_path (str = ''): Class path for the learning rate scheduler. Optional.
        lr_scheduler_params (dict = None): Parameters for the learning rate scheduler.
        n_epoch (int = 0): Number of training epochs per trial **(required)**.
        early_stop (bool = False): Whether to enable early stopping within each trial.
    """
    n_trials: int = 10
    direction: str = 'minimize'
    search_space: dict = None
    data_ranges: list = None

    data_offset_train: int = 0
    data_rolling_window_train: int = 4
    data_offset_eval: int = 0
    data_rolling_window_eval: int = 4
    batch_size_eval: int = 32

    criterion_cls_path: str = 'nazuna.criteria.MAE'
    criterion_params: dict = None

    model_cls_path: str = ''
    model_params: dict = None

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
        assert self.search_space is not None, 'search_space is required'
        assert self.data_ranges is not None, 'data_ranges is required'
        assert len(self.data_ranges) > 0, 'data_ranges must not be empty'
        assert self.n_trials > 0, 'n_trials must be positive'
        assert self.n_epoch > 0, 'n_epoch must be positive'

        self.criterion_cls = load_class(self.criterion_cls_path)
        self.model_cls = load_class(self.model_cls_path)
        self.batch_sampler_cls = load_class(self.batch_sampler_cls_path)
        self.optimizer_cls = load_class(self.optimizer_cls_path)
        self.lr_scheduler_cls = None
        if self.lr_scheduler_cls_path:
            self.lr_scheduler_cls = load_class(self.lr_scheduler_cls_path)

        self._best_model_state = None
        self._best_trial_number = None

    def _create_objective(self):
        def objective(trial):
            model_params, optimizer_params, batch_sampler_params = \
                OptunaHelper.build_params_for_trial(
                    trial,
                    self.search_space,
                    self.model_params,
                    self.optimizer_params,
                    self.batch_sampler_params,
                )

            losses = []
            best_model_state_this_trial = None
            best_loss_this_trial = float('inf')

            for i_fold, data_range in enumerate(self.data_ranges):
                data_range_train = data_range['train']
                data_range_eval = data_range['eval']

                runner = TrainTaskRunner(
                    dm=self.dm,
                    device=self.device,
                    name=f'Trial {trial.number} Fold {i_fold}',
                    out_dir=self.out_path / f'trial_{trial.number}' / f'fold_{i_fold}',
                    exist_ok=self.exist_ok,
                    data_range_train=data_range_train,
                    data_range_eval=data_range_eval,
                    data_offset_train=self.data_offset_train,
                    data_rolling_window_train=self.data_rolling_window_train,
                    data_offset_eval=self.data_offset_eval,
                    data_rolling_window_eval=self.data_rolling_window_eval,
                    batch_size_eval=self.batch_size_eval,
                    criterion_cls_path=self.criterion_cls_path,
                    criterion_params=self.criterion_params,
                    model_cls_path=self.model_cls_path,
                    model_params=model_params,
                    batch_sampler_cls_path=self.batch_sampler_cls_path,
                    batch_sampler_params=batch_sampler_params,
                    optimizer_cls_path=self.optimizer_cls_path,
                    optimizer_params=optimizer_params,
                    lr_scheduler_cls_path=self.lr_scheduler_cls_path,
                    lr_scheduler_params=self.lr_scheduler_params,
                    n_epoch=self.n_epoch,
                    early_stop=self.early_stop,
                )
                runner.out_path.mkdir(parents=True, exist_ok=True)
                runner._run()
                fold_loss = runner.result.get(
                    'loss_per_sample_eval_best', float('inf'),
                )
                losses.append(fold_loss)

                if fold_loss < best_loss_this_trial:
                    best_loss_this_trial = fold_loss
                    best_model_state_this_trial = copy.deepcopy(runner.model.state_dict())

            mean_loss = sum(losses) / len(losses)

            if self._best_model_state is None or mean_loss < self.result.get('best_value', float('inf')):
                self._best_model_state = best_model_state_this_trial
                self._best_trial_number = trial.number

            return mean_loss

        return objective

    def _run(self):
        study = optuna.create_study(direction=self.direction)
        study.optimize(self._create_objective(), n_trials=self.n_trials)

        self.result['best_trial_number'] = study.best_trial.number
        self.result['best_value'] = study.best_value
        self.result['best_params'] = study.best_params
        self.result['n_trials'] = len(study.trials)

        trials_history = []
        for t in study.trials:
            trials_history.append({
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': t.state.name,
            })
        self.result['trials'] = trials_history

        if self._best_model_state is not None:
            torch.save(self._best_model_state, self.out_path / 'best_model_state.pth')

        with open(self.out_path / 'study.pkl', 'wb') as f:
            pickle.dump(study, f)


class TaskType(Enum):
    eval = EvalTaskRunner
    train = TrainTaskRunner
    optuna = OptunaTaskRunner
    diag = DiagnosticsTaskRunner


@dataclasses.dataclass
class Config:
    """
    Class that holds a series of task settings.

    Attributes:
        seed (int = 0): Random seed for reproducibility.
        out_dir (str | Path = ''): Output path for the series of tasks.
            Outputs config.toml and report.md here.
            Defaults to 'out/YYYYmmdd-HHMMSS/' if not specified.

            - If individual task output paths are not specified, subdirectories are created
              under this path using task names.
            - You may also create this directory in advance and place a config.toml inside it
              (it will be overwritten with the resolved config.toml).
              In that case, set exist_ok to True.

        exist_ok (bool = False): Whether to allow the output path to already exist.
        data (dict = None): Data configuration for
            [TimeSeriesDataManager](reference.md#nazuna.data_manager.TimeSeriesDataManager)
            **(required)**.
        device (str = ''): Device name for computation (Ex. 'cpu', 'cuda').
            If not specified, it will be automatically detected from your environment.
        tasks (list[dict] = None): List of individual task configurations **(required)**.
            Each dict should have a 'task_type' key with a task type identifier
            (eval, train, optuna, diag), plus the required settings for that task type.
            See [Reference (Task Runners)](reference_task_runners.md) for details.

    !!! warning "About individual task names when running a series of tasks"

        Task names are used in the following cases:

        - If an individual task's output path is not specified, a subdirectory is created.
          The subdirectory name is the task name with symbols escaped and converted to snake_case.
        - You can specify model_state.pth trained in previous tasks by task name.

        Therefore, the following processing is done when creating a Config:

        - If a task name is not specified, it defaults to 'Task i' (0-indexed sequential number).
        - Duplicate task names are not allowed and will raise an error.
    """
    seed: int = 0
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
        assert self.data is not None
        self.device = self.device or str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        assert self.tasks is not None
        self.out_paths = {}
        for i_task, _ in enumerate(self.tasks):
            self.tasks[i_task].setdefault('name', f'Task {i_task}')
            if self.tasks[i_task]['name'] in self.out_paths:
                raise ValueError(f'Duplicate task name: {self.tasks[i_task]["name"]}')
            self.tasks[i_task].setdefault(
                'out_dir', (self.out_path / _to_snake_case(self.tasks[i_task]['name'])).as_posix(),
            )
            self.out_paths[self.tasks[i_task]['name']] = Path(self.tasks[i_task]['out_dir'])

        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
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
        params.setdefault('exist_ok', self.exist_ok)
        if 'n_epoch' in params:
            if isinstance(params['n_epoch'], dict):
                params['n_epoch_path'] = \
                    self.out_paths[params['n_epoch']['task_name']] / 'result.toml'
                params['n_epoch_path_defer'] = True
                del params['n_epoch']
        if 'model_state' in params:
            params['model_state_path'] = \
                self.out_paths[params['model_state']['task_name']] / 'model_state.pth'
            del params['model_state']
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


def run_tasks(
    conf_: Config | dict | Path | str,
    skip_task_ids_: str = '',
):
    conf = Config.create(conf_)
    if '-' in skip_task_ids_:
        a, b = skip_task_ids_.split('-', 1)
        skip_task_ids = list(range(int(a), int(b) + 1))
    else:
        skip_task_ids = [int(i) for i in skip_task_ids_.split(',') if i != '']
    fix_seed(conf.seed)

    dm = TimeSeriesDataManager(**conf.get_data_param())
    task_runners = []
    for i_task, _ in enumerate(conf.tasks):
        cls_, params_ = conf.get_task_runner(i_task)
        task_runners.append(cls_(dm=dm, **params_))

    result = {}
    with measure_time(result):
        for i_task, task_runner in enumerate(task_runners):
            if i_task in skip_task_ids:
                continue
            task_runner.run()

    report_path = conf.out_path / 'report.md'
    report(report_path, conf.to_toml_str(), task_runners)
    print(f'Finished all tasks: {report_path.as_posix()} ({result["elapsed"]})')
