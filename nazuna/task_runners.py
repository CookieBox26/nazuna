from abc import ABC, abstractmethod
from typing import IO, ClassVar
import os
from enum import Enum
import dataclasses
import toml
import copy
from pathlib import Path
import datetime
import inspect
import numpy as np
import optuna
import torch
from nazuna.data_manager import TimeSeriesDataManager
from nazuna.criteria import BaseImprovement
from nazuna.models._base import BasicBaseModel
from nazuna.analysis.diagnoser import Diagnoser
from nazuna.analysis.inspector import Inspector
from nazuna.utils import (
    fix_seed, load_class, measure_time, get_timestamp, get_env_info, load_toml,
)
from sqlalchemy import create_engine, text, bindparam


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
        exist_ok (bool = True): Whether to allow the output path to already exist.
        seed (int = 0): Random seed for reproducibility.
    """
    dm: TimeSeriesDataManager
    device: str = None
    name: str = None
    out_dir: str | Path = None
    exist_ok: bool = True
    seed: int = 0

    def __post_init__(self):
        if not self.device:
            self.device = str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.out_path = Path(self.out_dir or f'out/{get_timestamp()}/task_0/').expanduser()
        if (not self.exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')
        self.log_path = self.out_path / 'log.txt'
        self.result = {}
        self.result_path = type(self).to_result_path(self.out_path)

    @classmethod
    def to_result_path(cls, out_path):
        return out_path / 'result.toml'

    @abstractmethod
    def _run(self):
        pass

    def _log(self, message):
        timestamp = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        cls_name = type(self).__name__
        line = f'{timestamp} [{cls_name}] {message}\n'
        with self.log_path.open('a', newline='\n', encoding='utf8') as f:
            f.write(line)

    def _save_result(self):
        self.result_path.write_text(toml.dumps(self.result), newline='\n', encoding='utf8')

    def run(self):
        if self.result:
            raise RuntimeError('Running TaskRunner more than once is not supported.')
        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        if self.result_path.exists():
            self.result_path.unlink()
        if self.log_path.exists():
            self.log_path.unlink()
        print(f'[Task] Starting task: {self.out_path.as_posix()}')
        self._log('Started')
        self.result['env'] = get_env_info()
        with measure_time(self.result):
            fix_seed(self.seed)
            self._run()
        self._log('Finished')
        self._save_result()
        elapsed = self.result['elapsed']
        print(f'[Task] Finished task: {self.out_path.as_posix()} ({elapsed})')


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
        criterion (dict = None): Criterion configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        baseline_model (dict = None): Baseline model configuration.
            Must have 'cls_path' (str) and 'params' (dict) keys.
            Required only when the criterion requires a baseline.
        model (dict = None): Model configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        model_state_path (str | PathLike | IO[bytes] = None): Path to the model state file.
        dump_pred_data (bool = True): Whether to dump pred_first.npz and pred_last.npz.
    """
    data_range_eval: tuple[float, float] | None = None
    data_offset_eval: int = 0
    data_rolling_window_eval: int = 4
    batch_size_eval: int = 32
    criterion: dict = None
    baseline_model: dict = None
    model: dict = None
    model_state_path: str | os.PathLike[str] | IO[bytes] = None
    inspector_params: dict = None
    dump_pred_data: bool = True

    @classmethod
    def _get_required_params(cls, func):
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values() if p.name != 'self']
        params_required = [p.name for p in params if p.default is inspect._empty]
        params_optional = [p.name for p in params if p.default is not inspect._empty]
        return params_required, params_optional

    @classmethod
    def _validate_params(cls, func, params):
        params_required, params_optional = cls._get_required_params(func)
        for p in params_required:
            assert p in params, p
        for p in params:
            assert (p in params_required) or (p in params_optional), p

    @classmethod
    def extract_model_config(cls, conf):
        cls_, params_ = load_class(conf['cls_path']), conf['params']
        if issubclass(cls_, BasicBaseModel):
            params_ = cls_._resolve_seq_len(params_)
        scaler_cls_path = params_.pop('scaler_cls_path', '')
        if scaler_cls_path:
            params_['scaler_cls'] = load_class(scaler_cls_path)
            cls._validate_params(params_['scaler_cls'].__init__, params_['scaler_params'])
        cls._validate_params(cls_._setup, params_)
        return cls_, params_

    def __post_init__(self):
        super().__post_init__()

        if type(self) is EvalTaskRunner:
            assert self.data_range_eval is not None

        self.criterion_cls = load_class(self.criterion['cls_path'])
        self.criterion_params = self.criterion['params']
        self._validate_params(self.criterion_cls._setup, self.criterion_params)

        self.eval_improvement = issubclass(self.criterion_cls, BaseImprovement)
        if self.eval_improvement:
            self.baseline_model_cls, self.baseline_model_params = \
                self.extract_model_config(self.baseline_model)

        self.model_cls, self.model_params = self.extract_model_config(self.model)

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
            batch_sampler_cls=load_class('nazuna.batch_samplers.BatchSampler'),
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
                loss_total += loss.get_sum()
                if output_loss_per_channel and loss.each_channel is not None:
                    batch_channel_sum = loss.each_channel.sum(dim=0)
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
                    loss_scaled_total += loss_s.get_sum()
                    if (output_loss_per_channel
                            and loss_s.each_channel is not None):
                        batch_channel_sum_s = \
                            loss_s.each_channel.sum(dim=0)
                        if loss_scaled_per_channel_total is None:
                            loss_scaled_per_channel_total = batch_channel_sum_s
                        else:
                            loss_scaled_per_channel_total += batch_channel_sum_s

                model_seq_len = getattr(self.model, 'seq_len', None)

                if self.dump_pred_data and not sample_saved:
                    save_data = {
                        'pred': pred[0].cpu().numpy(),
                        'data': batch.data[0].cpu().numpy(),
                        'data_future': batch.data_future[0].cpu().numpy(),
                        'sample_index': np.array(i_batch * self.batch_size_eval),
                        'timestamp': np.array(str(batch.tsta[0, -1])),
                    }
                    if baseline is not None:
                        save_data['baseline'] = baseline[0].cpu().numpy()
                    if model_seq_len is not None:
                        save_data['seq_len'] = np.array(model_seq_len)
                    np.savez(self.out_path / 'pred_first.npz', **save_data)
                    sample_saved = True

                if self.dump_pred_data:
                    last_in_batch = pred.shape[0] - 1
                    last_save_data = {
                        'pred': pred[-1].cpu().numpy(),
                        'data': batch.data[-1].cpu().numpy(),
                        'data_future': batch.data_future[-1].cpu().numpy(),
                        'sample_index': np.array(i_batch * self.batch_size_eval + last_in_batch),
                        'timestamp': np.array(str(batch.tsta[last_in_batch, -1])),
                    }
                    if baseline is not None:
                        last_save_data['baseline'] = baseline[-1].cpu().numpy()
                    if model_seq_len is not None:
                        last_save_data['seq_len'] = np.array(model_seq_len)

        if self.dump_pred_data:
            np.savez(self.out_path / 'pred_last.npz', **last_save_data)

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
        self.model = self.model_cls.create(
            self.device, self.model_state_path, **self.model_params,
        )
        loss_eval = self.eval()
        if self.inspector_params is not None:
            inspected = Inspector.inspect(
                model=self.model,
                criterion=self.criterion,
                batches=self.data_loader_eval,
                **self.inspector_params,
            )
            loss_eval.update(inspected)
        self.result['cols_org'] = dict(zip(self.dm.cols, self.dm.cols_org))
        self.result['data_range_eval'] = self.data_loader_eval.dataset.info
        self.result['parameters_trainable'] = self.model.count_trainable_parameters()
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
        criterion_target (dict = None): Criterion configuration for the training loss.
            Must have 'cls_path' (str) and 'params' (dict) keys.
            If not specified, the eval `criterion` is used for the training loss.
        batch_sampler (dict = None): Batch sampler configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        optimizer (dict = None): Optimizer configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        lr_scheduler (dict = None): Learning rate scheduler configuration. Optional.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        n_epoch (int = 0): Number of training epochs **(required)**.
        early_stop (bool = False): Whether to enable early stopping.
            Stops training if evaluation loss does not improve for 5 consecutive epochs.
    """
    data_range_train: tuple[int, int] = None
    data_offset_train: int = 0
    data_rolling_window_train: int = 4

    criterion_target: dict = None

    batch_sampler: dict = None
    optimizer: dict = None
    lr_scheduler: dict = None
    lr_scheduler_interval: str = 'epoch'  # 'epoch' or 'step'

    n_epoch: int = 0
    n_epoch_path: str | Path = None
    n_epoch_path_defer: bool = False
    early_stop: bool = False
    patience: int = 5

    raise_if_epoch_elapsed_over_min: int = -1

    save_model_state_every_epoch: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.data_range_train is not None

        self.criterion_target_cls = None
        if self.criterion_target is not None:
            self.criterion_target_cls = load_class(self.criterion_target['cls_path'])
            self.criterion_target_params = self.criterion_target['params']
            self._validate_params(self.criterion_target_cls._setup, self.criterion_target_params)

        self.batch_sampler_cls = load_class(self.batch_sampler['cls_path'])
        self.batch_sampler_params = self.batch_sampler['params']

        self.optimizer_cls = load_class(self.optimizer['cls_path'])
        self.optimizer_params = self.optimizer['params']
        self.lr_scheduler_cls = None
        if self.lr_scheduler:
            assert self.lr_scheduler_interval in {'epoch', 'step'}
            self.lr_scheduler_cls = load_class(self.lr_scheduler['cls_path'])
            self.lr_scheduler_params = self.lr_scheduler['params']

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

    def save_model(self, filename):
        torch.save(self.model.state_dict(), self.out_path / filename)

    def train(self, i_epoch=-1):
        data_loader = self.data_loader_train
        loss_total = 0.0
        self.model.train()
        self.model.on_epoch_start(i_epoch)
        for i_batch, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            loss = self.model.get_loss_and_backward(batch, self.criterion_target, i_epoch)
            loss_total += loss.get_sum()
            if self.save_model_state_every_epoch and (i_epoch == 0):
                self.save_model('model_state_ini.pth')
            self.optimizer.step()
            if self.lr_scheduler is not None and self.lr_scheduler_interval == 'step':
                self.lr_scheduler.step()

        if self.lr_scheduler is not None and self.lr_scheduler_interval == 'epoch':
            self.lr_scheduler.step()

        if self.save_model_state_every_epoch and (i_epoch > -1):
            self.save_model(f'model_state_{i_epoch}.pth')

        return {
            'n_sample': data_loader.dataset.n_sample,
            'loss_total': loss_total,
            'loss_per_sample': loss_total / data_loader.dataset.n_sample,
        }

    def _run(self):
        if self.n_epoch_path is not None:
            self.n_epoch = load_toml(self.n_epoch_path)['i_epoch_best'] + 1

        self.set_data_loader_train()
        if self.data_range_eval is not None:
            self.set_data_loader_eval()

        self.criterion = self.criterion_cls.create(self.device, **self.criterion_params)
        if self.criterion_target_cls is None:
            self.criterion_target = self.criterion
        else:
            self.criterion_target = \
                self.criterion_target_cls.create(self.device, **self.criterion_target_params)
        self.model = self.model_cls.create(self.device, **self.model_params)
        self.optimizer = \
            self.optimizer_cls(self.model.get_args_for_optimizer(self.optimizer_params))
        self.lr_scheduler = None
        if self.lr_scheduler_cls:
            self.lr_scheduler = self.lr_scheduler_cls(self.optimizer, **self.lr_scheduler_params)

        loss_per_sample_eval_best = float('inf')
        early_stop_counter = 0
        stop = False

        self.result['cols_org'] = dict(zip(self.dm.cols, self.dm.cols_org))
        self.result['data_range_train'] = self.data_loader_train.dataset.info
        self.result['n_sample_train'] = self.data_loader_train.dataset.n_sample
        if self.data_range_eval is not None:
            self.result['data_range_eval'] = self.data_loader_eval.dataset.info
            self.result['n_sample_eval'] = self.data_loader_eval.dataset.n_sample

        loss_history = []
        for i_epoch in range(self.n_epoch):
            self._log(f'Epoch {i_epoch} started')
            print(f'----- Epoch {i_epoch} -----')
            epoch_record = {'i_epoch': i_epoch}

            with measure_time(raise_if_elapsed_over_min=self.raise_if_epoch_elapsed_over_min):
                loss_train = self.train(i_epoch)
            epoch_record['train'] = loss_train

            if self.data_range_eval is None:
                loss_history.append(epoch_record)
                continue

            loss_eval = self.eval(output_loss_per_channel=False, output_scaled_loss=False)
            if self.inspector_params is not None:
                inspected = Inspector.inspect(
                    model=self.model,
                    criterion=self.criterion_target,
                    batches=self.data_loader_eval,
                    **self.inspector_params,
                )
                loss_eval.update(inspected)

            epoch_record['eval'] = loss_eval
            loss_history.append(epoch_record)
            loss_per_sample_eval = loss_eval['loss_per_sample']

            if loss_per_sample_eval < loss_per_sample_eval_best:
                loss_per_sample_eval_best = loss_per_sample_eval
                early_stop_counter = 0
                self.result['i_epoch_best'] = i_epoch
                self.result['loss_per_sample_eval_best'] = loss_per_sample_eval_best
                self.save_model('model_state.pth')
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
        search_space (dict = None): Search space **(required)**.
        data_ranges (list = None): List of data range dicts for cross-validation **(required)**.
            Each dict should have 'train' and 'eval' keys with (start, end) ratio tuples.
        n_trials (int = 10): Number of Optuna trials to run.
        direction (str = 'minimize'): Optimization direction ('minimize' or 'maximize').
        data_offset_train (int = 0): Offset for training data.
        data_rolling_window_train (int = 4): Rolling window size for training data.
        data_offset_eval (int = 0): Offset for evaluation data.
        data_rolling_window_eval (int = 4): Rolling window size for evaluation data.
        batch_size_eval (int = 32): Batch size for evaluation.
        criterion (dict = None): Criterion configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        model (dict = None): Model configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        batch_sampler (dict = None): Batch sampler configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        optimizer (dict = None): Optimizer configuration **(required)**.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        lr_scheduler (dict = None): Learning rate scheduler configuration. Optional.
            Must have 'cls_path' (str) and 'params' (dict) keys.
        n_epoch (int = 0): Number of training epochs per trial **(required)**.
        early_stop (bool = False): Whether to enable early stopping within each trial.
    """
    load_if_exists: bool = True
    search_space: dict = None
    direction: str = 'minimize'
    n_trials: int = -1
    n_trials_to_add: int = -1

    data_ranges_with_train_seeds: list = None
    data_offset_train: int = 0
    data_rolling_window_train: int = 4
    data_offset_eval: int = 0
    data_rolling_window_eval: int = 4

    batch_size_eval: int = 32
    criterion: dict = None
    model: dict = None

    batch_sampler: dict = None
    optimizer: dict = None
    lr_scheduler: dict = None
    lr_scheduler_interval: str = 'epoch'  # 'epoch' or 'step'

    n_epoch: int = 0
    early_stop: bool = False
    patience: int = 5

    raise_if_epoch_elapsed_over_min: int = -1
    raise_if_trial_elapsed_over_min: int = -1
    timeout_hour: float = -1

    search_targets: ClassVar[list[str]] = [
        'model_params', 'batch_sampler_params',
        'optimizer_params', 'lr_scheduler_params',
    ]

    def __post_init__(self):
        super().__post_init__()
        assert self.search_space, 'search_space is required'
        assert set(self.search_space) <= set(OptunaTaskRunner.search_targets) | {'seed'}
        assert 'lr_scheduler' not in self.search_space or self.lr_scheduler, \
            'lr_scheduler config is required when search_space contains "lr_scheduler"'
        for d in self.data_ranges_with_train_seeds:
            assert 'train' in d
            assert 'eval' in d
            assert set(d) <= {'train', 'eval', 'seed'}

        assert self.data_ranges_with_train_seeds is not None, \
            'data_ranges_with_train_seeds is required'
        assert len(self.data_ranges_with_train_seeds) > 0, \
            'data_ranges_with_train_seeds must not be empty'
        assert (self.n_trials >= 0) != (self.n_trials_to_add >= 0), \
            'Specify exactly one of n_trials or n_trials_to_add as a non-negative value'
        assert self.n_epoch > 0, 'n_epoch must be positive'

        self.criterion_cls = load_class(self.criterion['cls_path'])
        self.criterion_params = self.criterion['params']
        self.model_cls_path = self.model['cls_path']
        self.model_params = self.model['params']
        self.batch_sampler_cls_path = self.batch_sampler['cls_path']
        self.batch_sampler_params = self.batch_sampler['params']
        self.optimizer_cls_path = self.optimizer['cls_path']
        self.optimizer_params = self.optimizer['params']
        self.lr_scheduler_cls_path = None
        self.lr_scheduler_params = None
        if self.lr_scheduler:
            assert self.lr_scheduler_interval in {'epoch', 'step'}
            self.lr_scheduler_cls_path = self.lr_scheduler['cls_path']
            self.lr_scheduler_params = self.lr_scheduler['params']
        self._best_model_state = None
        self._best_trial_number = None

    @classmethod
    def _parse_choices(cls, spec):
        if spec['type'] == 'bool':
            return [False, True]
        choices = spec['choices'].split(',')
        if spec['type'] == 'int':
            choices = [int(c) for c in choices]
        if spec['type'] == 'float':
            choices = [float(c) for c in choices]
        return choices

    @classmethod
    def suggest_param(cls, trial, name, spec):
        method = spec['method']
        if method == 'log_uniform':
            low, high = float(spec['range'][0]), float(spec['range'][1])
            return trial.suggest_float(name, low, high, log=True)
        elif method == 'uniform':
            low, high = float(spec['range'][0]), float(spec['range'][1])
            return trial.suggest_float(name, low, high)
        elif method == 'int':
            low, high = int(spec['range'][0]), int(spec['range'][1])
            return trial.suggest_int(name, low, high)
        elif method in ['index', 'categorical']:
            choices = cls._parse_choices(spec)
            if method == 'categorical':
                return trial.suggest_categorical(name, choices)
            i = trial.suggest_int(f'{name}_index', 0, len(choices) - 1)
            return choices[i]
        else:
            raise ValueError(f'Unknown search space method: {method}')

    @classmethod
    def resolve_params(cls, params, search_space):
        resolved = dict(params)
        for target_specs in search_space.values():
            for name, spec in target_specs.items():
                if spec['method'] != 'index':
                    continue
                i = resolved.pop(f'{name}_index')
                resolved[name] = cls._parse_choices(spec)[i]
        return resolved

    def get_suggested_params(self, trial, target):
        base_params = getattr(self, target)
        params = copy.deepcopy(base_params) if base_params else {}
        for name, spec in self.search_space.get(target, {}).items():
            params[name] = OptunaTaskRunner.suggest_param(trial, name, spec)
        return params

    def _create_objective(self):
        def objective(trial):
            params_all = {}
            for target in OptunaTaskRunner.search_targets:
                params_all[target] = self.get_suggested_params(trial, target)
            seed = None
            for name, spec in self.search_space.get('seed', {}).items():
                seed = OptunaTaskRunner.suggest_param(trial, name, spec)
            try:
                with measure_time(raise_if_elapsed_over_min=self.raise_if_trial_elapsed_over_min):
                    value = self._run_trial(trial, seed=seed, **params_all)
                return value
            except Exception as e:
                print(f'[Optuna] Trial {trial.number} failed: {type(e).__name__}: {e}')
                self._n_failed += 1
                raise
        return objective

    def _run_trial(
        self, trial, model_params, batch_sampler_params, optimizer_params,
        lr_scheduler_params, seed=None,
    ):
        i_trial = trial.number
        self._log(f'Trial {i_trial} started')
        losses = []
        i_epoch_bests = []
        best_model_state_this_trial = None
        best_loss_this_trial = float('inf')
        for i_fold, d in enumerate(self.data_ranges_with_train_seeds):
            runner = TrainTaskRunner(
                seed=(d.get('seed', 0) if seed is None else seed),
                dm=self.dm, device=self.device, name=f'Trial {i_trial} Fold {i_fold}',
                out_dir=(self.out_path / f'trial_{i_trial}' / f'fold_{i_fold}'), exist_ok=True,
                data_range_train=d['train'], data_range_eval=d['eval'],
                data_offset_train=self.data_offset_train,
                data_rolling_window_train=self.data_rolling_window_train,
                data_offset_eval=self.data_offset_eval,
                data_rolling_window_eval=self.data_rolling_window_eval,
                batch_size_eval=self.batch_size_eval,
                criterion=self.criterion,
                model={'cls_path': self.model_cls_path, 'params': model_params},
                batch_sampler={'cls_path': self.batch_sampler_cls_path, 'params': batch_sampler_params},
                optimizer={'cls_path': self.optimizer_cls_path, 'params': optimizer_params},
                lr_scheduler=({
                    'cls_path': self.lr_scheduler_cls_path, 'params': lr_scheduler_params,
                } if self.lr_scheduler_cls_path else None),
                lr_scheduler_interval=self.lr_scheduler_interval,
                n_epoch=self.n_epoch, early_stop=self.early_stop, patience=self.patience,
                raise_if_epoch_elapsed_over_min=self.raise_if_epoch_elapsed_over_min,
            )
            runner.out_path.mkdir(parents=True, exist_ok=True)
            runner._run()
            fold_loss = runner.result.get('loss_per_sample_eval_best', float('inf'))
            losses.append(fold_loss)
            i_epoch_bests.append(runner.result.get('i_epoch_best', -1))
            if fold_loss < best_loss_this_trial:
                best_loss_this_trial = fold_loss
                best_model_state_this_trial = copy.deepcopy(runner.model.state_dict())
        trial.set_user_attr('fold_losses', losses)
        trial.set_user_attr('fold_i_epoch_bests', i_epoch_bests)
        mean_loss = sum(losses) / len(losses)
        if self._best_model_state is None or mean_loss < self.result.get('best_value', float('inf')):
            self._best_model_state = best_model_state_this_trial
            self._best_trial_number = i_trial
        return mean_loss

    @staticmethod
    def delete_trailing_incomplete_trials(study, storage):
        incomplete_trial_ids = []
        for trial in reversed(study.trials):
            if trial.state.name == 'COMPLETE':
                break
            incomplete_trial_ids.append(trial._trial_id)
        if len(incomplete_trial_ids) == 0:
            return
        print(f'Deleting incomplete trials: {incomplete_trial_ids}')
        engine = create_engine(storage)
        with engine.connect() as conn:
            for table_name in ['trial_params', 'trial_values', 'trial_user_attributes', 'trials']:
                stmt = f'DELETE FROM {table_name} WHERE trial_id IN :trial_ids'
                stmt = text(stmt).bindparams(bindparam('trial_ids', expanding=True))
                conn.execute(stmt, {'trial_ids': incomplete_trial_ids})
            conn.commit()

    def _create_study(self, storage):
        return optuna.create_study(
            direction=self.direction, sampler=optuna.samplers.TPESampler(seed=self.seed),
            storage=storage, study_name=self.name, load_if_exists=self.load_if_exists,
        )

    def _get_study(self, optuna_db_path):
        storage = f'sqlite:///{optuna_db_path.as_posix()}'
        if not self.load_if_exists:
            if self.name in optuna.get_all_study_names(storage=storage):
                optuna.delete_study(storage=storage, study_name=self.name)
            return self._create_study(storage)
        study = self._create_study(storage)
        OptunaTaskRunner.delete_trailing_incomplete_trials(study, storage)
        return self._create_study(storage)

    def _run(self):
        optuna_db_path = self.out_path / 'optuna.db'
        study = self._get_study(optuna_db_path)

        n_trials = self.n_trials_to_add
        if n_trials < 0:
            n_trials = self.n_trials - len(study.trials)
        if n_trials <= 0:
            print('The required number of trials has already been completed.')
            return

        self._n_failed = 0
        timeout = self.timeout_hour * 3600 if self.timeout_hour > 0 else None
        study.optimize(
            self._create_objective(), n_trials=n_trials, timeout=timeout,
            catch=(Exception,),
        )
        n_total = len(study.trials)
        n_completed = n_total - self._n_failed
        print(f'[Optuna] {n_total} trials: {n_completed} completed, {self._n_failed} failed')
        self.result['n_trials'] = n_total
        self.result['n_completed'] = n_completed
        self.result['n_failed'] = self._n_failed
        if n_completed > 0:
            self.result['best_trial_number'] = study.best_trial.number
            self.result['best_value'] = study.best_value
            self.result['best_params'] = study.best_params
            self.result['best_params_resolved'] = \
                self.resolve_params(study.best_params, self.search_space)
        trials_history = []
        for t in study.trials:
            record = {'number': t.number, 'state': t.state.name, 'params': t.params}
            if t.value is not None:
                record['value'] = t.value
            if 'fold_losses' in t.user_attrs:
                record['fold_losses'] = t.user_attrs['fold_losses']
                record['fold_i_epoch_bests'] = t.user_attrs['fold_i_epoch_bests']
            trials_history.append(record)
        self.result['trials'] = trials_history
        if self._best_model_state is not None:
            torch.save(self._best_model_state, self.out_path / 'model_state.pth')


class TaskType(Enum):
    eval = EvalTaskRunner
    train = TrainTaskRunner
    optuna = OptunaTaskRunner
    diag = DiagnosticsTaskRunner
