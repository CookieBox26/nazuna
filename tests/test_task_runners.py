from nazuna.task_runners import (
    EvalTaskRunner,
    TrainTaskRunner,
    OptunaTaskRunner,
)
import optuna
import pytest
from unittest.mock import patch


def test_eval_task_runner_run(tmp_path, get_data_manager):
    dm = get_data_manager()
    conf_task = {
        'data_range_eval': (0.8, 1.0),
        'criterion': {
            'cls_path': 'nazuna.criteria.MAE',
            'params': {'n_channel': 2, 'pred_len': 7},
        },
        'model': {
            'cls_path': 'nazuna.models.simple_average.SimpleAverage',
            'params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
        },
    }
    runner = EvalTaskRunner(dm=dm, name='Task 0', out_dir=tmp_path / 'task_0', **conf_task)
    runner.run()
    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()


def test_train_task_runner_run(tmp_path, get_data_manager):
    dm = get_data_manager()
    conf_task = {
        'data_range_eval': (0.6, 0.8),
        'criterion': {
            'cls_path': 'nazuna.criteria.MSE',
            'params': {'n_channel': 2, 'pred_len': 7},
        },
        'model': {
            'cls_path': 'nazuna.models.simple_average.SimpleAverageVariableDecay',
            'params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
        },
        'data_range_train': (0.0, 0.6),
        'batch_sampler': {
            'cls_path': 'nazuna.batch_samplers.BatchSamplerShuffle',
            'params': {'batch_size': 32},
        },
        'optimizer': {
            'cls_path': 'torch.optim.Adam',
            'params': {'lr': 0.01},
        },
        'lr_scheduler': {
            'cls_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
            'params': {'T_max': 10},
        },
        'n_epoch': 2,
    }
    runner = TrainTaskRunner(dm=dm, name='Task 0', out_dir=tmp_path / 'task_0', **conf_task)
    runner.run()
    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()


def test_train_task_runner_run_optimizer_groups(tmp_path, get_data_manager):
    dm = get_data_manager()
    conf_task = {
        'data_range_eval': (0.6, 0.8),
        'criterion': {
            'cls_path': 'nazuna.criteria.MSE',
            'params': {'n_channel': 2, 'pred_len': 7},
        },
        'model': {
            'cls_path': 'nazuna.models.simple_average.SimpleAverageVariableDecay',
            'params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
        },
        'data_range_train': (0.0, 0.6),
        'batch_sampler': {
            'cls_path': 'nazuna.batch_samplers.BatchSamplerShuffle',
            'params': {'batch_size': 32},
        },
        'optimizer_groups': {
            'model': {
                'optimizer': {'cls_path': 'torch.optim.Adam', 'params': {'lr': 0.01}},
                'lr_scheduler': {
                    'cls_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
                    'params': {'T_max': 10},
                },
                'lr_scheduler_interval': 'epoch',
            },
        },
        'n_epoch': 2,
    }
    runner = TrainTaskRunner(dm=dm, name='Task 0', out_dir=tmp_path / 'task_0', **conf_task)
    runner.run()
    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()


def test_optuna_task_runner_parse_choices():
    spec = {'type': 'int', 'choices': '15,25,35'}
    choices = OptunaTaskRunner._parse_choices(spec)
    assert choices == [15, 25, 35]
    spec = {'type': 'bool'}
    choices = OptunaTaskRunner._parse_choices(spec)
    assert choices == [False, True]


def get_optuna_task_runner(dm, out_dir, search_space=None):
    if search_space is None:
        search_space = {
            'optimizer_params': {'lr': {'method': 'log_uniform', 'range': [1e-4, 1e-1]}},
        }
    return OptunaTaskRunner(
        dm=dm, name='Optuna Task 0', out_dir=out_dir, n_trials=4,
        search_space=search_space,
        data_ranges_with_train_seeds=[
            {'train': (0.0, 0.6), 'eval': (0.6, 0.8)},
            {'train': (0.0, 0.8), 'eval': (0.8, 1.0)},
        ],
        criterion={
            'cls_path': 'nazuna.criteria.MSE',
            'params': {'n_channel': 2, 'pred_len': 7},
        },
        model={
            'cls_path': 'nazuna.models.simple_average.SimpleAverageVariableDecay',
            'params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
        },
        batch_sampler={
            'cls_path': 'nazuna.batch_samplers.BatchSamplerShuffle',
            'params': {'batch_size': 16},
        },
        optimizer={'cls_path': 'torch.optim.Adam', 'params': {'lr': 0.01}},
        n_epoch=1,
    )


def test_optuna_task_runner_get_suggested_params(tmp_path, get_data_manager):
    dm = get_data_manager()
    runner = get_optuna_task_runner(dm, tmp_path / 'task_0')
    trial = optuna.trial.FixedTrial({'optimizer_params__lr': 0.02})
    params_all = {}
    for target in OptunaTaskRunner.search_targets:
        params_all[target] = runner.get_suggested_params(trial, target)
    assert params_all['optimizer_params']['lr'] == pytest.approx(0.02)


def test_optuna_task_runner_run(tmp_path, get_data_manager):
    dm = get_data_manager()
    runner = get_optuna_task_runner(dm, tmp_path / 'task_0')
    runner.run()

    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()
    assert runner.result['n_trials'] == 4
    assert runner.result['n_completed'] == 4
    assert runner.result['n_failed'] == 0
    assert 'best_value' in runner.result
    assert 'best_params' in runner.result
    assert 'optimizer_params__lr' in runner.result['best_params']
    assert len(runner.result['trials']) == 4
    for t in runner.result['trials']:
        assert t['state'] == 'COMPLETE'


def test_optuna_task_runner_run_optimizer_groups(tmp_path, get_data_manager):
    dm = get_data_manager()
    runner = OptunaTaskRunner(
        dm=dm, name='Optuna Task 0', out_dir=tmp_path / 'task_0', n_trials=4,
        search_space_optimizer_groups={
            'model': {
                'optimizer_params': {
                    'lr': {'method': 'log_uniform', 'range': [1e-4, 1e-1]},
                },
            },
        },
        data_ranges_with_train_seeds=[
            {'train': (0.0, 0.6), 'eval': (0.6, 0.8)},
            {'train': (0.0, 0.8), 'eval': (0.8, 1.0)},
        ],
        criterion={
            'cls_path': 'nazuna.criteria.MSE',
            'params': {'n_channel': 2, 'pred_len': 7},
        },
        model={
            'cls_path': 'nazuna.models.simple_average.SimpleAverageVariableDecay',
            'params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
        },
        batch_sampler={
            'cls_path': 'nazuna.batch_samplers.BatchSamplerShuffle',
            'params': {'batch_size': 16},
        },
        optimizer_groups={
            'model': {
                'optimizer': {'cls_path': 'torch.optim.Adam', 'params': {'lr': 0.01}},
            },
        },
        n_epoch=1,
    )
    runner.run()

    assert runner.result_path.is_file()
    assert runner.result['n_trials'] == 4
    assert runner.result['n_completed'] == 4
    assert runner.result['n_failed'] == 0
    assert 'model__optimizer__lr' in runner.result['best_params']
    for t in runner.result['trials']:
        assert t['state'] == 'COMPLETE'


def test_optuna_task_runner_suggest_seed(tmp_path, get_data_manager):
    dm = get_data_manager()
    runner = get_optuna_task_runner(
        dm, tmp_path / 'task_0',
        search_space={
            'optimizer_params': {'lr': {'method': 'log_uniform', 'range': [1e-4, 1e-1]}},
            'seed': {'seed': {'method': 'categorical', 'type': 'int', 'choices': '7,8'}},
        },
    )
    seeds = []
    real_cls = TrainTaskRunner

    def recording_cls(*args, seed=0, **kwargs):
        seeds.append(seed)
        return real_cls(*args, seed=seed, **kwargs)

    with patch('nazuna.task_runners.TrainTaskRunner', side_effect=recording_cls):
        runner.run()

    # The suggested seed is used for every fold, overriding the per-fold default 0.
    assert seeds and set(seeds) <= {7, 8}
    assert runner.result['best_params']['seed'] in {7, 8}


def test_optuna_task_runner_run_with_failures(tmp_path, get_data_manager):
    dm = get_data_manager()
    runner = get_optuna_task_runner(dm, tmp_path / 'task_0')
    original_run_trial = runner._run_trial
    call_count = [0]

    def sometimes_failing_run_trial(trial, *args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            raise AssertionError('simulated constraint violation')
        return original_run_trial(trial, *args, **kwargs)

    with patch.object(runner, '_run_trial', side_effect=sometimes_failing_run_trial):
        runner.run()

    assert runner.result_path.is_file()
    assert runner.result['n_trials'] == 4
    assert runner.result['n_completed'] == 2
    assert runner.result['n_failed'] == 2
    assert 'best_value' in runner.result
    assert 'best_params' in runner.result
