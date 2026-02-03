import pytest
from nazuna.task_runner import (
    EvalTaskRunner,
    TrainTaskRunner,
    OptunaTaskRunner,
    TaskType,
    run_tasks,
)


def test_eval_task_runner(tmp_path, get_data_manager):
    dm = get_data_manager()
    conf_task = {
        'data_range_eval': (0.8, 1.0),
        'criterion_cls_path': 'nazuna.criteria.MAE',
        'criterion_params': {'n_channel': 2, 'pred_len': 7},
        'model_cls_path': 'nazuna.models.simple_average.SimpleAverage',
        'model_params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
    }
    runner = EvalTaskRunner(dm=dm, name='Task 0', out_dir=tmp_path / 'task_0', **conf_task)
    runner.run()
    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()


def test_train_task_runner(tmp_path, get_data_manager):
    dm = get_data_manager()
    conf_task = {
        'data_range_eval': (0.6, 0.8),
        'criterion_cls_path': 'nazuna.criteria.MSE',
        'criterion_params': {'n_channel': 2, 'pred_len': 7},
        'model_cls_path': 'nazuna.models.simple_average.SimpleAverageVariableDecay',
        'model_params': {'seq_len': 28, 'pred_len': 7, 'period_len': 7},
        'data_range_train': (0.0, 0.6),
        'batch_sampler_cls_path': 'nazuna.batch_sampler.BatchSamplerShuffle',
        'batch_sampler_params': {'batch_size': 32},
        'optimizer_cls_path': 'torch.optim.Adam',
        'optimizer_params': {'lr': 0.01},
        'lr_scheduler_cls_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
        'lr_scheduler_params': {'T_max': 10},
        'n_epoch': 2,
    }
    runner = TrainTaskRunner(dm=dm, name='Task 0', out_dir=tmp_path / 'task_0', **conf_task)
    runner.run()
    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()


conf_toml_str = '''
out_dir = "out/20260000-000000"
exist_ok = true

[data]
path = [ "jma", "weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv",]
colname_timestamp = "date"
seq_len = 28
pred_len = 7
white_list = [ "temp_avg_nagoya", "temp_avg_fukuoka",]

[[tasks]]
task_type = "eval"
data_range_eval = [ 0.8, 1.0,]
criterion_cls_path = "nazuna.criteria.MSE"
criterion_params = { n_channel = 2, pred_len = 7 }
model_cls_path = "nazuna.models.simple_average.SimpleAverage"
model_params = { seq_len = 28, pred_len = 7, period_len = 7 }

[[tasks]]
task_type = "train"
data_range_eval = [ 0.8, 1.0,]
criterion_cls_path = "nazuna.criteria.MSE"
criterion_params = { n_channel = 2, pred_len = 7 }
model_cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecay"
model_params = { seq_len = 28, pred_len = 7, period_len = 7 }
data_range_train = [ 0.0, 0.8,]
batch_sampler_cls_path = "nazuna.batch_sampler.BatchSamplerShuffle"
batch_sampler_params = { batch_size = 16 }
optimizer_cls_path = "torch.optim.Adam"
optimizer_params = { lr = 0.05 }
lr_scheduler_cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR"
lr_scheduler_params = { T_max = 3 }
n_epoch = 3
'''


def test_run_tasks():
    run_tasks(conf_toml_str)


def test_merge_params():
    base_params = {'seq_len': 96, 'pred_len': 24, 'lr': 0.01}
    suggested = {'lr': 0.001, 'batch_size': 32}
    search_space = {'lr': ['log_uniform', 1e-5, 1e-2]}
    result = OptunaTaskRunner._merge_params(base_params, suggested, search_space)
    assert result['seq_len'] == 96
    assert result['pred_len'] == 24
    assert result['lr'] == 0.001


def test_merge_params_with_none_base():
    suggested = {'lr': 0.001}
    search_space = {'lr': ['log_uniform', 1e-5, 1e-2]}
    result = OptunaTaskRunner._merge_params(None, suggested, search_space)
    assert result['lr'] == 0.001


def test_merge_params_does_not_mutate_base():
    base_params = {'lr': 0.01}
    suggested = {'lr': 0.001}
    search_space = {'lr': ['log_uniform', 1e-5, 1e-2]}
    OptunaTaskRunner._merge_params(base_params, suggested, search_space)
    assert base_params['lr'] == 0.01


def test_optuna_in_task_type():
    assert TaskType.optuna.value is OptunaTaskRunner


def test_optuna_task_runner_requires_search_space(get_data_manager):
    dm = get_data_manager()
    with pytest.raises(AssertionError, match='search_space is required'):
        OptunaTaskRunner(
            dm=dm,
            search_space=None,
            data_ranges=[{'train': [0, 0.8], 'eval': [0.8, 1.0]}],
            n_epoch=1,
            model_cls_path='nazuna.models.simple_average.SimpleAverage',
            batch_sampler_cls_path='nazuna.batch_sampler.BatchSamplerShuffle',
            optimizer_cls_path='torch.optim.Adam',
        )


def test_optuna_task_runner_requires_data_ranges(get_data_manager):
    dm = get_data_manager()
    with pytest.raises(AssertionError, match='data_ranges is required'):
        OptunaTaskRunner(
            dm=dm,
            search_space={'lr': ['log_uniform', 1e-5, 1e-2]},
            data_ranges=None,
            n_epoch=1,
            model_cls_path='nazuna.models.simple_average.SimpleAverage',
            batch_sampler_cls_path='nazuna.batch_sampler.BatchSamplerShuffle',
            optimizer_cls_path='torch.optim.Adam',
        )


def test_optuna_task_runner_requires_non_empty_data_ranges(get_data_manager):
    dm = get_data_manager()
    with pytest.raises(AssertionError, match='data_ranges must not be empty'):
        OptunaTaskRunner(
            dm=dm,
            search_space={'lr': ['log_uniform', 1e-5, 1e-2]},
            data_ranges=[],
            n_epoch=1,
            model_cls_path='nazuna.models.simple_average.SimpleAverage',
            batch_sampler_cls_path='nazuna.batch_sampler.BatchSamplerShuffle',
            optimizer_cls_path='torch.optim.Adam',
        )


def test_optuna_task_runner_requires_positive_n_epoch(get_data_manager):
    dm = get_data_manager()
    with pytest.raises(AssertionError, match='n_epoch must be positive'):
        OptunaTaskRunner(
            dm=dm,
            search_space={'lr': ['log_uniform', 1e-5, 1e-2]},
            data_ranges=[{'train': [0, 0.8], 'eval': [0.8, 1.0]}],
            n_epoch=0,
            model_cls_path='nazuna.models.simple_average.SimpleAverage',
            batch_sampler_cls_path='nazuna.batch_sampler.BatchSamplerShuffle',
            optimizer_cls_path='torch.optim.Adam',
        )
