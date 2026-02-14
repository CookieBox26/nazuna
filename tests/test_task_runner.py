from nazuna.task_runner import (
    EvalTaskRunner,
    TrainTaskRunner,
    OptunaTaskRunner,
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


def test_optuna_task_runner(tmp_path, get_data_manager):
    dm = get_data_manager()
    runner = OptunaTaskRunner(
        dm=dm,
        name='Optuna Task 0',
        out_dir=tmp_path / 'task_0',
        n_trials=2,
        search_space={'lr': ['log_uniform', 1e-4, 1e-1]},
        data_ranges=[
            {'train': (0.0, 0.6), 'eval': (0.6, 0.8)},
            {'train': (0.0, 0.8), 'eval': (0.8, 1.0)},
        ],
        criterion_cls_path='nazuna.criteria.MSE',
        criterion_params={'n_channel': 2, 'pred_len': 7},
        model_cls_path=(
            'nazuna.models.simple_average'
            '.SimpleAverageVariableDecay'
        ),
        model_params={
            'seq_len': 28, 'pred_len': 7, 'period_len': 7,
        },
        batch_sampler_cls_path=(
            'nazuna.batch_sampler.BatchSamplerShuffle'
        ),
        batch_sampler_params={'batch_size': 16},
        optimizer_cls_path='torch.optim.Adam',
        optimizer_params={'lr': 0.01},
        n_epoch=2,
    )
    runner.run()

    assert runner.result_path.is_file()
    assert runner.out_path.is_dir()
    assert (runner.out_path / 'best_model_state.pth').is_file()
    assert (runner.out_path / 'study.pkl').is_file()
    assert runner.result['n_trials'] == 2
    assert 'best_value' in runner.result
    assert 'best_params' in runner.result
    assert 'lr' in runner.result['best_params']
    assert len(runner.result['trials']) == 2
    for t in runner.result['trials']:
        assert t['state'] == 'COMPLETE'


conf_toml_str = '''
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


def test_run_tasks_0(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    run_tasks(f'out_dir = "{out_dir.as_posix()}"\n' + conf_toml_str)
    assert out_dir.is_dir()


def test_run_tasks_1(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    run_tasks(f'out_dir = "{out_dir.as_posix()}"\n' + conf_toml_str, skip_task_ids_='1')
    assert out_dir.is_dir()
