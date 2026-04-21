from nazuna.task_runners import (
    Config,
    EvalTaskRunner,
    TrainTaskRunner,
    OptunaTaskRunner,
    run_tasks,
)


def test_eval_task_runner(tmp_path, get_data_manager):
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


def test_train_task_runner(tmp_path, get_data_manager):
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
        optimizer={
            'cls_path': 'torch.optim.Adam',
            'params': {'lr': 0.01},
        },
        n_epoch=2,
    )
    runner.run()

    assert runner.result_path.is_file()

    assert runner.out_path.is_dir()
    assert (runner.out_path / 'best_model_state.pth').is_file()
    assert (runner.out_path / 'study.pkl').is_file()
    assert runner.result['n_trials'] == 2
    assert runner.result['n_completed'] == 2
    assert runner.result['n_failed'] == 0
    assert 'best_value' in runner.result
    assert 'best_params' in runner.result
    assert 'lr' in runner.result['best_params']
    assert len(runner.result['trials']) == 2
    for t in runner.result['trials']:
        assert t['state'] == 'COMPLETE'
    summary_path = runner.out_path / 'trial_summary.txt'
    assert summary_path.is_file()
    assert '2 completed' in summary_path.read_text(encoding='utf8')


def test_optuna_task_runner_with_failures(
    tmp_path, get_data_manager,
):
    from unittest.mock import patch
    dm = get_data_manager()
    runner = OptunaTaskRunner(
        dm=dm,
        name='Optuna Fail Test',
        out_dir=tmp_path / 'task_fail',
        n_trials=4,
        search_space={'lr': ['log_uniform', 1e-4, 1e-1]},
        data_ranges=[
            {'train': (0.0, 0.6), 'eval': (0.6, 0.8)},
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
        optimizer={
            'cls_path': 'torch.optim.Adam',
            'params': {'lr': 0.01},
        },
        n_epoch=1,
    )

    original_run_trial = runner._run_trial
    call_count = [0]

    def sometimes_failing_run_trial(trial):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            raise AssertionError('simulated constraint violation')
        return original_run_trial(trial)

    with patch.object(
        runner, '_run_trial', side_effect=sometimes_failing_run_trial,
    ):
        runner.run()

    assert runner.result_path.is_file()
    assert runner.result['n_trials'] == 4
    assert runner.result['n_completed'] == 2
    assert runner.result['n_failed'] == 2
    assert 'best_value' in runner.result
    assert 'best_params' in runner.result

    summary_text = (
        runner.out_path / 'trial_summary.txt'
    ).read_text(encoding='utf8')
    assert '2 completed' in summary_text
    assert '2 failed' in summary_text


conf_toml_str_0 = '''
exist_ok = true

[data]
path = [ "jma", "weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv",]
colname_timestamp = "date"
seq_len = 28
pred_len = 7
white_list = [ "temp_avg_nagoya", "temp_avg_fukuoka",]

[[tasks]]
task_type = "eval"
name = "Eval"
data_range_eval = [ 0.8, 1.0,]
criterion = { cls_path = "nazuna.criteria.MSE", params = { n_channel = 2, pred_len = 7 } }
model = { cls_path = "nazuna.models.simple_average.SimpleAverage", params = { seq_len = 28, pred_len = 7, period_len = 7 } }

[[tasks]]
task_type = "train"
name = "Pilot"
data_range_train = [ 0.0, 0.6,]
data_range_eval = [ 0.6, 0.8,]
criterion = { cls_path = "nazuna.criteria.MSE", params = { n_channel = 2, pred_len = 7 } }
model = { cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecay", params = { seq_len = 28, pred_len = 7, period_len = 7 } }
batch_sampler = { cls_path = "nazuna.batch_samplers.BatchSamplerShuffle", params = { batch_size = 16 } }
optimizer = { cls_path = "torch.optim.Adam", params = { lr = 0.001 } }
lr_scheduler = { cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR", params = { T_max = 10 } }
n_epoch = 10
early_stop = true

[[tasks]]
task_type = "train"
name = "Train"
data_range_train = [ 0.0, 0.8,]
criterion = { cls_path = "nazuna.criteria.MSE", params = { n_channel = 2, pred_len = 7 } }
model = { cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecay", params = { seq_len = 28, pred_len = 7, period_len = 7 } }
batch_sampler = { cls_path = "nazuna.batch_samplers.BatchSamplerShuffle", params = { batch_size = 16 } }
optimizer = { cls_path = "torch.optim.Adam", params = { lr = 0.001 } }
lr_scheduler = { cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR", params = { T_max = 10 } }
n_epoch = { task_name = "Pilot" }
'''


def test_run_tasks_0(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    print(out_dir)
    run_tasks(f'out_dir = "{out_dir.as_posix()}"\n' + conf_toml_str_0)
    assert out_dir.is_dir()


def test_run_tasks_1(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    run_tasks(f'out_dir = "{out_dir.as_posix()}"\n' + conf_toml_str_0, skip_task_ids_='1-2')
    assert out_dir.is_dir()


conf_toml_str_definitions = '''
exist_ok = true

[data]
path = [ "jma", "weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv",]
colname_timestamp = "date"
seq_len = 28
pred_len = 7
white_list = [ "temp_avg_nagoya", "temp_avg_fukuoka",]

[definitions.MSE]
cls_path = "nazuna.criteria.MSE"
params = { n_channel = 2, pred_len = 7 }

[definitions.SimpleAverage]
cls_path = "nazuna.models.simple_average.SimpleAverage"
[definitions.SimpleAverage.params]
seq_len = 28
pred_len = 7
period_len = 7

[[tasks]]
task_type = "eval"
name = "Eval"
data_range_eval = [ 0.8, 1.0,]
criterion = "MSE"
model = "SimpleAverage"
'''


def test_config_definitions(tmp_path):
    out_dir = tmp_path / 'tasks_def'
    run_tasks(
        f'out_dir = "{out_dir.as_posix()}"\n'
        + conf_toml_str_definitions,
    )
    assert out_dir.is_dir()
    assert (out_dir / 'eval' / 'result.toml').is_file()


def test_config_definitions_resolve(tmp_path):
    conf = Config(
        out_dir=tmp_path / 'test_resolve',
        exist_ok=True,
        data={
            'path': [
                'jma',
                'weather_japan_daily_2025-01-01_2025-12-31'
                '_2_blocks.csv',
            ],
            'colname_timestamp': 'date',
            'seq_len': 28,
            'pred_len': 7,
            'white_list': [
                'temp_avg_nagoya', 'temp_avg_fukuoka',
            ],
        },
        definitions={
            'MAE': {
                'cls_path': 'nazuna.criteria.MAE',
                'params': {'n_channel': 2, 'pred_len': 7},
            },
            'SimpleAverage': {
                'cls_path': 'nazuna.models.simple_average.SimpleAverage',
                'params': {
                    'seq_len': 28,
                    'pred_len': 7,
                    'period_len': 7,
                },
            },
        },
        tasks=[
            {
                'task_type': 'eval',
                'data_range_eval': [0.8, 1.0],
                'criterion': 'MAE',
                'model': 'SimpleAverage',
            },
        ],
    )
    _, params = conf.get_task_runner(0)
    assert isinstance(params['criterion'], dict)
    assert params['criterion']['cls_path'] == 'nazuna.criteria.MAE'
    assert params['criterion']['params']['n_channel'] == 2
    assert isinstance(params['model'], dict)
    assert params['model']['cls_path'] == 'nazuna.models.simple_average.SimpleAverage'
    assert params['model']['params']['seq_len'] == 28
