from nazuna.workflow import Workflow, run_tasks


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
    conf = Workflow(
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
