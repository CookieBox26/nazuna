from nazuna.workflow import Workflow, load_config_from_path, run
import toml
import pytest


premise = '''
exist_ok = true

# =============== data ===============
[data]
path = [ "jma", "weather_japan_daily_2025-01-01_2025-12-31_2_blocks.csv",]
colname_timestamp = "date"
seq_len = 28
pred_len = 7
white_list = [ "temp_avg_nagoya", "temp_avg_fukuoka",]

# =============== definitions ===============
[definitions.MSE]
cls_path = "nazuna.criteria.MSE"
params = { n_channel = 2, pred_len = 7 }

[definitions.ImprovementRate]
cls_path = "nazuna.criteria.ImprovementRate"
params = { n_channel = 2, pred_len = 7, error_type = "mse" }

[definitions.SimpleAverage]
cls_path = "nazuna.models.simple_average.SimpleAverage"
params = { seq_len = 28, pred_len = 7, period_len = 7 }

[definitions.SimpleAverageVariableDecay]
cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecay"
params = { seq_len = 28, pred_len = 7, period_len = 7 }

[definitions.BatchSamplerShuffle]
cls_path = "nazuna.batch_samplers.BatchSamplerShuffle"
params = { batch_size = 32 }

[definitions.Adam]
cls_path = "torch.optim.Adam"
params = { lr = 0.001 }

[definitions.CosineAnnealingLR]
cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR"
params = { T_max = 5 }
'''


tasks_0 = '''
# =============== tasks ===============
[[tasks]]
task_type = "eval"
name = "Eval Baseline"
data_range_eval = [ 0.8, 1.0,]
criterion = "MSE"
model = "SimpleAverage"

[[tasks]]
task_type = "train"
name = "Pilot"
data_range_train = [ 0.0, 0.6,]
data_range_eval = [ 0.6, 0.8,]
criterion = "MSE"
model = "SimpleAverageVariableDecay"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
early_stop = true

[[tasks]]
task_type = "train"
name = "Train"
data_range_train = [ 0.0, 0.8,]
criterion = "MSE"
model = "SimpleAverageVariableDecay"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = { task_name = "Pilot" }
'''


template_train_with_baseline = '''
# =============== template ===============
[template]
template_type = "train_with_baseline"
criterion = "MSE"
criterion_imprate = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
patience = 5
'''


template_train_with_baseline_multiseeds = '''
# =============== template ===============
[template]
template_type = "train_with_baseline_multiseeds"
criterion = "MSE"
criterion_imprate = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
patience = 5
seeds = [ 0, 1, 2,]
'''


def validate_outputs(out_dir):
    assert out_dir.is_dir()
    assert (out_dir / 'config.toml').is_file()
    assert (out_dir / 'report.md').is_file()
    return toml.loads((out_dir / 'config.toml').read_text(encoding='utf8'))


def test_run_tasks_0(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + tasks_0
    run(conf)
    validate_outputs(out_dir)


@pytest.mark.slow
def test_run_template_train_with_baseline(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline
    run(conf)
    validate_outputs(out_dir)


@pytest.mark.slow
def test_run_template_train_with_baseline_multiseeds(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline_multiseeds'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline_multiseeds
    run(conf)
    conf = validate_outputs(out_dir)
    for task in conf['tasks']:
        if task['name'] == 'Train 1':
            assert task['n_epoch']['task_name'] == 'Pilot 1'
        if task['name'] == 'Train 2':
            assert task['n_epoch']['task_name'] == 'Pilot 2'


def test_load_config_from_path(tmp_path):
    conf_path = tmp_path / 'hoge.toml'
    conf_path.write_text('out_dir = "__CONFIG_STEM__"', newline='\n', encoding='utf8')
    d = load_config_from_path(conf_path)
    assert d['out_dir'] == (tmp_path / 'hoge').as_posix()


def test_workflow_get_task_runner():
    d = {
        'data': {},
        'definitions': {
            'DataRange': [0.8, 1.0],
            'MAE': {'cls_path': 'nazuna.criteria.MAE', 'params': {'n_channel': 4, 'pred_len': 24}},
            'SimpleAverage': {
                'cls_path': 'nazuna.models.simple_average.SimpleAverage',
                'params': {'seq_len': 97, 'pred_len': 24, 'period_len': 24},
            },
            'SimpleAverage_10': {'base': 'SimpleAverage', 'params': {'decay_rate': 1.0}},
            'SimpleAverage_07': {'base': 'SimpleAverage', 'params': {'decay_rate': 0.7}},
        },
        'tasks': [
            {'task_type': 'eval', 'data_range_eval': 'DataRange', 'model': 'SimpleAverage_10'},
            {'task_type': 'eval', 'data_range_eval': 'DataRange', 'model': 'SimpleAverage_07'},
        ],
    }
    wf = Workflow(**d)
    _, params = wf.parse_task_runner_config(0)
    assert params['model']['cls_path'] == 'nazuna.models.simple_average.SimpleAverage'
    assert set(params['model']['params'].keys()) == {'seq_len', 'pred_len', 'period_len', 'decay_rate'}
    assert params['model']['params']['decay_rate'] == pytest.approx(1.0)
    _, params = wf.parse_task_runner_config(1)
    assert params['model']['cls_path'] == 'nazuna.models.simple_average.SimpleAverage'
    assert set(params['model']['params'].keys()) == {'seq_len', 'pred_len', 'period_len', 'decay_rate'}
    assert params['model']['params']['decay_rate'] == pytest.approx(0.7)
