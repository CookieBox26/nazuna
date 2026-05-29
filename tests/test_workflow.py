from nazuna.workflow import (
    Workflow, WorkflowTemplateResolver, normalize_config, run,
)
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

[definitions.MAE]
cls_path = "nazuna.criteria.MAE"
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

[definitions.SimpleAverageVariableDecayChannelwise]
cls_path = "nazuna.models.simple_average.SimpleAverageVariableDecayChannelwise"
params = { seq_len = 28, pred_len = 7, period_len = 7, n_channel = 2 }

[definitions.BS]
cls_path = "nazuna.batch_samplers.BatchSamplerShuffle"
params = { batch_size = 32 }

[definitions.BS64]
cls_path = "nazuna.batch_samplers.BatchSamplerShuffle"
params = { batch_size = 64 }

[definitions.Adam]
cls_path = "torch.optim.Adam"
params = { lr = 0.0001 }

[definitions.CosineAnnealingLR]
cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR"
params = { T_max = 5 }

[definitions.ExponentialLR]
cls_path = "torch.optim.lr_scheduler.ExponentialLR"
params = { gamma = 0.95 }
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
batch_sampler = "BS"
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
batch_sampler = "BS"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = { task_name = "Pilot" }
'''


def test_run_tasks_0(tmp_path):
    out_dir = tmp_path / 'tasks_0'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + tasks_0
    run(conf)
    assert out_dir.is_dir()
    assert (out_dir / 'report.md').is_file()


template_train_with_baseline = '''
# =============== template ===============
[template]
template_type = "train_with_baseline"
criterion_target = "MSE"
criterion_eval = "MSE"
criteria_additional = [ "MAE",]
criterion_imprate = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BS"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
patience = 5
'''


@pytest.mark.slow
def test_run_template_train_with_baseline(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline
    wf = run(conf)
    assert out_dir.is_dir()
    assert (out_dir / 'report.md').is_file()
    assert wf.tasks[4]['name'] == 'Eval 0'


template_train_with_baseline_multiparams = '''
# =============== template ===============
[template]
template_type = "train_with_baseline_multiparams"
criterion_target = "MSE"
criterion_eval = "MSE"
criterion_imprate = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BS"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
patience = 5
[[template.params]]
batch_sampler = "BS"
lr_scheduler = "CosineAnnealingLR"
[[template.params]]
batch_sampler = "BS64"
lr_scheduler = "ExponentialLR"
'''


@pytest.mark.slow
def test_run_template_train_with_baseline_multiparams(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline_multiparams'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline_multiparams
    wf = run(conf)
    assert out_dir.is_dir()
    assert (out_dir / 'report.md').is_file()
    assert wf.tasks[1 + 0 * 4 + 1]['name'] == 'Train 0'
    assert wf.tasks[1 + 0 * 4 + 1]['batch_sampler'] == 'BS'
    assert wf.tasks[1 + 0 * 4 + 1]['lr_scheduler'] == 'CosineAnnealingLR'
    assert wf.tasks[1 + 1 * 4 + 1]['name'] == 'Train 1'
    assert wf.tasks[1 + 1 * 4 + 1]['lr_scheduler'] == 'ExponentialLR'


template_train_with_baseline_multiparams_pilot_reuse = '''
# =============== template ===============
[template]
template_type = "train_with_baseline_multiparams"
criterion_target = "MSE"
criterion_eval = "MSE"
criterion_imprate = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "SimpleAverageVariableDecay"
data_range_train = [ 0.0, 0.8,]
data_range_eval = [ 0.8, 1.0,]
data_range_train_pilot = [ 0.0, 0.6,]
data_range_eval_pilot = [ 0.6, 0.8,]
batch_sampler = "BS"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = 5
patience = 5
[[template.params]]
seed = 0
[[template.params]]
seed = 1
i_taskset_pilot = 0
[[template.params]]
seed = 2
[[template.params]]
seed = 3
i_taskset_pilot = 2
'''


def test_template_train_with_baseline_multiparams_pilot_reuse(tmp_path):
    out_dir = tmp_path / 'template_train_with_baseline_multiparams_pilot_reuse'
    conf = f'out_dir = "{out_dir.as_posix()}"\n' + premise + \
        template_train_with_baseline_multiparams_pilot_reuse
    d = normalize_config(conf)
    d = WorkflowTemplateResolver.resolve(d)
    wf = Workflow(**d)
    task_names = [t['name'] for t in wf.tasks]
    assert task_names == [
        'Eval Baseline',
        'Pilot 0', 'Train 0', 'Eval 0', 'Eval ImpRate 0',
        'Train 1', 'Eval 1', 'Eval ImpRate 1',
        'Pilot 2', 'Train 2', 'Eval 2', 'Eval ImpRate 2',
        'Train 3', 'Eval 3', 'Eval ImpRate 3',
    ]
    i_train_1 = task_names.index('Train 1')
    assert wf.tasks[i_train_1]['n_epoch']['task_name'] == 'Pilot 0'
    assert wf.tasks[i_train_1]['seed'] == 1
    i_train_3 = task_names.index('Train 3')
    assert wf.tasks[i_train_3]['n_epoch']['task_name'] == 'Pilot 2'
    assert wf.tasks[i_train_3]['seed'] == 3
    # The original Pilot of taskset 0 still references its own training data
    i_pilot_0 = task_names.index('Pilot 0')
    assert wf.tasks[i_pilot_0]['name'] == 'Pilot 0'


dummy_conf = '''
definition_includes = [
{ bundled = "common" },
{ relpath = "defs.toml" },
]

[definition_includes_data]
n_channel = 4
pred_len = 24
period_len = 24

[definitions]
NEpoch = 100
'''
dummy_defs = '''
[definitions]
NEpoch = 50

[definitions.BSS8]
cls_path = "nazuna.batch_samplers.BatchSamplerShuffle"
params = { batch_size = 8 }
'''


def test_normalize_config(tmp_path):
    conf_path = tmp_path / 'hoge.toml'
    defs_path = tmp_path / 'defs.toml'

    conf_path.write_text('out_dir = "__CONFIG_STEM__"', newline='\n', encoding='utf8')
    d = normalize_config(conf_path)
    assert d['out_dir'] == (tmp_path / 'hoge').as_posix()

    conf_path.write_text(dummy_conf, newline='\n', encoding='utf8')
    defs_path.write_text(dummy_defs, newline='\n', encoding='utf8')
    d = normalize_config(conf_path)
    assert d['definitions']['BSS8'] == {
        'cls_path': 'nazuna.batch_samplers.BatchSamplerShuffle',
        'params': {'batch_size': 8},
    }
    assert d['definitions']['BSS32'] == {
        'cls_path': 'nazuna.batch_samplers.BatchSamplerShuffle',
        'params': {'batch_size': 32},
    }
    assert d['definitions']['NEpoch'] == 100
    assert d['definitions']['MAE'] == {
        'cls_path': 'nazuna.criteria.MAE',
        'params': {'n_channel': 4, 'pred_len': 24},
    }
    assert d['definitions']['MSE'] == {
        'cls_path': 'nazuna.criteria.MSE',
        'params': {'n_channel': 4, 'pred_len': 24},
    }


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
