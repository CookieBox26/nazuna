In Nazuna, you can describe the configuration for a workflow in TOML and run it.

```bash
uv run nazuna config.toml
uv run nazuna config.toml --skip_task_ids 0,1  # if you want to skip tasks 0 and 1
```

!!! info

    If you do not install via `pyproject.toml`, use `python -m nazuna` instead of `nazuna`.

The first argument is the path to the config file.
The following options are also available:

- `-s, --skip_task_ids` &mdash; Skip tasks by index (0-based), e.g., `0,1` or `1-3`.
  Dependencies are not validated, so ensure that no subsequent tasks depend on the skipped ones.
  This is useful for resuming or fixing part of a workflow.
- `-t, --target_tasks` &mdash; Run only the specified tasks by name, e.g., `Train,Eval`.
  Cannot be combined with `--skip_task_ids`.
- `-f, --force_rerun` &mdash; Re-run a task even when its result file already exists.
  By default, tasks whose results already exist are skipped.
- `--suppress_plot` &mdash; Do not generate plots.
- `--force_replot` &mdash; Force regeneration of data-based plots even when they already exist.
  This flag only takes effect when all tasks are skipped (via `--skip_task_ids` or `--report_only`),
  and is intended for cases such as updating the plotting function.
- `--report_only` &mdash; Do not run any task; only generate the report
  (equivalent to skipping all tasks via `--skip_task_ids`).
- `--example` &mdash; Run a bundled example config (including data).
  Specify a TOML file under [nazuna/examples/](https://github.com/CookieBox26/nazuna/tree/main/nazuna/examples), e.g., `jma_train_dlinear`
  (extension optional). No config path is required in this case.


## About Config File

Specify the arguments for the [`nazuna.workflow.Workflow`](reference_workflow.md) class and the arguments for each [task runner class](reference_task_runners.md).

### Examples of Config Files

#### Example 1 (Defining tasks explicitly)

The following is an example workflow that trains a model and measures its improvement rate over a baseline.

- Setting `out_dir` to the special keyword `"__CONFIG_STEM__"` uses the config file path (without extension) as the output directory.
- The `definitions` key lets you define named settings shared across tasks. Reference a definition by its key name in each task. You can also specify values directly in each task without using `definitions`.


```toml
out_dir = "__CONFIG_STEM__"
exist_ok = true

# =============== data ===============
[data]
path = "~/datasets/traffic/traffic.csv"
colname_timestamp = "date"
seq_len = 672
pred_len = 168
white_list = "0,1,2,3,4,5,6,7"

# =============== definitions ===============
[definitions]
DataRangeEval = [ 0.8, 1.0,]
DataRangeTrain = [ 0.0, 0.8,]
DataRangeEvalPilot = [ 0.6, 0.8,]
DataRangeTrainPilot = [ 0.0, 0.6,]
NEpoch = 30

[definitions.MAE]
cls_path = "nazuna.criteria.MAE"
params = { n_channel = 8, pred_len = 168 }

[definitions.ImprovementRate]
cls_path = "nazuna.criteria.ImprovementRate"
params = { n_channel = 8, pred_len = 168, error_type = "mae" }

[definitions.SimpleAverage]
cls_path = "nazuna.models.simple_average.SimpleAverage"
params = { seq_len = 672, pred_len = 168, period_len = 168 }

[definitions.DLinear]
cls_path = "nazuna.models.dlinear.DLinear"
[definitions.DLinear.params]
seq_len = 672
pred_len = 168
kernel_size = 25

[definitions.BatchSamplerShuffle]
cls_path = "nazuna.batch_samplers.BatchSamplerShuffle"
params = { batch_size = 32 }

[definitions.Adam]
cls_path = "torch.optim.Adam"
params = { lr = 0.001 }

[definitions.CosineAnnealingLR]
cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR"
params = { T_max = 10 }

# =============== tasks ===============
[[tasks]]
task_type = "eval"
name = "Eval Baseline"
data_range_eval = "DataRangeEval"
criterion = "MAE"
model = "SimpleAverage"

[[tasks]]
task_type = "train"
name = "Pilot"
data_range_train = "DataRangeTrainPilot"
data_range_eval = "DataRangeEvalPilot"
criterion = "MAE"
model = "DLinear"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = "NEpoch"
early_stop = true

[[tasks]]
task_type = "train"
name = "Train"
data_range_train = "DataRangeTrain"
criterion = "MAE"
model = "DLinear"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = { task_name = "Pilot" }

[[tasks]]
task_type = "eval"
name = "Eval"
data_range_eval = "DataRangeEval"
criterion = "MAE"
model = "DLinear"
model_state = { task_name = "Train" }

[[tasks]]
task_type = "eval"
name = "Eval ImpRate"
data_range_eval = "DataRangeEval"
criterion = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "DLinear"
model_state = { task_name = "Train" }
```

#### Example 2 (Using a template)

Example 1 is a common scenario and is available as a template. Replace the `tasks` key with the following to run the same set of tasks.

```toml
# =============== template ===============
[template]
template_type = "train_with_baseline"
criterion = "MAE"
criterion_imprate = "ImprovementRate"
baseline_model = "SimpleAverage"
model = "DLinear"
data_range_eval = "DataRangeEval"
data_range_train = "DataRangeTrain"
data_range_eval_pilot = "DataRangeEvalPilot"
data_range_train_pilot = "DataRangeTrainPilot"
batch_sampler = "BatchSamplerShuffle"
optimizer = "Adam"
lr_scheduler = "CosineAnnealingLR"
n_epoch = "NEpoch"
```

Additional templates are also available for running the above scenario with multiple random seeds or multiple model configurations (documentation in progress).


### Examples of Learning Rate Scheduler Definitions

```toml
[definitions.CosineAnnealingLR]
cls_path = "torch.optim.lr_scheduler.CosineAnnealingLR"
params = { T_max = 20 }
```