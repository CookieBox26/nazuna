In Nazuna, you can describe the configuration for a workflow in TOML and run it.

```bash
uv run nazuna config.toml
uv run nazuna config.toml --skip_task_ids 0,1  # if you want to skip tasks 0 and 1
```

!!! info

    If you do not install via `pyproject.toml`, use `python -m nazuna` instead of `nazuna`.

The first argument is the path to the config file.
The following options are also available:

- `--skip_task_ids` – Skip tasks by index (0-based), e.g., `0,1` or `1-3`.
  Dependencies are not validated, so ensure that no subsequent tasks depend on the skipped ones.
  This is useful for resuming or fixing part of a workflow.
- `--example` – Run a bundled example config (including data).
  Specify a TOML file under [nazuna/examples/](https://github.com/CookieBox26/nazuna/tree/main/nazuna/examples), e.g., `jma_train_dlinear`
  (extension optional). No config path is required in this case.


## About Config File

Specify the arguments for the [`nazuna.workflow.Workflow`](#nazuna.workflow.Workflow) class and the arguments for each task runner class.
For details on task runner classes, see [Reference (Task Runners)](reference_task_runners.md).

::: nazuna.workflow.Workflow
