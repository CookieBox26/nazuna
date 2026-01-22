from importlib.resources import files
from nazuna.task_runner import run_tasks


def _get_conf_toml_path(identifier):
    if not identifier.endswith('.toml'):
        identifier = f'{identifier}.toml'
    conf_toml_path = files('nazuna.examples').joinpath(identifier)
    if not conf_toml_path.is_file():
        raise ValueError('Not exist:', conf_toml_path)
    return conf_toml_path


def run_example(identifier):
    run_tasks(_get_conf_toml_path(identifier))
