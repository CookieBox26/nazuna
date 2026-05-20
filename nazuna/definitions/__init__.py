from importlib.resources import files
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pathlib import Path
import toml


def _normalize_suffixes(filename):
    p = Path(filename)
    suffixes = p.suffixes
    if suffixes[-2:] == ['.toml', '.j2']:
        return str(p)
    if suffixes[-1:] == ['.toml']:
        return str(p.with_suffix('.toml.j2'))
    if not suffixes:
        return str(p.with_suffix('.toml.j2'))
    raise ValueError(f'Unsupported template path suffix: {filename}')


def get_definition(filename, data):
    assert data is not None, 'Template data for a bundled definition must not be None.'
    template_dir = files('nazuna.definitions')
    normalized = _normalize_suffixes(filename)
    assert (template_dir / normalized).is_file(), \
        f'Bundled definition file not found: {filename}'
    env = Environment(loader=FileSystemLoader(template_dir), undefined=StrictUndefined)
    template = env.get_template(normalized)
    data['period_len_2'] = 2 * data['period_len']
    data['period_len_4'] = 4 * data['period_len']
    return toml.loads(template.render(data))
