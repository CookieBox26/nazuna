import copy
import dataclasses
from typing import ClassVar
from enum import Enum
from pathlib import Path
import toml
import torch
from nazuna.datasets import get_dataset_path
from nazuna.definitions import get_definitions
from nazuna.data_manager import TimeSeriesDataManager
from nazuna.task_runners import TaskType
from nazuna.report import report
from nazuna.utils import as_path_if_length_safe, measure_time, get_timestamp, load_toml


@dataclasses.dataclass
class Workflow:
    """
    Class that holds the configuration for a workflow.

    Attributes:
        out_dir (str | Path = ''): Output path for for the workflow.
            Outputs config.toml and report.md here.
            Defaults to 'out/YYYYmmdd-HHMMSS/' if not specified.

            - If individual task output paths are not specified, subdirectories are created
              under this path using task names.
            - You may also create this directory in advance and place a config.toml inside it
              (it will be overwritten with the resolved config.toml).
              In that case, set exist_ok to True.

        exist_ok (bool = True): Whether to allow the output path to already exist.
        data (dict = None): Data configuration for
            [TimeSeriesDataManager](reference_others.md#nazuna.data_manager.TimeSeriesDataManager)
            **(required)**.
        device (str = ''): Device name for computation (Ex. 'cpu', 'cuda').
            If not specified, it will be automatically detected from your environment.
        tasks (list[dict] = None): List of individual task configurations **(required)**.
            Each dict should have a 'task_type' key with a task type identifier
            (eval, train, optuna, diag), plus the required settings for that task type.
            See [Reference (Task Runners)](reference_task_runners.md) for details.

    !!! warning "About individual task names when running a workflow"

        Task names are used in the following cases:

        - If an individual task's output path is not specified, a subdirectory is created.
          The subdirectory name is the task name with symbols converted to snake_case.
        - You can specify model_state.pth trained in previous tasks by task name.

        Therefore, the following processing is done when creating a Workflow:

        - If a task name is not specified, it defaults to 'Task i' (0-indexed sequential number).
        - Duplicate task names are not allowed and will raise an error.
    """

    out_dir: str | Path = ''
    comment: str = ''
    exist_ok: bool = True
    device: str = ''
    definitions: dict = None
    data: dict = None
    tasks: list[dict] = None

    # If any of the following task keys is specified as a string,
    # resolve it using the definitions
    task_keys_accepting_definitions: ClassVar[list[str]] = [
        # a dict with cls_path and params keys
        'criterion', 'criterion_target', 'baseline_model', 'model',
        'batch_sampler', 'optimizer', 'lr_scheduler',
        # a list
        'batch_size_eval', 'data_range_train', 'data_range_eval', 'data_ranges',
        # an integer
        'n_epoch',
    ]

    @classmethod
    def _to_snake(cls, s):
        s = s.translate(str.maketrans('()=', '___'))
        return '_'.join(s.lower().split())

    def __post_init__(self):
        assert self.data is not None
        assert self.tasks is not None

        self.out_dir = self.out_dir or f'out/{get_timestamp()}/'
        self.out_path = Path(self.out_dir).expanduser()
        self.device = self.device or \
            str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        assert self.exist_ok or (not self.out_path.exists()), \
            f'Already exists: {self.out_path.as_posix()}'

        self.out_paths = {}
        self.task_names = []  # To use when analyzing results
        for i_task, _ in enumerate(self.tasks):
            self.tasks[i_task].setdefault('name', f'Task {i_task}')
            name = self.tasks[i_task]['name']
            if name in self.out_paths:
                raise ValueError(f'Duplicate task name: {self.tasks[i_task]["name"]}')
            self.task_names.append(name)
            out_dir_default = (self.out_path / type(self)._to_snake(name)).as_posix()
            self.tasks[i_task].setdefault('out_dir', out_dir_default)
            task_out_path = Path(self.tasks[i_task]['out_dir'])
            assert task_out_path.parent == self.out_path, \
                f'Task path must be under the workflow path.\n{self.out_path}\n{task_out_path}'
            self.out_paths[name] = task_out_path

        # Dry-run definition resolution
        self.used_definitions = set()
        if isinstance(self.data, str):
            _ = self.get_definition(self.data)
        for i_task, _ in enumerate(self.tasks):
            params = copy.deepcopy(self.tasks[i_task])
            for target in type(self).task_keys_accepting_definitions:
                if target in params and isinstance(params[target], str):
                    _ = self.get_definition(params[target])
            if 'optimizer_groups' in params:
                self.resolve_optimizer_groups_definitions(params['optimizer_groups'])

    def get_data_param(self):
        if isinstance(self.data, str):
            param = self.get_definition(self.data)
        else:
            param = copy.deepcopy(self.data)
        if isinstance(param['path'], dict):
            param['path'] = get_dataset_path(**param['path'])
        elif isinstance(param['path'], (list, tuple)):  # legacy compatibility
            param['path'] = get_dataset_path(*param['path'])
        return param

    def resolve_optimizer_groups_definitions(self, optimizer_groups):
        for group in optimizer_groups.values():
            for key in ['optimizer', 'lr_scheduler']:
                if key in group and isinstance(group[key], str):
                    group[key] = self.get_definition(group[key])

    def get_definition(self, name, _depth=0):
        assert _depth <= 5, \
            f'Inheritance depth exceeded (>5) at "{name}"'
        assert name in self.definitions, f'There is no definition named {name}'
        definition_raw = self.definitions[name]
        self.used_definitions.add(name)
        if not isinstance(definition_raw, dict):
            return copy.deepcopy(definition_raw)
        definition = {}
        if 'base' in definition_raw:
            # If a definition has a base key, first copy the (recursively resolved) base
            base_name = definition_raw['base']
            definition_base = self.get_definition(base_name, _depth=_depth + 1)
            for k, v in definition_base.items():
                 definition[k] = copy.deepcopy(v)
        for k, v in definition_raw.items():
            if k == 'base':
                continue
            if (k not in definition) or (not isinstance(v, dict)):
                # If there is no base, or the value is not a dictionary, deep-copy it
                definition[k] = copy.deepcopy(v)
            else:
                # If the value is a dict and a base definition exists, update it
                assert isinstance(definition[k], dict)
                definition[k].update(v)
        return definition

    def parse_task_runner_config(self, i_task):
        params = copy.deepcopy(self.tasks[i_task])
        self.task_type = params.pop('task_type')
        task_runner_cls = TaskType[self.task_type].value
        params.setdefault('device', self.device)
        params.setdefault('exist_ok', self.exist_ok)

        for target in type(self).task_keys_accepting_definitions:
            if target in params and isinstance(params[target], str):
                params[target] = self.get_definition(params[target])
        if 'optimizer_groups' in params:
            self.resolve_optimizer_groups_definitions(params['optimizer_groups'])

        if 'n_epoch' in params and isinstance(params['n_epoch'], dict):
            target_path = self.out_paths[params['n_epoch']['task_name']]
            params['n_epoch_path'] = task_runner_cls.to_result_path(target_path)
            params['n_epoch_path_defer'] = True
            del params['n_epoch']

        if 'model_state' in params:
            target_path = self.out_paths[params['model_state']['task_name']]
            params['model_state_path'] = target_path / 'model_state.pth'
            del params['model_state']

        return task_runner_cls, params

    def create_task_runners(self, dm):
        task_runners = []
        for i_task, _ in enumerate(self.tasks):
            cls_, params_ = self.parse_task_runner_config(i_task)
            task_runners.append(cls_(dm=dm, **params_))
        return task_runners

    def to_toml_str(self):
        assert [field.name for field in dataclasses.fields(self)] == \
            ['out_dir', 'comment', 'exist_ok', 'device', 'definitions', 'data', 'tasks'], \
            'Update the custom TOML stringification when fields are changed.'
        header = {
            'out_dir': self.out_dir, 'comment': self.comment,
            # 'exist_ok': self.exist_ok,
            # 'device': self.device,
        }
        toml_str = toml.dumps(header) + '\n'
        toml_str += '# =============== data ===============\n'
        toml_str += toml.dumps({'data': self.data}) + '\n'
        if self.definitions:
            toml_str += '# =============== definitions ===============\n'
            toml_str += '[definitions]\n'
            for k, v in self.definitions.items():
                if k not in self.used_definitions:
                    continue
                if not isinstance(v, dict):
                    toml_str += toml.dumps({k: v})
            toml_str += '\n'
            for k, v in self.definitions.items():
                if k not in self.used_definitions:
                    continue
                if isinstance(v, dict):
                    toml_str += toml.dumps({'definitions': {k: v}}).replace('\n\n', '\n') + '\n'
        toml_str += '# =============== tasks ===============\n'
        for i_task, task in enumerate(self.tasks):
            toml_str += f'# ------------- task {i_task} -------------\n'
            s = toml.dumps({'tasks': [task]}).replace('\n\n', '\n')
            toml_str += s + '\n'
        return toml_str

    @classmethod
    def parse_skip_task_ids(cls, skip_task_ids_):
        if '-' in skip_task_ids_:
            a, b = skip_task_ids_.split('-', 1)
            return list(range(int(a), int(b) + 1))
        return [int(i) for i in skip_task_ids_.split(',') if i != '']

    def run(
        self,
        skip_task_ids_: str = '',
        target_tasks_: str = '',
        force_rerun: bool = False,
        suppress_plot: bool = False,
        force_replot: bool = False,
        report_only: bool = False,
        dryrun: bool = False,
    ):
        skip_task_ids = type(self).parse_skip_task_ids(skip_task_ids_)
        target_tasks = [t for t in target_tasks_.split(',') if t != '']
        assert len(skip_task_ids) == 0 or len(target_tasks) == 0

        dm = TimeSeriesDataManager(**self.get_data_param())
        task_runners = self.create_task_runners(dm)

        report_path = self.out_path / 'report.md'
        any_task_run = False
        info = {}
        with measure_time(info):
            if not report_only:
                for i_task, task_runner in enumerate(task_runners):
                    if len(target_tasks) > 0:
                        if not task_runner.name in target_tasks:
                            continue
                    if i_task in skip_task_ids:
                        continue
                    if task_runner.result_path.exists():
                        if force_rerun:
                            print('Result already exists, but rerunning.')
                        else:
                            print('Result already exists. Skipping.')
                            continue
                    if dryrun:
                        print(f'[Dry-run] Would run task: {task_runner.name}')
                        continue
                    if not any_task_run:
                        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
                        any_task_run = True
                    task_runner.run()
                    report(report_path, self.to_toml_str(), task_runners, suppress_plot)
            if not any_task_run and not dryrun:
                report(report_path, self.to_toml_str(), task_runners, suppress_plot, force_replot)
        print(f'Finished: {report_path.as_posix()} ({info.get("elapsed")})')


class WorkflowTemplateResolver:
    """
    Resolves a template into tasks when a template is specified instead of tasks.
    """
    Type = Enum('Type', [
        'train_with_baseline',
        'train_with_baseline_multiparams',
        'train_multiparams',
        'repeat',
    ])

    @classmethod
    def update(cls, d_dst, d_src, keys, rename=None):
        rename = rename or {}
        for key in keys:
            d_dst[rename.get(key, key)] = copy.deepcopy(d_src[key])
        return d_dst

    @classmethod
    def _optimizer_keys(cls, d):
        if 'optimizer_groups' in d:
            assert 'optimizer' not in d and 'lr_scheduler' not in d, \
                "Set either 'optimizer_groups' or 'optimizer'/'lr_scheduler', not both"
            return ['optimizer_groups']
        keys = ['optimizer', 'lr_scheduler']
        if 'lr_scheduler_interval' in d:
            keys.append('lr_scheduler_interval')
        return keys

    @classmethod
    def get_task_eval_baseline(cls, d):
        task = {'task_type': 'eval', 'name': 'Eval Baseline', 'dump_pred_data': False}
        keys = ['data_range_eval', 'criterion_eval', 'baseline_model']
        rename = {'criterion_eval': 'criterion', 'baseline_model': 'model'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_pilot(cls, d, i_taskset=0):
        task = {
            'task_type': 'train', 'name': f'Pilot {i_taskset}', 'early_stop': True,
            'model_state_path': None, 'seed': 0,
        }
        keys = ['data_range_train_pilot', 'data_range_eval_pilot', 'criterion_eval'] + \
            ['model', 'batch_sampler', 'n_epoch', 'patience'] + cls._optimizer_keys(d)
        rename = {f'data_range_{t}_pilot': f'data_range_{t}' for t in ['train', 'eval']}
        rename |= {'criterion_eval': 'criterion'}
        if 'criterion_train' in d:
            keys.append('criterion_train')
            rename |= {'criterion_train': 'criterion'}
        if 'criterion_train_target' in d:
            keys.append('criterion_train_target')
            rename |= {'criterion_train_target': 'criterion_target'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_train(cls, d, i_taskset=0, i_taskset_pilot=None, nopilot=False):
        task = {
            'task_type': 'train', 'name': f'Train {i_taskset}', 'early_stop': False,
            'model_state_path': None, 'seed': 0, 'n_batch': 0,
            'save_model_state_ini': False,
        }
        keys = ['data_range_train', 'criterion_eval', 'model', 'batch_sampler'] + \
            cls._optimizer_keys(d)
        if 'save_model_state_ini' in d:
            keys.append('save_model_state_ini')
        rename = {'criterion_eval': 'criterion'}
        if 'criterion_train' in d:
            keys.append('criterion_train')
            rename |= {'criterion_train': 'criterion'}
        if 'criterion_train_target' in d:
            keys.append('criterion_train_target')
            rename |= {'criterion_train_target': 'criterion_target'}
        if nopilot:
            keys.append('n_epoch')
        else:
            if i_taskset_pilot is None:
                i_taskset_pilot = i_taskset
            task['n_epoch'] = {'task_name': f'Pilot {i_taskset_pilot}'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_eval(cls, d, i_taskset=0):
        task = {'task_type': 'eval', 'name': f'Eval {i_taskset}'}
        task['model_state'] = {'task_name': f'Train {i_taskset}'}
        keys = ['data_range_eval', 'criterion_eval', 'model']
        rename = {'criterion_eval': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_eval_imprate(cls, d, i_taskset=0):
        task = {'task_type': 'eval', 'name': f'Eval ImpRate {i_taskset}'}
        task['model_state'] = {'task_name': f'Train {i_taskset}'}
        keys = ['data_range_eval', 'criterion_imprate', 'baseline_model', 'model']
        rename = {'criterion_imprate': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_tasks_baseline(cls, d):
        tasks = [cls.get_task_eval_baseline(d)]
        if 'criteria_additional' in d:
            for criterion in d['criteria_additional']:
                task = cls.get_task_eval_baseline(d)
                task['name'] = f'Eval Baseline {criterion}'
                task['criterion'] = criterion
                tasks.append(task)
        return tasks

    @classmethod
    def get_taskset(cls, d, i_taskset=0, i_taskset_pilot=None, nopilot=False):
        # When i_taskset_pilot is None, generate a Pilot for this taskset and
        # have Train reference it. When given, skip Pilot generation and have
        # Train reference Pilot of taskset `i_taskset_pilot`. When nopilot is
        # set, generate no Pilot and have Train use a fixed n_epoch instead.
        tasks = []
        if not nopilot and i_taskset_pilot is None:
            tasks.append(cls.get_task_pilot(d, i_taskset))
        tasks.append(cls.get_task_train(
            d, i_taskset, i_taskset_pilot=i_taskset_pilot, nopilot=nopilot,
        ))
        tasks.append(cls.get_task_eval(d, i_taskset))
        if 'criteria_additional' in d:
            for criterion in d['criteria_additional']:
                task = cls.get_task_eval(d, i_taskset)
                task['name'] = f'Eval {criterion} {i_taskset}'
                task['criterion'] = criterion
                tasks.append(task)
        if 'criterion_imprate' in d:
            tasks.append(cls.get_task_eval_imprate(d, i_taskset))
        return tasks

    @classmethod
    def get_tasks_train_with_baseline(cls, d):
        return cls.get_tasks_baseline(d) + cls.get_taskset(d)

    @classmethod
    def _get_tasksets_multiparams(cls, d):
        tasks = []
        for i_taskset, params_target in enumerate(d['params']):
            params_target = dict(params_target)
            nopilot = params_target.pop('nopilot', False)
            i_taskset_pilot = params_target.pop('i_taskset_pilot', None)
            assert not (nopilot and i_taskset_pilot is not None), \
                'i_taskset_pilot cannot be used when nopilot is set'
            tasks_ = cls.get_taskset(
                d, i_taskset, i_taskset_pilot=i_taskset_pilot, nopilot=nopilot,
            )
            for k, v in params_target.items():
                if k == 'criterion_train':
                    for task in tasks_:
                        if task['task_type'] == 'train':
                            task['criterion'] = v
                elif k == 'criterion_train_target':
                    for task in tasks_:
                        if task['task_type'] == 'train':
                            task['criterion_target'] = v
                else:
                    for task in tasks_:
                        if k in task:
                            task[k] = v
            tasks += tasks_
        return tasks

    @classmethod
    def get_tasks_train_with_baseline_multiparams(cls, d):
        return cls.get_tasks_baseline(d) + cls._get_tasksets_multiparams(d)

    @classmethod
    def get_tasks_train_multiparams(cls, d):
        return cls._get_tasksets_multiparams(d)

    @classmethod
    def get_tasks_repeat(cls, d):
        assert set(d) == {'template_type', 'tasks', 'params'}
        counter = {}
        tasks = []
        for i_task, task_raw in enumerate(d['tasks']):
            assert 'task_type' in task_raw
            task_type = task_raw['task_type']
            if task_type not in counter:
                counter[task_type] = 0
            counter[task_type] += 1
            task_name_base = task_type.capitalize() + ' ' + str(counter[task_type] - 1)
            for i_param, param_raw in enumerate(d['params']):
                task = copy.deepcopy(task_raw | param_raw)
                param_name = task.pop('param_name', i_param)
                task['name'] = f'{task_name_base} {param_name}'
                tasks.append(task)
        return tasks

    @classmethod
    def resolve(cls, d: dict) -> dict:
        if 'template' not in d:
            return d
        if 'tasks' in d:
            raise ValueError('Template and tasks cannot be set at the same time')
        d_tmpl = d.pop('template')
        type_ = cls.Type[d_tmpl['template_type']]
        if type_ == cls.Type.train_with_baseline:
            d['tasks'] = cls.get_tasks_train_with_baseline(d_tmpl)
        if type_ == cls.Type.train_with_baseline_multiparams:
            d['tasks'] = cls.get_tasks_train_with_baseline_multiparams(d_tmpl)
        if type_ == cls.Type.train_multiparams:
            d['tasks'] = cls.get_tasks_train_multiparams(d_tmpl)
        if type_ == cls.Type.repeat:
            d['tasks'] = cls.get_tasks_repeat(d_tmpl)
        return d


def resolve_definition_includes(d: dict, p: Path | None):
    # Resolve definition_includes by merging definitions from listed files.
    # Later files override earlier ones; the config's own definitions have
    # the highest priority. "relpath" is resolved relative to the config TOML's
    # directory and is only usable when the config is loaded from a TOML file.
    definition_includes = d.pop('definition_includes', None)
    definition_includes_data = d.pop('definition_includes_data', None)
    if not definition_includes:
        return
    merged = {}
    for include in definition_includes:
        assert isinstance(include, dict), 'include must be specified as a table.'
        assert set(include) in ({'bundled'}, {'path'}, {'relpath'}), \
            'include must have exactly one of "bundled", "path", or "relpath".'
        if 'bundled' in include:
            included = get_definitions(include['bundled'], definition_includes_data)
        else:
            if 'relpath' in include:
                assert p is not None, \
                    'relpath cannot be used when the config is not loaded from a TOML file.'
                include_path = (p.parent / Path(include['relpath']).expanduser())
            else:
                include_path = Path(include['path']).expanduser()
            assert include_path.exists(), \
                f'definition_includes file not found: {include_path.as_posix()}'
            included = load_toml(include_path)
        merged.update(included['definitions'])
    merged.update(d.get('definitions', {}))
    d['definitions'] = merged


def load_config_from_path(p: Path):
    d = load_toml(p)

    # If out_dir is set to __CONFIG_STEM__, resolve it to the config file stem.
    out_dir = d.get('out_dir')
    if out_dir == '__CONFIG_STEM__':
        assert all('out_dir' not in t for t in d.get('tasks', [])), \
            'Do not specify out_dir for individual tasks when using __CONFIG_STEM__.'
        d['out_dir'] = (p.parent / p.stem).as_posix()

    return d


def normalize_config(source: dict | Path | str):
    p = None
    if isinstance(source, dict):
        d = source
    elif isinstance(source, Path):
        d = load_config_from_path(source)
        p = source
    elif isinstance(source, str):
        s = source.strip()
        path_or_none = as_path_if_length_safe(s)
        if isinstance(path_or_none, Path) and path_or_none.is_file():
            d = load_config_from_path(path_or_none)
            p = path_or_none
        else:
            d = toml.loads(s)
    else:
        raise AssertionError(f'Cannot normalize config from {type(source).__name__}')
    resolve_definition_includes(d, p)
    return d


def run(
    source: dict | Path | str,
    skip_task_ids_: str = '',
    target_tasks_: str = '',
    force_rerun: bool = False,
    suppress_plot: bool = False,
    force_replot: bool = False,
    report_only: bool = False,
    dryrun: bool = False,
):
    d = normalize_config(source)
    d = WorkflowTemplateResolver.resolve(d)
    wf = Workflow(**d)
    wf.run(
        skip_task_ids_=skip_task_ids_,
        target_tasks_=target_tasks_,
        force_rerun=force_rerun,
        suppress_plot=suppress_plot,
        force_replot=force_replot,
        report_only=report_only,
        dryrun=dryrun,
    )
    return wf
