import copy
import dataclasses
from typing import ClassVar
from enum import Enum
from pathlib import Path
import toml
import torch
from nazuna.datasets import get_dataset_path
from nazuna.data_manager import TimeSeriesDataManager
from nazuna.task_runners import TaskType
from nazuna.report import report
from nazuna.utils import as_path_if_length_safe, measure_time, get_timestamp


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

        exist_ok (bool = False): Whether to allow the output path to already exist.
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
    exist_ok: bool = False
    device: str = ''
    definitions: dict = None
    data: dict = None
    tasks: list[dict] = None

    # If any of the following task keys is specified as a string,
    # resolve it using the definitions
    task_keys_accepting_definitions: ClassVar[list[str]] = [
        # a dict with cls_path and params keys
        'criterion', 'baseline_model', 'model',
        'batch_sampler', 'optimizer', 'lr_scheduler',
        # a list
        'batch_size_eval', 'data_range_train', 'data_range_eval',
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

        if (not self.exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')

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
            self.out_paths[name] = Path(self.tasks[i_task]['out_dir'])

        # Dry-run definition resolution
        if isinstance(self.data, str):
            _ = self.get_definition(self.data)
        for i_task, _ in enumerate(self.tasks):
            params = copy.deepcopy(self.tasks[i_task])
            for target in type(self).task_keys_accepting_definitions:
                if target in params and isinstance(params[target], str):
                    _ = self.get_definition(params[target])

    def get_data_param(self):
        if isinstance(self.data, str):
            param = self.get_definition(self.data)
        else:
            param = copy.deepcopy(self.data)
        if isinstance(param['path'], (list, tuple)):
            param['path'] = get_dataset_path(*param['path'])
        return param

    def get_definition(self, name):
        assert name in self.definitions, f'There is no definition named {name}'
        definition_raw = self.definitions[name]
        if not isinstance(definition_raw, dict):
            return copy.deepcopy(definition_raw)
        definition = {}
        if 'base' in definition_raw:
            # If a definition has a base key, first copy the base definition
            definition_base = self.definitions[definition_raw['base']]
            for k, v in definition_base.items():
                 assert k != 'base', 'Multiple inheritance is not supported'
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

        if 'n_epoch' in params and isinstance(params['n_epoch'], dict):
            target_path = self.out_paths[params['n_epoch']['task_name']]
            params['n_epoch_path'] = target_path / 'result.toml'
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
            ['out_dir', 'exist_ok', 'device', 'definitions', 'data', 'tasks'], \
            'Update the custom TOML stringification when fields are changed.'
        header = {'out_dir': self.out_dir, 'exist_ok': self.exist_ok, 'device': self.device}
        toml_str = toml.dumps(header) + '\n'
        toml_str += '# =============== data ===============\n'
        toml_str += toml.dumps({'data': self.data}) + '\n'
        if self.definitions:
            toml_str += '# =============== definitions ===============\n'
            toml_str += '[definitions]\n'
            for k, v in self.definitions.items():
                if not isinstance(v, dict):
                    toml_str += toml.dumps({k: v})
            toml_str += '\n'
            for k, v in self.definitions.items():
                if isinstance(v, dict):
                    toml_str += toml.dumps({'definitions': {k: v}}).replace('\n\n', '\n') + '\n'
        toml_str += '# =============== tasks ===============\n'
        for i_task, task in enumerate(self.tasks):
            toml_str += f'# ------------- task {i_task} -------------\n'
            s = toml.dumps({'tasks': [task]}).replace('\n\n', '\n')
            toml_str += s + '\n'
        return toml_str

    def save_toml(self, conf_save_path):
        conf_save_path.write_text(self.to_toml_str(), newline='\n', encoding='utf8')

    @classmethod
    def load_from_path_without_validation(cls, conf_path):
        conf_dict = toml.loads(conf_path.read_text(encoding='utf8'))
        exist_ok_org = conf_dict['exist_ok']
        conf_dict['exist_ok'] = True
        wf = Workflow(**conf_dict)
        wf.exist_ok = exist_ok_org
        return wf

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
        suppress_plot: bool = False,
        force_replot: bool = False,
        report_only: bool = False,
    ):
        skip_task_ids = type(self).parse_skip_task_ids(skip_task_ids_)
        target_tasks = [t for t in target_tasks_.split(',') if t != '']
        assert len(skip_task_ids) == 0 or len(target_tasks) == 0

        dm = TimeSeriesDataManager(**self.get_data_param())
        task_runners = self.create_task_runners(dm)

        conf_save_path = self.out_path / 'config.toml'
        report_path = self.out_path / 'report.md'
        any_task_run = False
        info = {}
        with measure_time(info):
            if not report_only:
                self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
                self.save_toml(conf_save_path)
                for i_task, task_runner in enumerate(task_runners):
                    if len(target_tasks) > 0:
                        if not task_runner.name in target_tasks:
                            continue
                    if i_task in skip_task_ids:
                        continue
                    task_runner.run()
                    report(report_path, self.to_toml_str(), task_runners, suppress_plot)
                    any_task_run = True
            if not any_task_run:
                wf_saved = type(self).load_from_path_without_validation(conf_save_path)
                toml_str = wf_saved.to_toml_str()
                report(report_path, toml_str, task_runners, suppress_plot, force_replot)
        print(f'Finished: {report_path.as_posix()} ({info.get("elapsed")})')


class WorkflowTemplateResolver:
    """
    Resolves a template into tasks when a template is specified instead of tasks.
    """
    Type = Enum('Type', [
        'train_with_baseline',
        'train_with_baseline_multiseeds',
        'train_with_baseline_multimodels',
        'train_with_baseline_multiparams',
    ])

    @classmethod
    def update(cls, d_dst, d_src, keys, rename=None):
        rename = rename or {}
        for key in keys:
            d_dst[rename.get(key, key)] = copy.deepcopy(d_src[key])
        return d_dst

    @classmethod
    def get_task_eval_baseline(cls, d):
        task = {'task_type': 'eval', 'name': 'Eval Baseline', 'dump_pred_data': False}
        keys = ['data_range_eval', 'criterion_eval', 'baseline_model']
        rename = {'criterion_eval': 'criterion', 'baseline_model': 'model'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_pilot(cls, d, i_trial=0):
        task = {
            'task_type': 'train', 'name': f'Pilot {i_trial}', 'early_stop': True,
            'model_state_path': None,
        }
        keys = ['data_range_train_pilot', 'data_range_eval_pilot', 'criterion_target'] + \
            ['model', 'batch_sampler', 'optimizer', 'lr_scheduler', 'n_epoch', 'patience']
        rename = {f'data_range_{t}_pilot': f'data_range_{t}' for t in ['train', 'eval']}
        rename |= {'criterion_target': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_train(cls, d, i_trial=0):
        task = {'task_type': 'train', 'name': f'Train {i_trial}', 'model_state_path': None}
        task['n_epoch'] = {'task_name': f'Pilot {i_trial}'}
        keys = ['data_range_train', 'criterion_target'] + \
            ['model', 'batch_sampler', 'optimizer', 'lr_scheduler']
        rename = {'criterion_target': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_eval(cls, d, i_trial=0):
        task = {'task_type': 'eval', 'name': f'Eval {i_trial}'}
        task['model_state'] = {'task_name': f'Train {i_trial}'}
        keys = ['data_range_eval', 'criterion_eval', 'model']
        rename = {'criterion_eval': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_eval_imprate(cls, d, i_trial=0):
        task = {'task_type': 'eval', 'name': f'Eval ImpRate {i_trial}'}
        task['model_state'] = {'task_name': f'Train {i_trial}'}
        keys = ['data_range_eval', 'criterion_imprate', 'baseline_model', 'model']
        rename = {'criterion_imprate': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_baseline(cls, d):
        tasks = [cls.get_task_eval_baseline(d)]
        if 'criteria_additional' in d:
            for criterion in d['criteria_additional']:
                task = cls.get_task_eval_baseline(d)
                task['name'] = f'Eval Baseline {criterion}'
                task['criterion'] = criterion
                tasks.append(task)
        return tasks

    @classmethod
    def get_trial(cls, d, i_trial=0):
        tasks = [
            cls.get_task_pilot(d, i_trial),
            cls.get_task_train(d, i_trial),
            cls.get_task_eval(d, i_trial),
        ]
        if 'criteria_additional' in d:
            for criterion in d['criteria_additional']:
                task = cls.get_task_eval(d, i_trial)
                task['name'] = f'Eval {criterion} {i_trial}'
                task['criterion'] = criterion
                tasks.append(task)
        if 'criterion_imprate' in d:
            tasks.append(cls.get_task_eval_imprate(d, i_trial))
        return tasks

    @classmethod
    def get_tasks_train_with_baseline(cls, d):
        return cls.get_baseline(d) + cls.get_trial(d)

    @classmethod
    def get_tasks_train_with_baseline_multiseeds(cls, d):
        tasks = cls.get_baseline(d)
        for i_trial, seed in enumerate(d['seeds']):
            tasks_ = cls.get_trial(d, i_trial)
            tasks_[0]['seed'] = seed
            tasks_[1]['seed'] = seed
            tasks += tasks_
        return tasks

    @classmethod
    def get_tasks_train_with_baseline_multimodels(cls, d):
        tasks = cls.get_baseline(d)
        for i_trial, model in enumerate(d['models']):
            tasks_ = cls.get_trial(d, i_trial)
            for i_task in range(len(tasks_)):
                tasks_[i_task]['model'] = model
            tasks += tasks_
        return tasks

    @classmethod
    def get_tasks_train_with_baseline_multiparams(cls, d):
        tasks = cls.get_baseline(d)
        for i_trial, params in enumerate(d['params']):
            tasks_ = cls.get_trial(d, i_trial)
            for k, v in params.items():
                for i_task in range(len(tasks_)):
                    if k in tasks_[i_task]:
                        tasks_[i_task][k] = v
            tasks += tasks_
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
        if type_ == cls.Type.train_with_baseline_multiseeds:
            d['tasks'] = cls.get_tasks_train_with_baseline_multiseeds(d_tmpl)
        if type_ == cls.Type.train_with_baseline_multimodels:
            d['tasks'] = cls.get_tasks_train_with_baseline_multimodels(d_tmpl)
        if type_ == cls.Type.train_with_baseline_multiparams:
            d['tasks'] = cls.get_tasks_train_with_baseline_multiparams(d_tmpl)
        return d


def load_config_from_path(p: Path):
    d = toml.loads(p.read_text(encoding='utf8'))

    # If out_dir is set to __CONFIG_STEM__, resolve it to the config file stem.
    out_dir = d.get('out_dir')
    if out_dir == '__CONFIG_STEM__':
        d['out_dir'] = (p.parent / p.stem).as_posix()

    # Resolve definition_includes by merging definitions from listed files.
    # Later files override earlier ones; the config's own definitions have
    # the highest priority. Paths may be absolute or relative to this file.
    includes = d.pop('definition_includes', None)
    if includes:
        merged = {}
        for include in includes:
            include_path = Path(include).expanduser()
            if not include_path.is_absolute():
                include_path = p.parent / include_path
            if not include_path.exists():
                raise FileNotFoundError(
                    f'definition_includes file not found: '
                    f'{include_path.as_posix()}'
                )
            included = toml.loads(include_path.read_text(encoding='utf8'))
            merged.update(included.get('definitions', {}))
        merged.update(d.get('definitions', {}))
        d['definitions'] = merged

    return d


def normalize_config(source: dict | Path | str):
    if isinstance(source, dict):
        return source
    if isinstance(source, Path):
        return load_config_from_path(source)
    if isinstance(source, str):
        s = source.strip()
        p = as_path_if_length_safe(s)
        if isinstance(p, Path):
            return load_config_from_path(p)
        return toml.loads(s)
    return None  # Cannot cast to dict


def run(
    source: dict | Path | str,
    skip_task_ids_: str = '',
    target_tasks_: str = '',
    suppress_plot: bool = False,
    force_replot: bool = False,
    report_only: bool = False,
):
    d = normalize_config(source)
    d = WorkflowTemplateResolver.resolve(d)
    wf = Workflow(**d)
    wf.run(
        skip_task_ids_=skip_task_ids_,
        target_tasks_=target_tasks_,
        suppress_plot=suppress_plot,
        force_replot=force_replot,
        report_only=report_only,
    )
