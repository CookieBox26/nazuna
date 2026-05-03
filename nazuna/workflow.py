import copy
import dataclasses
from enum import Enum
from pathlib import Path
import toml
import torch
from nazuna.datasets import get_path
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
    data: dict = None
    device: str = ''
    definitions: dict = None
    tasks: list[dict] = None

    @classmethod
    def _to_snake(cls, s):
        s = s.translate(str.maketrans('()=', '___'))
        return '_'.join(s.lower().split())

    def __post_init__(self):
        self.out_dir = self.out_dir or f'out/{get_timestamp()}/'
        self.out_path = Path(self.out_dir).expanduser()
        if (not self.exist_ok) and self.out_path.exists():
            raise FileExistsError(f'Already exists: {self.out_path.as_posix()}')
        assert self.data is not None
        self.device = self.device or \
            str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        assert self.tasks is not None
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

    def get_data_param(self):
        param = copy.deepcopy(self.data)
        if isinstance(param['path'], (list, tuple)):
            param['path'] = get_path(*param['path'])
        return param

    def parse_task_runner_config(self, i_task):
        params = copy.deepcopy(self.tasks[i_task])
        self.task_type = params.pop('task_type')
        task_runner_cls = TaskType[self.task_type].value
        params.setdefault('device', self.device)
        params.setdefault('exist_ok', self.exist_ok)

        if 'n_epoch' in params and isinstance(params['n_epoch'], dict):
            target_path = self.out_paths[params['n_epoch']['task_name']]
            params['n_epoch_path'] = target_path / 'result.toml'
            params['n_epoch_path_defer'] = True
            del params['n_epoch']

        # If the following parameters are specified as strings,
        # resolve them using the definitions.
        for target in [
            'criterion', 'baseline_model', 'model',
            'batch_sampler', 'optimizer', 'lr_scheduler',
        ]:
            if target in params and isinstance(params[target], str):
                definition = self.definitions[params[target]]
                if 'base' in definition:
                    definition_base = self.definitions[definition['base']]
                    params[target] = {
                        'cls_path': definition_base['cls_path'],
                        'params': copy.deepcopy(definition_base['params']),
                    }
                    if 'cls_path' in definition:
                        params[target]['cls_path'] = definition['cls_path']
                    if 'params' in definition:
                        params[target]['params'].update(definition['params'])
                else:
                    params[target] = {
                        'cls_path': definition['cls_path'],
                        'params': copy.deepcopy(definition['params']),
                    }
        for target in [
            'batch_size_eval', 'data_range_train', 'data_range_eval',
            'n_epoch',
        ]:
            if target in params and isinstance(params[target], str):
                params[target] = self.definitions[params[target]]

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
            ['out_dir', 'exist_ok', 'data', 'device', 'definitions', 'tasks'], \
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

    def save_toml(self):
        self.conf_path = self.out_path / 'config.toml'
        self.conf_path.write_text(self.to_toml_str(), newline='\n', encoding='utf8')

    @classmethod
    def parse_skip_task_ids(cls, skip_task_ids_):
        if '-' in skip_task_ids_:
            a, b = skip_task_ids_.split('-', 1)
            return list(range(int(a), int(b) + 1))
        return [int(i) for i in skip_task_ids_.split(',') if i != '']

    def run(self, skip_task_ids_: str = ''):
        skip_task_ids = type(self).parse_skip_task_ids(skip_task_ids_)
        dm = TimeSeriesDataManager(**self.get_data_param())
        task_runners = self.create_task_runners(dm)
        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        self.save_toml()
        info = {}
        with measure_time(info):
            for i_task, task_runner in enumerate(task_runners):
                if i_task in skip_task_ids:
                    continue
                task_runner.run()
        report_path = self.out_path / 'report.md'
        report(report_path, self.to_toml_str(), task_runners)
        print(f'Finished all tasks: {report_path.as_posix()} ({info["elapsed"]})')


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
        task = {'task_type': 'eval', 'name': 'Eval Baseline'}
        keys = ['data_range_eval', 'criterion_eval', 'baseline_model']
        rename = {'criterion_eval': 'criterion', 'baseline_model': 'model'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_pilot(cls, d, i_trial=0):
        task = {'task_type': 'train', 'name': f'Pilot {i_trial}', 'early_stop': True}
        keys = ['data_range_train_pilot', 'data_range_eval_pilot', 'criterion_target'] + \
            ['model', 'batch_sampler', 'optimizer', 'lr_scheduler', 'n_epoch', 'patience']
        rename = {f'data_range_{t}_pilot': f'data_range_{t}' for t in ['train', 'eval']}
        rename |= {'criterion_target': 'criterion'}
        return cls.update(task, d, keys, rename)

    @classmethod
    def get_task_train(cls, d, i_trial=0):
        task = {'task_type': 'train', 'name': f'Train {i_trial}'}
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
    out_dir = d.get('out_dir')
    if out_dir == '__CONFIG_STEM__':
        d['out_dir'] = (p.parent / p.stem).as_posix()
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
):
    d = normalize_config(source)
    d = WorkflowTemplateResolver.resolve(d)
    wf = Workflow(**d)
    wf.run(skip_task_ids_=skip_task_ids_)
