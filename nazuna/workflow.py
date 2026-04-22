import copy
import dataclasses
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
        for i_task, _ in enumerate(self.tasks):
            self.tasks[i_task].setdefault('name', f'Task {i_task}')
            name = self.tasks[i_task]['name']
            if name in self.out_paths:
                raise ValueError(f'Duplicate task name: {self.tasks[i_task]["name"]}')
            out_dir_default = (self.out_path / type(self)._to_snake(name)).as_posix()
            self.tasks[i_task].setdefault('out_dir', out_dir_default)
            self.out_paths[name] = Path(self.tasks[i_task]['out_dir'])

        self.out_path.mkdir(parents=True, exist_ok=self.exist_ok)
        self.to_toml_path()

    def get_data_param(self):
        param = copy.deepcopy(self.data)
        if isinstance(param['path'], (list, tuple)):
            param['path'] = get_path(*param['path'])
        return param

    def get_task_runner(self, i_task):
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
                d = self.definitions[params[target]]
                params[target] = {
                    'cls_path': d['cls_path'],
                    'params': copy.deepcopy(d['params']),
                }

        if 'model_state' in params:
            target_path = self.out_paths[params['model_state']['task_name']]
            params['model_state_path'] = target_path / 'model_state.pth'
            del params['model_state']

        return task_runner_cls, params

    @classmethod
    def from_toml_str(cls, toml_str: str | Path):
        d = toml.loads(toml_str)
        return cls(**d)

    @classmethod
    def from_toml_path(cls, toml_path: str | Path):
        p = toml_path
        if isinstance(p, str):
            p = Path(p)
        return cls.from_toml_str(p.read_text(encoding='utf8'))

    @classmethod
    def create(cls, source):
        if isinstance(source, cls):
            return source
        if isinstance(source, dict):
            return cls(**source)
        if isinstance(source, Path):
            return cls.from_toml_path(source)
        if isinstance(source, str):
            s = source.strip()
            p = as_path_if_length_safe(s)
            if isinstance(p, Path):
                return cls.from_toml_path(p)
            return cls.from_toml_str(s)
        raise ValueError('Cannot cast to Workflow')

    def to_toml_str(self):
        assert [field.name for field in dataclasses.fields(self)] == \
            ['out_dir', 'exist_ok', 'data', 'device', 'definitions', 'tasks'], \
            'Update the custom TOML stringification when fields are changed.'
        header = {
            'out_dir': self.out_dir,
            'exist_ok': self.exist_ok,
            'device': self.device,
        }
        toml_str = toml.dumps(header) + '\n'
        toml_str += '# =============== data ===============\n'
        toml_str += toml.dumps({'data': self.data}) + '\n'
        if self.definitions:
            toml_str += '# =============== definitions ===============\n'
            for k, v in self.definitions.items():
                s = toml.dumps({'definitions': {k: v}}).replace('\n\n', '\n')
                toml_str += s + '\n'
        toml_str += '# =============== tasks ===============\n'
        for i_task, task in enumerate(self.tasks):
            toml_str += f'# ------------- task {i_task} -------------\n'
            s = toml.dumps({'tasks': [task]}).replace('\n\n', '\n')
            toml_str += s + '\n'
        return toml_str

    def to_toml_path(self):
        self.conf_path = self.out_path / 'config.toml'
        self.conf_path.write_text(
            self.to_toml_str(), newline='\n', encoding='utf8',
        )

    def run(self, skip_task_ids_: str = ''):
        if '-' in skip_task_ids_:
            a, b = skip_task_ids_.split('-', 1)
            skip_task_ids = list(range(int(a), int(b) + 1))
        else:
            skip_task_ids = [
                int(i) for i in skip_task_ids_.split(',')
                if i != ''
            ]

        dm = TimeSeriesDataManager(**self.get_data_param())
        task_runners = []
        for i_task, _ in enumerate(self.tasks):
            cls_, params_ = self.get_task_runner(i_task)
            task_runners.append(cls_(dm=dm, **params_))

        result = {}
        with measure_time(result):
            for i_task, task_runner in enumerate(task_runners):
                if i_task in skip_task_ids:
                    continue
                task_runner.run()

        report_path = self.out_path / 'report.md'
        report(report_path, self.to_toml_str(), task_runners)
        print(f'Finished all tasks: {report_path.as_posix()} ({result["elapsed"]})')


def run(
    conf_: 'Workflow | dict | Path | str',
    skip_task_ids_: str = '',
):
    Workflow.create(conf_).run(skip_task_ids_=skip_task_ids_)
