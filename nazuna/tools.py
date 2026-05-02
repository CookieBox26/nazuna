from nazuna.workflow import Workflow, load_config_from_path
from pathlib import Path
import toml
import re
import types
import pandas as pd
import collections


def parse_param_conf(model_cls, overrides=None, n_derived=0):
    doc = model_cls.__doc__.splitlines()
    key = f'[definitions.{model_cls.__name__}]'
    flag = False
    indent = None
    text = ''
    for line in doc:
        if flag:
            if line.strip() == '```':
                break
            s = line[indent:]
            text += s + '\n'
        if key in line:
            indent = line.find(key)
            text += line[indent:] + '\n'
            flag = True

    if overrides is not None:
        for k, v_raw in overrides.items():
            pattern = rf'(?m)^({re.escape(k)}\s*=\s*)([^#\n]*)(\s*(?:#.*)?)$'
            v = f'"{v_raw}"' if type(v_raw) is str else v_raw
            text = re.sub(pattern, rf'\g<1>{v}  \g<3>', text)

    for i in range(n_derived):
        text += '\n'
        text += f'[definitions.{model_cls.__name__}_{i:02}]\n'
        text += f'base = "{model_cls.__name__}"\n'
        text += f'# [definitions.{model_cls.__name__}_{i:02}.params]\n'
    return text


class WorkflowResult(Workflow):
    @classmethod
    def from_conf_toml_path(cls, p: Path | str):
        conf_path_raw = load_config_from_path(Path(p))
        conf_path = Path(conf_path_raw['out_dir']) / 'config.toml'
        conf_dict = toml.loads(conf_path.read_text(encoding='utf8'))
        conf_dict['exist_ok'] = True
        for task in conf_dict['tasks']:
            task['exist_ok'] = True
        obj = cls(**conf_dict)
        obj.rename = None
        return obj

    def get_conf_and_result(self, task_name, as_sn=False):
        if task_name not in self.task_names:
            return None if as_sn else (None, None)
        i_task = self.task_names.index(task_name)
        _, conf = self.parse_task_runner_config(i_task)
        out_path = self.out_paths[task_name]
        if self.rename is not None:
            out_path = out_path.as_posix()
            for k, v in self.rename.items():
                out_path = out_path.replace(k, v)
            out_path = Path(out_path)
        result_path = out_path / 'result.toml'
        result = toml.loads(result_path.read_text(encoding='utf8'))
        if as_sn:
            return types.SimpleNamespace(conf=conf, result=result)
        return (conf, result)

    def get_trial(self, i_trial=None):
        d = {}
        suffix = '' if i_trial is None else f' {i_trial}'
        d['baseline'] = self.get_conf_and_result('Eval Baseline', as_sn=True)
        if f'Pilot{suffix}' in self.task_names:
            d['pilot'] = self.get_conf_and_result(f'Pilot{suffix}', as_sn=True)
        else:
            d['pilot'] = self.get_conf_and_result(f'Pilot 0', as_sn=True)
        d['train'] = self.get_conf_and_result(f'Train{suffix}', as_sn=True)
        d['eval'] = self.get_conf_and_result(f'Eval{suffix}', as_sn=True)
        d['imprate'] = self.get_conf_and_result(f'Eval ImpRate{suffix}', as_sn=True)
        return types.SimpleNamespace(**d)

    @classmethod
    def cls_path_to_name(cls, cls_path):
        return cls_path.rsplit('.', 1)[-1]

    @classmethod
    def params_to_str(cls, params):
        s = '(' + ','.join([f'{k}={v}' for k, v in params.items()]) + ')'
        return s.replace('\'', '').replace(' ', '')

    @classmethod
    def cls_to_str(cls, conf):
        return cls.cls_path_to_name(conf['cls_path']) + cls.params_to_str(conf['params'])

    @classmethod
    def get_row(cls, trial, index_):
        if any(k is None for k in [
            trial.baseline, trial.pilot, trial.train,
            trial.eval, trial.imprate,
        ]):
            return None
        row = collections.OrderedDict([
            ('index', index_),
            ('model', cls.cls_path_to_name(trial.train.conf['model']['cls_path'])),
            ('criterion', cls.cls_to_str(trial.eval.conf['criterion'])),
            ('loss(bl)', trial.baseline.result['loss_per_sample']),
            ('loss(mo)', trial.eval.result['loss_per_sample']),
            ('imprate', trial.imprate.result['loss_per_sample']),
            ('seed', trial.train.conf.get('seed', 0)),
            ('n_epoch', trial.pilot.result['i_epoch_best'] + 1),
        ])
        row_model = collections.OrderedDict([('index', index_)])
        for k, v in trial.train.conf['model']['params'].items():
            if k == 'scaler_cls_path':
                row_model['scaler_cls'] = cls.cls_path_to_name(v)
            elif k == 'scaler_params':
                row_model[k] = cls.params_to_str(v)
            else:
                row_model[k] = v
        bs = cls.cls_to_str(trial.train.conf['batch_sampler'])
        row_train = collections.OrderedDict([
            ('index', index_),
            ('criterion', cls.cls_to_str(trial.train.conf['criterion'])),
            ('batch_sampler', re.sub('^BatchSampler', '', bs).replace('batch_size=', '')),
            ('optimizer', cls.cls_to_str(trial.train.conf['optimizer'])),
            ('lr_scheduler', cls.cls_to_str(trial.train.conf['lr_scheduler'])),
        ])
        return {'loss': row, 'model': row_model, 'train': row_train}

    @classmethod
    def get_rows(cls, trials):
        rows = {}
        for index_, trial in trials.items():
            row = cls.get_row(trial, index_=index_)
            if row is None:
                continue
            for k, v in row.items():
                if k not in rows:
                    rows[k] = []
                rows[k].append(v)
        return rows

    @classmethod
    def rows_to_df(cls, rows):
        common = list(rows[0].keys())
        for row in rows[1:]:
            s = set(row.keys())
            common = [c for c in common if c in s]
        df = pd.DataFrame([{k: r[k] for k in common} for r in rows])
        df = df.set_index('index').rename_axis(None)
        return df

    @classmethod
    def get_dfs(cls, trials):
        rows = cls.get_rows(trials)
        return {k: cls.rows_to_df(v) for k, v in rows.items()}
