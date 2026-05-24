from nazuna.workflow import normalize_config, WorkflowTemplateResolver, Workflow
from nazuna.task_runners import BaseTaskRunner
from nazuna.models._base import BasicBaseModel
import re
import types
import pandas as pd
import collections
from nazuna.utils import load_class, load_toml


class WorkflowResult(Workflow):
    @classmethod
    def load(cls, source):
        d = normalize_config(source)
        if d is None:
            return None
        d = WorkflowTemplateResolver.resolve(d)
        exist_ok_org = d['exist_ok']
        d['exist_ok'] = True
        wf = cls(**d)
        wf.exist_ok = exist_ok_org
        wf.criteria_additional = []  # Ex. ['MAE']
        return wf

    def get_conf_and_result(self, task_name, as_sn=False):
        if task_name not in self.task_names:
            return None if as_sn else (None, None)
        i_task = self.task_names.index(task_name)
        _, conf = self.parse_task_runner_config(i_task)

        result_path = BaseTaskRunner.to_result_path(self.out_paths[task_name])
        if not result_path.is_file():
            print(f'No result file found: {result_path}')
            return None if as_sn else (None, None)
        result = load_toml(result_path)

        if as_sn:
            return types.SimpleNamespace(conf=conf, result=result)
        return (conf, result)

    def get_trial(self, i_trial=None):
        d = {}
        suffix = '' if i_trial is None else f' {i_trial}'
        d['baseline'] = self.get_conf_and_result('Eval Baseline', as_sn=True)
        d['pilot'] = self.get_conf_and_result(f'Pilot{suffix}', as_sn=True)
        d['train'] = self.get_conf_and_result(f'Train{suffix}', as_sn=True)
        d['eval'] = self.get_conf_and_result(f'Eval{suffix}', as_sn=True)
        d['criteria_additional'] = self.criteria_additional
        for criterion_a in self.criteria_additional:
            d[f'eval_baseline_{criterion_a.lower()}'] = \
                self.get_conf_and_result(f'Eval Baseline {criterion_a}', as_sn=True)
            d[f'eval_{criterion_a.lower()}'] = \
                self.get_conf_and_result(f'Eval {criterion_a}{suffix}', as_sn=True)
        d['imprate'] = self.get_conf_and_result(f'Eval ImpRate{suffix}', as_sn=True)
        return types.SimpleNamespace(**d)

    def get_trials(self, prefix, target_trials=None):
        if target_trials is None:
            target_trials = list(range(10))
        trials = {}
        for i_trial in target_trials:
            trial = self.get_trial(i_trial)
            if trial.pilot is None:
                break
            trials[f'{prefix}({i_trial})'] = trial
        return trials

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
    def shorten_time(cls, s):
        m = re.fullmatch(r'(\d+) min (\d+) sec', s)
        return f'{int(m.group(1)):2d}m{int(m.group(2)):2d}s'

    @classmethod
    def get_row(cls, trial, index_):
        if any(k is None for k in [
            trial.baseline, trial.pilot, trial.train,
            trial.eval, trial.imprate,
        ]):
            return None
        criterion = cls.cls_path_to_name(trial.eval.conf['criterion']['cls_path'])
        row = collections.OrderedDict([
            ('index', index_),
            ('Model', cls.cls_path_to_name(trial.train.conf['model']['cls_path'])),
            # ('criterion', cls.cls_to_str()),
            (f'{criterion}_bl', trial.baseline.result['loss_per_sample']),
            (f'{criterion}_mo', trial.eval.result['loss_per_sample']),
        ])
        for criterion_a in trial.criteria_additional:
            row[f'{criterion_a}_bl'] = \
                getattr(trial, f'eval_baseline_{criterion_a.lower()}').result['loss_per_sample']
            row[f'{criterion_a}_mo'] = \
                getattr(trial, f'eval_{criterion_a.lower()}').result['loss_per_sample']
        row['ImpRate'] = trial.imprate.result['loss_per_sample']
        row['seed'] = trial.train.conf.get('seed', 0)
        row['n_epoch'] = str(trial.pilot.result['i_epoch_best'] + 1) + ' / ' + \
            str(trial.pilot.conf['n_epoch'])
        row['Elapsed_pilot'] = cls.shorten_time(trial.pilot.result['elapsed'])
        row['Elapsed'] = cls.shorten_time(trial.train.result['elapsed'])
        row['n_parameters'] = trial.eval.result.get('parameters_trainable', -1)

        row_model = collections.OrderedDict([('index', index_)])
        mo = trial.train.conf['model']
        cls_, params_ = load_class(mo['cls_path']), mo['params']
        if issubclass(cls_, BasicBaseModel):
            params_ = cls_._resolve_seq_len(params_)
        for k, v in params_.items():
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
