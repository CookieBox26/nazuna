from nazuna.workflow import normalize_config, WorkflowTemplateResolver, Workflow
from nazuna.task_runners import BaseTaskRunner
from nazuna.models._base import BasicBaseModel
import re
import types
import numpy as np
import pandas as pd
import collections
from nazuna.utils import load_class, load_toml
import optuna
from pathlib import Path


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

    def get_taskset(self, i_taskset=None):
        d = {}
        suffix = '' if i_taskset is None else f' {i_taskset}'
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

    def get_tasksets(self, prefix, target_tasksets=None):
        if target_tasksets is None:
            target_tasksets = list(range(20))
        tasksets = {}
        for i_taskset in target_tasksets:
            taskset = self.get_taskset(i_taskset)
            tasksets[f'{prefix}({i_taskset})'] = taskset
        return tasksets

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
    def get_taskset_row(cls, taskset, index_):
        if any(k is None for k in [
            taskset.baseline, taskset.train,
            taskset.eval, taskset.imprate,
        ]):
            return None
        criterion = cls.cls_path_to_name(taskset.eval.conf['criterion']['cls_path'])
        row = collections.OrderedDict([
            ('index', index_),
            ('Model', cls.cls_path_to_name(taskset.train.conf['model']['cls_path'])),
            ('\\#Params', taskset.eval.result.get('parameters_trainable', -1)),
            (f'{criterion}(Naive)', taskset.baseline.result['loss_per_sample']),
            (f'{criterion}(Model)', taskset.eval.result['loss_per_sample']),
        ])
        row['Model'] = row['Model'].replace('Channelwise', 'Cw')
        row['Model'] = row['Model'].replace('CrossChannel', 'CC')
        row[f'{criterion}(ImpRate)'] = 1.0 - row[f'{criterion}(Model)'] / row[f'{criterion}(Naive)']
        for criterion_a in taskset.criteria_additional:
            row[f'{criterion_a}(Naive)'] = \
                getattr(taskset, f'eval_baseline_{criterion_a.lower()}').result['loss_per_sample']
            row[f'{criterion_a}(Model)'] = \
                getattr(taskset, f'eval_{criterion_a.lower()}').result['loss_per_sample']
            row[f'{criterion_a}(ImpRate)'] = \
                1.0 - row[f'{criterion_a}(Model)'] / row[f'{criterion_a}(Naive)']
        row['ImpRate\\dag'] = taskset.imprate.result['loss_per_sample']
        row['Seed'] = taskset.train.conf.get('seed', 0)
        row['\\#Epochs'] = ''
        row['Elapsed(Pilot)'] = ''
        if taskset.pilot is not None:
            row['\\#Epochs'] = str(taskset.pilot.result['i_epoch_best'] + 1) + ' / ' + \
                str(taskset.pilot.conf['n_epoch'])
            row['Elapsed(Pilot)'] = cls.shorten_time(taskset.pilot.result['elapsed'])
        row['Elapsed'] = cls.shorten_time(taskset.train.result['elapsed'])

        row_arch = collections.OrderedDict([('index', index_)])
        mo = taskset.train.conf['model']
        cls_, params_ = load_class(mo['cls_path']), mo['params']
        if issubclass(cls_, BasicBaseModel):
            params_ = cls_._resolve_seq_len(params_)
        for k, v in params_.items():
            if k == 'scaler_cls_path':
                row_arch['scaler_cls'] = cls.cls_path_to_name(v)
            elif k == 'scaler_params':
                row_arch[k] = cls.params_to_str(v)
            else:
                row_arch[k] = v

        bs = cls.cls_to_str(taskset.train.conf['batch_sampler'])
        row_opt = collections.OrderedDict([
            ('index', index_),
            ('criterion', cls.cls_to_str(taskset.train.conf['criterion'])),
            ('batch_sampler', re.sub('^BatchSampler', '', bs).replace('batch_size=', '')),
            ('optimizer', cls.cls_to_str(taskset.train.conf['optimizer'])),
            ('lr_scheduler', cls.cls_to_str(taskset.train.conf['lr_scheduler'])),
        ])
        return {'loss': row, 'arch': row_arch, 'opt': row_opt}

    @classmethod
    def get_taskset_rows(cls, tasksets):
        rows = {}
        for index_, taskset in tasksets.items():
            row = cls.get_taskset_row(taskset, index_=index_)
            if row is None:
                continue
            for k, v in row.items():
                if k not in rows:
                    rows[k] = []
                rows[k].append(v)
        return rows

    @classmethod
    def rows_to_df(cls, rows, how='inner'):
        assert how in ['inner', 'outer']
        if how == 'inner':
            keys = list(rows[0].keys())
            for row in rows[1:]:
                s = set(row.keys())
                keys = [k for k in keys if k in s]
        elif how == 'outer':
            keys = []
            seen = set()
            for row in rows:
                for k in row.keys():
                    if k not in seen:
                        keys.append(k)
                        seen.add(k)
        df = pd.DataFrame([{k: r.get(k) for k in keys} for r in rows])
        df = df.convert_dtypes(dtype_backend='numpy_nullable')
        return df.set_index('index').rename_axis(None)

    @classmethod
    def get_taskset_dfs(cls, tasksets, how='inner'):
        rows = cls.get_taskset_rows(tasksets)
        dfs = {k: cls.rows_to_df(v, how=how) for k, v in rows.items()}

        df = dfs['loss']
        s = df['MAE(Naive)']
        if np.isclose(s, s.iloc[0], rtol=1e-6).all():
            df = df.reset_index()
            new_row = pd.DataFrame([{'index': '', 'Model': '(Naive)', '\\#Params': 0}])
            if 'MSE(Naive)' in df.columns:
                new_row['MSE(Model)'] = df['MSE(Naive)'].iloc[0]
                new_row['MSE(ImpRate)'] = 0.0
            new_row['MAE(Model)'] = df['MAE(Naive)'].iloc[0]
            new_row['MAE(ImpRate)'] = 0.0
            new_row['ImpRate\\dag'] = 0.0
            new_row['Seed'] = df['Seed'].iloc[0]
            new_row['\\#Epochs'] = ''
            new_row['Elapsed(Pilot)'] = ''
            new_row['Elapsed'] = ''
            df = pd.concat([new_row, df], ignore_index=True)
            df.drop(columns=['MAE(Naive)', 'MSE(Naive)'], inplace=True, errors='ignore')
            df.rename(columns={
                'MAE(Model)': 'MAE',
                'MSE(Model)': 'MSE',
            }, inplace=True, errors='ignore')
            df = df.set_index('index').rename_axis(None)
            dfs['loss'] = df

        return dfs


def print_study(storage, study_name):
    study = optuna.load_study(storage=storage, study_name=study_name)
    print(f'\nstudy_name: {study.study_name}')
    if study.best_trial is not None:
        print(f'best_trial_number: {study.best_trial.number}')
        print(f'best_value: {study.best_value}')
        print(f'best_params: {study.best_params}')
        print(f'best_attrs: {study.best_trial.user_attrs}')
        
    print('trials:')
    for t in study.trials:
        print(
            f'{t.number}: '
            # f'state={t.state.name}, '
            f'value={t.value}, '
            f'params={t.params}'
        )
        print(f'{t.number}: attrs={t.user_attrs}')


def print_storage(storage):
    summaries = optuna.study.get_all_study_summaries(storage)
    for s in summaries:
        print_study(storage, s.study_name)


def print_storages(out_path):
    for sub_dir in Path(out_path).iterdir():
        if sub_dir.is_dir():
            storage = sub_dir / 'optuna.db'
            if storage.is_file():
                print_storage(f'sqlite:///{storage.as_posix()}')
