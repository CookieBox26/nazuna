import collections
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    TimeSeriesBatch = collections.namedtuple('TimeSeriesBatch', [
        'tsta', 'tste', 'data', 'tsta_future', 'tste_future', 'data_future',
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def to_tensor(cls, x):
        return torch.tensor(x, dtype=torch.float32, device=cls.device)

    def __init__(self, df, seq_len, pred_len, cols_org):
        self.df = df
        self.tsta = list(self.df['timestamp'].values)
        self.tste = list(self.df['timestep'].values)
        del self.df['timestamp'], self.df['timestep']
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cols_org = cols_org
        self.n_sample = len(df) - (seq_len - 1) - pred_len
        self.n_feats = len(self.df.columns)
        self.means = []
        self.stds = []
        self.q1s = []
        self.q2s = []
        self.q3s = []
        for col in self.df.columns:
            self.means.append(self.df[col].mean())
            self.stds.append(self.df[col].std())
            qtiles = list(self.df[col].quantile([0.25, 0.5, 0.7]))
            self.q1s.append(qtiles[0])
            self.q2s.append(qtiles[1])
            self.q3s.append(qtiles[2])

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx_0):
        # idx_0 :  Start of the reference window
        # idx_0 + seq_len - 1 :  End of the reference window (current time) (= idx_1 - 1)
        # idx_0 + seq_len - 1 + 1 :  Start of the prediction window (= idx_1)
        # idx_0 + seq_len - 1 + pred_len :  End of the prediction window (= idx_2 - 1)
        idx_1 = idx_0 + self.seq_len
        idx_2 = idx_0 + self.seq_len + self.pred_len
        return TimeSeriesDataset.TimeSeriesBatch(
            self.tsta[idx_0:idx_1], self.tste[idx_0:idx_1], self.df.iloc[idx_0:idx_1, :].values,
            self.tsta[idx_1:idx_2], self.tste[idx_1:idx_2], self.df.iloc[idx_1:idx_2, :].values,
        )

    @staticmethod
    def collate_fn(batch):
        to_tensor = TimeSeriesDataset.to_tensor
        return TimeSeriesDataset.TimeSeriesBatch(
            np.array([v[0] for v in batch]),  # batch_size, seq_len
            to_tensor(np.array([v[1] for v in batch])),  # batch_size, seq_len
            to_tensor(np.array([v[2] for v in batch])),  # batch_size, seq_len, n_channel
            np.array([v[3] for v in batch]),  # batch_size, pred_len
            to_tensor(np.array([v[4] for v in batch])),  # batch_size, pred_len
            to_tensor(np.array([v[5] for v in batch])),  # batch_size, pred_len, n_channel
        )

    def get_data_loader(self, batch_sampler_cls, batch_sampler_kwargs):
        batch_sampler = batch_sampler_cls(self.n_sample, **batch_sampler_kwargs)
        return DataLoader(self, batch_sampler=batch_sampler, collate_fn=self.collate_fn)


class TimeSeriesDataManager:
    def __init__(
        self,
        path,
        colname_timestamp,
        seq_len,
        pred_len,
        white_list=None,
        step_start=0,
        step_width=1,
    ):
        self.path = path
        self.colname_timestamp = colname_timestamp
        if isinstance(white_list, str) and (white_list != ''):
            self.white_list = white_list.split(',')
        elif isinstance(white_list, list):
            self.white_list = [str(w) for w in white_list]
        else:
            self.white_list = ''
        self.step_start = step_start
        self.step_width = step_width
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.load_data()

    def load_data(self):
        df = pd.read_csv(self.path)
        df.columns = [str(col) for col in df.columns]
        cols = []
        if self.white_list != '':
            cols += self.white_list
        else:
            cols += [col for col in df.columns if col != self.colname_timestamp]
        self.cols_org = cols
        df = df.loc[:, [self.colname_timestamp] + cols]
        self.n_channel = len(df.columns) - 1
        df.columns = ['timestamp'] + [f'y{i}' for i in range(self.n_channel)]
        n_rows = len(df)
        df.insert(1, 'timestep', [self.step_start + i * self.step_width for i in range(n_rows)])
        self.df = df

    def extract_data(self, data_range):
        n_rows = len(self.df)
        n_front = n_rows - (self.seq_len - 1) - self.pred_len
        i_start = int(n_front * data_range[0])
        i_end = int(n_front * data_range[1]) + (self.seq_len - 1) + self.pred_len
        return self.df.iloc[i_start:i_end, :]

    def get_data_loader(self, data_range, batch_sampler_cls, batch_sampler_kwargs):
        df_target = self.extract_data(data_range)
        dataset = TimeSeriesDataset(df_target, self.seq_len, self.pred_len, self.cols_org)
        return dataset.get_data_loader(batch_sampler_cls, batch_sampler_kwargs)
