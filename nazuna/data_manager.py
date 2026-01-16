import collections
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    TimeSeriesBatch = collections.namedtuple('TimeSeriesBatch', [
        'tsta', 'tste', 'data', 'tsta_future', 'tste_future', 'data_future',
        'quantiles_full', 'quantiles_cum', 'quantiles_rolling',
    ])

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def get_quantiles(self, df_):
        quantiles = [[], [], []]
        for col in self.df_org.columns:
            qtiles = list(df_[col].quantile([0.25, 0.5, 0.75]))
            for i_q in range(3):
                quantiles[i_q].append(qtiles[i_q])
        return quantiles

    def __init__(self, df, seq_len, pred_len, offset, rolling_window, device):
        self.device = device
        self.df_org = df
        self.tsta_org = list(df['timestamp'].values)
        self.tste_org = list(df['timestep'].values)
        del self.df_org['timestamp'], self.df_org['timestep']

        self.offset = offset
        self.rolling_window = rolling_window
        self.df = self.df_org.iloc[self.offset:, :]
        self.tsta = self.tsta_org[self.offset:]
        self.tste = self.tste_org[self.offset:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_sample = len(self.df) - (seq_len - 1) - pred_len
        self.n_channel = len(self.df.columns)
        self.quantiles_full = self.get_quantiles(self.df_org)

    def __len__(self):
        return self.n_sample

    def __getitem__(self, idx_0):
        # idx_0 :  Start of the reference window
        # idx_0 + seq_len - 1 :  End of the reference window (current time) (= idx_1 - 1)
        # idx_0 + seq_len - 1 + 1 :  Start of the prediction window (= idx_1)
        # idx_0 + seq_len - 1 + pred_len :  End of the prediction window (= idx_2 - 1)
        idx_1 = idx_0 + self.seq_len
        idx_2 = idx_0 + self.seq_len + self.pred_len

        quantiles_cum = self.get_quantiles(self.df_org[:(self.offset + idx_1)])
        quantiles_rolling = self.get_quantiles(
            self.df_org[(self.offset + idx_1 - self.rolling_window):(self.offset + idx_1)]
        )

        return TimeSeriesDataset.TimeSeriesBatch(
            self.tsta[idx_0:idx_1], self.tste[idx_0:idx_1], self.df.iloc[idx_0:idx_1, :].values,
            self.tsta[idx_1:idx_2], self.tste[idx_1:idx_2], self.df.iloc[idx_1:idx_2, :].values,
            self.quantiles_full, quantiles_cum, quantiles_rolling,
        )

    def collate_fn(self, batch):
        to_tensor = self.to_tensor
        return TimeSeriesDataset.TimeSeriesBatch(
            np.array([b[0] for b in batch]),  # batch_size, seq_len
            to_tensor(np.array([b[1] for b in batch])),  # batch_size, seq_len
            to_tensor(np.array([b[2] for b in batch])),  # batch_size, seq_len, n_channel
            np.array([b[3] for b in batch]),  # batch_size, pred_len
            to_tensor(np.array([b[4] for b in batch])),  # batch_size, pred_len
            to_tensor(np.array([b[5] for b in batch])),  # batch_size, pred_len, n_channel
            to_tensor(np.array([b[6] for b in batch])),
            to_tensor(np.array([b[7] for b in batch])),
            to_tensor(np.array([b[8] for b in batch])),
        )

    def get_data_loader(self, batch_sampler_cls, batch_sampler_kwargs):
        batch_sampler = batch_sampler_cls(self.n_sample, **batch_sampler_kwargs)
        return DataLoader(self, batch_sampler=batch_sampler, collate_fn=self.collate_fn)


class TimeSeriesDataManager:
    """
    Manager for time-series data that handles loading and splitting.

    Note on seq_len and pred_len:
        These parameters affect data splitting. When you specify a data range
        (e.g., 0.0-0.5 for training), the actual row indices depend on seq_len
        and pred_len. The actual sequence lengths used during evaluation or
        training can be shorter (specified in the criterion and model).
        E.g., for fair comparison across experiments with varying input lengths,
        set seq_len to the maximum value you plan to use.
    """
    def __init__(
        self,
        path: str,
        colname_timestamp: str,
        seq_len: int,
        pred_len: int,
        white_list: str | list[str] | None = None,
        step_start: int = 0,
        step_width: int = 1,
    ):
        """
        Args:
            path: Path to the CSV file.
            colname_timestamp: Column name for timestamps.
            seq_len: Input sequence length (affects data splitting).
            pred_len: Prediction sequence length (affects data splitting).
            white_list: Column names to use. If None, all columns are used.
        """
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

    def _extract_data(self, data_range):
        n_rows = len(self.df)
        n_front = n_rows - (self.seq_len - 1) - self.pred_len
        i_start = int(n_front * data_range[0])
        i_end = int(n_front * data_range[1]) + (self.seq_len - 1) + self.pred_len
        return self.df.iloc[i_start:i_end, :]

    def get_data_loader(
        self,
        data_range,
        batch_sampler_cls,
        batch_sampler_kwargs,
        offset,
        rolling_window,
        device,
    ):
        df_target = self._extract_data(data_range)
        dataset = TimeSeriesDataset(
            df_target, self.seq_len, self.pred_len, offset, rolling_window, device,
        )
        return dataset.get_data_loader(batch_sampler_cls, batch_sampler_kwargs)
