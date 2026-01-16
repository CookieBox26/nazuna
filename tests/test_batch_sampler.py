import pytest
from nazuna.batch_sampler import (
    BatchSampler, BatchSamplerShuffle, BatchSamplerBatchShuffle, BatchSamplerPeriodic,
)
from torch.utils.data import DataLoader
import pandas as pd
import torch


class Dataset:
    def __init__(self, df):
        self.df = df
        self.n_sample = len(df)
    def __getitem__(self, batch_idx):
        return self.df.loc[batch_idx, :].values
    def __len__(self):
        return self.n_sample


@pytest.fixture
def dataset():
    df = pd.DataFrame({
        'a': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        'b': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        'c': [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    })
    return Dataset(df)


@pytest.fixture
def batch_expected():
    return [
        torch.tensor([[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]]),
        torch.tensor([[14, 24, 34], [15, 25, 35], [16, 26, 36], [17, 27, 37]]),
        torch.tensor([[18, 28, 38], [19, 29, 39], [20, 30, 40]]),
    ]


def test_batch_sampler(dataset, batch_expected):
    batch_sampler = BatchSampler.create(dataset.n_sample, batch_size=4)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    assert dataloader.batch_sampler.n_batch == 3

    for i_batch, batch in enumerate(dataloader):  # 1st epoch
        assert torch.equal(batch, batch_expected[i_batch])

    for i_batch, batch in enumerate(dataloader):  # 2nd epoch
        assert torch.equal(batch, batch_expected[i_batch])


def test_batch_sampler_shuffle(dataset):
    batch_sampler = BatchSamplerShuffle.create(dataset.n_sample, batch_size=4)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    assert dataloader.batch_sampler.n_batch == 3

    data_1 = None
    for i_batch, batch in enumerate(dataloader):  # 1st epoch
        assert batch.shape == ((4, 3) if i_batch != 2 else (3, 3))
        data_1 = batch if (data_1 is None) else torch.cat((data_1, batch))

    data_2 = None
    for i_batch, batch in enumerate(dataloader):  # 2nd epoch
        assert batch.shape == ((4, 3) if i_batch != 2 else (3, 3))
        data_2 = batch if (data_2 is None) else torch.cat((data_2, batch))

    assert not torch.equal(data_1[0], data_2[0])

    data_0 = sorted([tuple(row) for _, row in dataset.df.iterrows()])
    data_1 = sorted([tuple(row.tolist()) for row in data_1])
    data_2 = sorted([tuple(row.tolist()) for row in data_2])
    assert data_0 == data_1
    assert data_0 == data_2


def test_batch_sampler_batch_shuffle(dataset, batch_expected):
    batch_sampler = BatchSamplerBatchShuffle.create(dataset.n_sample, batch_size=4, seed=1)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    assert dataloader.batch_sampler.n_batch == 3

    for i_batch, batch in enumerate(dataloader):  # 1st epoch
        assert any([torch.equal(batch, batch_expected[j_batch]) for j_batch in range(3)])
        if i_batch == 0:
            batch_0_1 = batch.detach().clone()

    for i_batch, batch in enumerate(dataloader):  # 2nd epoch
        assert any([torch.equal(batch, batch_expected[j_batch]) for j_batch in range(3)])
        if i_batch == 0:
            batch_0_2 = batch.detach().clone()

    assert not torch.equal(batch_0_1, batch_0_2)


def test_batch_sampler_periodic(dataset):
    batch_sampler = BatchSamplerPeriodic.create(dataset.n_sample, batch_size=4, period=5)
    dataloader = DataLoader(dataset=dataset, batch_sampler=batch_sampler)
    assert dataloader.batch_sampler.n_batch == 5

    batch_expected = [
        torch.tensor([[10, 20, 30], [15, 25, 35], [20, 30, 40]]),
        torch.tensor([[11, 21, 31], [16, 26, 36]]),
        torch.tensor([[12, 22, 32], [17, 27, 37]]),
        torch.tensor([[13, 23, 33], [18, 28, 38]]),
        torch.tensor([[14, 24, 34], [19, 29, 39]]),
    ]
    for i_batch, batch in enumerate(dataloader):  # 1st epoch
        assert any([torch.equal(batch, batch_expected[j_batch]) for j_batch in range(5)])
        if i_batch == 0:
            batch_0_1 = batch.detach().clone()

    for i_batch, batch in enumerate(dataloader):  # 2nd epoch
        assert any([torch.equal(batch, batch_expected[j_batch]) for j_batch in range(5)])
        if i_batch == 0:
            batch_0_2 = batch.detach().clone()

    assert not torch.equal(batch_0_1, batch_0_2)
