import torch
import pytest


@pytest.fixture
def dummy_data(device):
    def _cycle_1_n_3(n):
        # Values repeat the [10, 20, 30, 40] cycle along the time axis.
        base = [[10., 10., 10.], [20., 20., 20.], [30., 30., 30.], [40., 40., 40.]]
        rows = [base[i % 4] for i in range(n)]
        return torch.tensor([rows], device=device)
    def _dummy_data_1_2_3():
        return torch.tensor([[
            [10., 10., 10.], [20., 20., 20.],
        ]], device=device)
    def _dummy_data_1_4_3():
        return _cycle_1_n_3(4)
    def _dummy_data_1_16_3():
        return _cycle_1_n_3(16)
    def _dummy_data_1_17_3():
        return torch.tensor([[
            [10., 10., 10.], [20., 20., 20.], [30., 30., 30.], [40., 40., 40.],
            [10., 10., 10.], [20., 20., 20.], [30., 30., 30.], [40., 40., 40.],
            [10., 10., 10.], [20., 20., 20.], [30., 30., 30.], [40., 40., 40.],
            [10., 10., 10.], [20., 20., 20.], [30., 30., 30.], [40., 40., 40.],
            [10., 10., 10.],
        ]], device=device)
    def _dummy_data_2_16_3():
        single = _cycle_1_n_3(16)
        return single.repeat(2, 1, 1)
    def _dummy_data_2_96_3():
        single = _cycle_1_n_3(96)
        return single.repeat(2, 1, 1)
    def _dummy_data(shape):
        if shape == (1, 2, 3):
            return _dummy_data_1_2_3()
        if shape == (1, 4, 3):
            return _dummy_data_1_4_3()
        if shape == (1, 16, 3):
            return _dummy_data_1_16_3()
        if shape == (1, 17, 3):
            return _dummy_data_1_17_3()
        if shape == (2, 16, 3):
            return _dummy_data_2_16_3()
        if shape == (2, 96, 3):
            return _dummy_data_2_96_3()
    return _dummy_data
