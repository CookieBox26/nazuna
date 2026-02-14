from nazuna.scaler import IqrScaler
import torch
from collections import namedtuple


def test_iqr_scaler():
    scaler = IqrScaler('full', 'saved')
    DummyBatch = namedtuple('DummyBatch', ['quantiles'])
    quantiles = {
        'full': torch.tensor([[
            [0 - 0.5, 2.5 - 1.25, 5.0 - 2.5],
            [0, 2.5, 5],
            [0 + 0.5, 2.5 + 1.25, 5.0 + 2.5],
        ]]),
    }
    batch = DummyBatch(quantiles)
    x = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]])
    z = scaler.scale(x, batch)
    expected = torch.tensor([10., 3., 1.])
    assert torch.allclose(z[0][0], expected)
    z = scaler.rescale(z, batch)
    assert torch.allclose(z, x)
