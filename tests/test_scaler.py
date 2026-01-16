from nazuna.scaler import IqrScaler
import torch


def test_iqr_scaler():
    scaler = IqrScaler()
    q1s_ = torch.tensor([0 - 0.5, 2.5 - 1.25, 5.0 - 2.5])
    q2s_ = torch.tensor([0, 2.5, 5])
    q3s_ = torch.tensor([0 + 0.5, 2.5 + 1.25, 5.0 + 2.5])
    batch_0 = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]])
    batch_1 = scaler.scale(batch_0, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)
    expected = torch.tensor([10., 3., 1.])
    assert torch.allclose(batch_1[0][0], expected)
    batch_1 = scaler.rescale(batch_1, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)
    assert torch.allclose(batch_1, batch_0)
