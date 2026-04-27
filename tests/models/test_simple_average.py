from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.simple_average import (
    SimpleAverage,
    SimpleAverageVariableDecay, SimpleAverageVariableDecayChannelwise,
)
import torch


def test_simple_average(device, dummy_data):
    model = SimpleAverage.create(
        device=device,
        seq_len=4,
        pred_len=2,
        period_len=2,
    )

    x = dummy_data((1, 4, 3))
    expected = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)

    output, _ = model(x)
    assert torch.allclose(output, expected)

    # prep_type='diff': data length 5, diffs of [10,20,30,40,50] = [10,10,10,10],
    # periods avg = [10,10], cumsum + last_val (50) = [60, 70]
    model_diff = SimpleAverage.create(
        device=device, seq_len=4, pred_len=2, period_len=2, prep_type='diff',
    )
    data = torch.tensor([[[10.], [20.], [30.], [40.], [50.]]], device=device)
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None, tste=None, data=data,
        tsta_future=None, tste_future=None, data_future=None, stats=None,
    )
    output_diff, _ = model_diff.predict(batch)
    expected_diff = torch.tensor([[[60.], [70.]]], device=device)
    assert torch.allclose(output_diff, expected_diff)

    # prep_type='diff' with non-uniform diffs and decay_rate=0.5 to verify decay weighting.
    # x: [0, 1, 2, 10, 20] -> diff: [1, 1, 8, 10]
    # periods: [[1, 1], [8, 10]]
    # decay_rate=0.5: w_raw = [0.5, 1.0], w = [1/3, 2/3]
    # avg_diff = 1/3*[1,1] + 2/3*[8,10] = [17/3, 21/3]
    # last_val = 20, pred = [20 + 17/3, 20 + 17/3 + 21/3]
    model_diff_decay = SimpleAverage.create(
        device=device, seq_len=4, pred_len=2, period_len=2,
        decay_rate=0.5, prep_type='diff',
    )
    data2 = torch.tensor([[[0.], [1.], [2.], [10.], [20.]]], device=device)
    batch2 = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None, tste=None, data=data2,
        tsta_future=None, tste_future=None, data_future=None, stats=None,
    )
    output_diff_decay, _ = model_diff_decay.predict(batch2)
    expected_diff_decay = torch.tensor(
        [[[20. + 17. / 3.], [20. + 17. / 3. + 21. / 3.]]], device=device,
    )
    assert torch.allclose(output_diff_decay, expected_diff_decay)


def test_simple_average_variable_decay(device, dummy_data):
    model = SimpleAverageVariableDecay.create(
        device=device,
        seq_len=4,
        pred_len=2,
        period_len=2,
    )
    params = dict(model.named_parameters())
    assert 'decay_rate' in params
    assert params['decay_rate'].requires_grad is True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    x = dummy_data((1, 4, 3))
    true = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)

    # Test that backward computes a gradient.
    output, _ = model(x)
    loss = ((output - true) ** 2).mean()
    loss.backward()
    assert model.decay_rate.grad is not None

    # Test that optimizer.step() updates the value.
    before = model.decay_rate.detach().clone()
    optimizer.step()
    after = model.decay_rate.detach()
    assert not torch.allclose(before, after)


def test_simple_average_variable_decay_channelwise(device, dummy_data):
    n_channel = 3
    model = SimpleAverageVariableDecayChannelwise.create(
        device=device,
        seq_len=4,
        pred_len=2,
        period_len=2,
        n_channel=n_channel,
    )
    params = dict(model.named_parameters())
    assert 'decay_rate' in params
    assert params['decay_rate'].shape == (n_channel,)
    assert params['decay_rate'].requires_grad is True

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()

    x = dummy_data((1, 4, 3))
    true = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)

    # Test that backward computes a gradient.
    output, _ = model(x)
    loss = ((output - true) ** 2).mean()
    loss.backward()
    assert model.decay_rate.grad is not None
    assert model.decay_rate.grad.shape == (n_channel,)

    # Test that optimizer.step() updates the value.
    before = model.decay_rate.detach().clone()
    optimizer.step()
    after = model.decay_rate.detach()
    assert not torch.allclose(before, after)
