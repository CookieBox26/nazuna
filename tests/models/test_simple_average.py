from nazuna.models.simple_average import (
    SimpleAverage, SimpleDiffAverage,
    SimpleAverageVariableDecay, SimpleAverageVariableDecayChannelwise,
)
import torch


def test_simple_average(device):
    model = SimpleAverage.create(
        device=device,
        seq_len=4,
        pred_len=2,
        period_len=2,
    )

    x = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
    expected = torch.tensor([[
        [20., 20., 20.],
        [30., 30., 30.],
    ]], device=model.device)

    output, _ = model(x)
    assert torch.allclose(output, expected)


def test_simple_diff_average(device):
    # seq_len=5, so diff length = 4, period_len=2, n_period=2
    model = SimpleDiffAverage.create(
        device=device,
        seq_len=5,
        pred_len=2,
        period_len=2,
    )

    # x: 5 steps, 1 channel. Values: [10, 20, 30, 40, 50]
    # diff: [10, 10, 10, 10]
    # Reshaped into 2 periods of length 2: [[10,10],[10,10]]
    # Weighted average (decay_rate=1.0, equal weights): [10, 10]
    # last_val = 50
    # pred = 50 + cumsum([10, 10]) = [60, 70]
    x = torch.tensor([[[10.], [20.], [30.], [40.], [50.]]],
                      device=device)
    output, _ = model(x)
    expected = torch.tensor([[[60.], [70.]]], device=device)
    assert torch.allclose(output, expected)

    # With decay_rate=0.5: weights are [0.5^1, 0.5^0] = [0.5, 1.0]
    # normalized: [1/3, 2/3]
    # period 0: [10, 10], period 1: [10, 10]
    # avg_diff = 1/3*[10,10] + 2/3*[10,10] = [10, 10]
    # (all diffs are equal, so decay doesn't change the result here)
    model_decay = SimpleDiffAverage.create(
        device=device,
        seq_len=5,
        pred_len=2,
        period_len=2,
        decay_rate=0.5,
    )
    output_decay, _ = model_decay(x)
    assert torch.allclose(output_decay, expected)

    # Test with non-uniform diffs to verify decay weighting.
    # x: [0, 1, 2, 10, 20] -> diff: [1, 1, 8, 10]
    # periods: [[1, 1], [8, 10]]
    # decay_rate=0.5: w_raw = [0.5, 1.0], w = [1/3, 2/3]
    # avg_diff = 1/3*[1,1] + 2/3*[8,10] = [17/3, 21/3] = [5.6667, 7.0]
    # last_val = 20
    # pred = [20 + 17/3, 20 + 17/3 + 21/3] = [25.6667, 32.6667]
    x2 = torch.tensor([[[0.], [1.], [2.], [10.], [20.]]],
                       device=device)
    output2, _ = model_decay(x2)
    expected2 = torch.tensor(
        [[[20. + 17. / 3.], [20. + 17. / 3. + 21. / 3.]]],
        device=device,
    )
    assert torch.allclose(output2, expected2)


def test_simple_average_variable_decay(device):
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

    x = torch.tensor([[  # A batch with 3 roads and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
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


def test_simple_average_variable_decay_channelwise(device):
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

    x = torch.tensor([[  # A batch with 3 channels and 4 steps
        [10., 10., 10.],
        [20., 20., 20.],
        [30., 30., 30.],
        [40., 40., 40.],
    ]], device=model.device)
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
