from nazuna.data_manager import TimeSeriesDataset
from nazuna.models.itransformer import iTransformer, DiffiTransformer
from nazuna.criteria import MSE


def test_forward(device, dummy_data):
    model = iTransformer.create(
        device=device,
        seq_len=16,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
    )
    batch = dummy_data((1, 16, 3))
    output, _ = model(batch)
    assert list(output.size()) == [1, 4, 3]


def test_get_loss(device, dummy_data):
    model = iTransformer.create(
        device=device,
        seq_len=16,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 16, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0


def test_diff_itransformer_forward(device, dummy_data):
    model = DiffiTransformer.create(
        device=device,
        seq_len=17,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
    )
    batch = dummy_data((1, 17, 3))
    output, _ = model(batch)
    assert list(output.size()) == [1, 4, 3]


def test_diff_itransformer_get_loss(device, dummy_data):
    model = DiffiTransformer.create(
        device=device,
        seq_len=17,
        pred_len=4,
        d_model=32,
        n_heads=4,
        d_ff=64,
        e_layers=2,
    )
    batch = TimeSeriesDataset.TimeSeriesBatch(
        tsta=None,
        tste=None,
        data=dummy_data((1, 17, 3)),
        tsta_future=None,
        tste_future=None,
        data_future=dummy_data((1, 4, 3)),
        stats={},
    )
    criterion = MSE.create(device, n_channel=3, pred_len=4)
    loss = model.get_loss(batch, criterion)
    assert loss.batch_mean.item() > 0.0
