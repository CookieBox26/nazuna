from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer, \
    MovingAverageDecomp, BatchSeriesNorm
from nazuna.models.autoformer import AutoCorrelation
import torch
import pytest


@pytest.mark.parametrize('l_q, l_kv', [
    (5, 5),
    (5, 7),
    (7, 5),
])
def test_multiheadattention(device, l_q, l_kv):
    mha = MultiheadAttention(d_model=16, n_heads=2)
    mha.to(device)
    q = torch.ones(8, l_q, 16, device=device)
    kv = torch.ones(8, l_kv, 16, device=device)
    x, attn_scores = mha(q, kv, kv)
    assert x.shape == (8, l_q, 16)
    assert attn_scores.shape == (8, 2, l_q, l_kv)


@pytest.mark.parametrize('self_attn_cls, norm_cls', [
    (AutoCorrelation, MovingAverageDecomp),
    (MultiheadAttention, BatchSeriesNorm),
    (MultiheadAttention, torch.nn.LayerNorm),
])
@pytest.mark.parametrize('norm_first', [False, True])
def test_transformerencoderlayer(device, self_attn_cls, norm_cls, norm_first):
    d_model = 16
    d_ff = 64

    self_attn = self_attn_cls(d_model=d_model, n_heads=2)
    if norm_cls is MovingAverageDecomp:
        norm_0 = MovingAverageDecomp(kernel_size=25, drop_moving_average=True)
        norm_1 = MovingAverageDecomp(kernel_size=25, drop_moving_average=True)
    else:
        norm_0 = norm_cls(d_model)
        norm_1 = norm_cls(d_model)
    activation = torch.nn.GELU()

    enc_layer = TransformerEncoderLayer(
        self_attn, d_model, d_ff, norm_0, norm_1, activation,
        norm_first=norm_first,
    )
    enc_layer.to(device)
    x = torch.ones(8, 5, 16, device=device)
    x, attn_scores = enc_layer(x)
    assert x.shape == (8, 5, 16)
    if self_attn_cls is AutoCorrelation:
        assert attn_scores.shape == (8, 5)
    else:
        assert attn_scores.shape == (8, 2, 5, 5)
