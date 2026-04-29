from nazuna.models.common import \
    MultiheadAttention, TransformerEncoderLayer, BatchSeriesNorm
import torch
import pytest


def test_multiheadattention(device):
    mha = MultiheadAttention(d_model=16, n_heads=2)
    mha.to(device)
    x = torch.ones(8, 5, 16, device=device)
    x, attn_scores = mha(x)
    assert x.shape == (8, 5, 16)
    assert attn_scores.shape == (8, 2, 5, 5)


@pytest.mark.parametrize('norm_cls', [torch.nn.LayerNorm, BatchSeriesNorm])
@pytest.mark.parametrize('norm_first', [False, True])
def test_transformerencoderlayer(device, norm_cls, norm_first):
    d_model = 16
    d_ff = 64
    self_attn = MultiheadAttention(d_model, n_heads=2)
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
    assert attn_scores.shape == (8, 2, 5, 5)
