import torch
import math
import torch.nn.functional as F


class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_aw=0.1):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.dropout_aw = torch.nn.Dropout(dropout_aw)

    def forward(self, q, k, v, prev_attn_scores=None):
        B, Lq, _ = q.shape  # (B, Lq, d_m)
        Lk = k.size(1)
        H, d_h = self.n_heads, self.d_head
        q = self.q_proj(q).view(B, Lq, H, d_h).transpose(1, 2)  # (B, H, Lq, d_h)
        k = self.k_proj(k).view(B, Lk, H, d_h).transpose(1, 2)  # (B, H, Lk, d_h)
        v = self.v_proj(v).view(B, Lk, H, d_h).transpose(1, 2)  # (B, H, Lk, d_h)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        if prev_attn_scores is not None:
            attn_scores = attn_scores + prev_attn_scores
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_aw(attn_weights)
        output = torch.matmul(attn_weights, v)  # [B, H, Lq, d_h]
        output = output.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        output = self.out_proj(output)
        return output, attn_scores


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self, self_attn, d_model, d_ff, norm_0, norm_1,
        activation, dropout_sa=0.1, dropout_ff=(0.0, 0.1),
        norm_first=False, bias=True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.dropout_sa = torch.nn.Dropout(dropout_sa)
        self.ff = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_ff, bias=bias),
            activation,
            torch.nn.Dropout(dropout_ff[0]),
            torch.nn.Linear(d_ff, d_model, bias=bias),
            torch.nn.Dropout(dropout_ff[1]),
        ])
        self.norm_0 = norm_0
        self.norm_1 = norm_1
        self.norm_first = norm_first

    def _forward_ff(self, x):
        x = self.ff[0](x)  # (..., d_ff)
        x = self.ff[1](x)
        x = self.ff[2](x)
        x_f1_debug = x.detach().clone()
        x = self.ff[3](x)  # (..., d_model)
        x = self.ff[4](x)
        return x, x_f1_debug

    def forward(self, x, prev_attn_scores=None):
        if not self.norm_first:
            x_save = x
            x, attn_scores = self.self_attn(x, x, x, prev_attn_scores=prev_attn_scores)
            x = self.dropout_sa(x)
            x = x_save + x
            x = self.norm_0(x)
            x_save = x
            x, x_f1_debug = self._forward_ff(x)
            x = x_save + x
            x = self.norm_1(x)
        else:
            x_save = x
            x = self.norm_0(x)
            x, attn_scores = self.self_attn(x, x, x, prev_attn_scores=prev_attn_scores)
            x = self.dropout_sa(x)
            x = x_save + x
            x_save = x
            x = self.norm_1(x)
            x, x_f1_debug = self._forward_ff(x)
            x = x_save + x
        return x, attn_scores, x_f1_debug
