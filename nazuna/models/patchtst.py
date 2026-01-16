from abc import abstractmethod
from nazuna.models.base import BaseModel
from nazuna.scaler import IqrScaler
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, S, D]
        s = x.size(1)
        return self.dropout(x + self.pe[:s].unsqueeze(0))


class BasePatchTST(BaseModel):
    def _init_layers(self, seq_len, pred_len):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.patch_len = 8
        self.stride = 4
        self.d_model = 32
        self.n_heads = 2
        self.n_layers = 2
        self.d_ff = 128
        self.dropout = 0.1
        self.use_layernorm_patch = True
        self.pool = 'last'  # 'last' or 'mean'

        assert seq_len >= self.patch_len, 'seq_len >= patch_len'
        self.n_patches = 1 + (self.seq_len - self.patch_len) // self.stride

        self.patch_proj = nn.Linear(self.patch_len, self.d_model)
        self.patch_norm = (
            nn.LayerNorm(self.patch_len) if self.use_layernorm_patch else nn.Identity()
        )

        self.pos = PositionalEncoding(
            self.d_model, max_len=self.n_patches, dropout=self.dropout
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads, dim_feedforward=self.d_ff,
            dropout=self.dropout, batch_first=True, norm_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=self.n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(self.d_model, self.pred_len)

    def _patchify(self, x):  # x: [B, L, C] -> [B, C, P, patch_len]
        x = x.transpose(1, 2)  # [B, C, L]
        return x.unfold(dimension=2, size=self.patch_len, step=self.stride)

    def forward(self, x):
        B, L, C = x.shape
        patches = self._patchify(x)  # [B, C, P, pl]
        P = patches.size(2)
        patches = self.patch_norm(patches)  # [B, C, P, pl]

        z = patches.reshape(B * C, P, self.patch_len)  # [B*C, P, pl]
        z = self.patch_proj(z)  # [B*C, P, D]
        z = self.pos(z)  # [B*C, P, D]
        z = self.encoder(z)  # [B*C, P, D]

        if self.pool == 'last':
            token = z[:, -1, :]  # [B*C, D]
        else:
            token = z.mean(dim=1)  # [B*C, D]

        yhat = self.head(token)  # [B*C, pred_len]
        yhat = yhat.view(B, C, self.pred_len)  # [B, C, H]
        yhat = yhat.transpose(1, 2)  # [B, H, C]
        return yhat

    @abstractmethod
    def extract_input(self, batch):
        pass

    @abstractmethod
    def extract_target(self, batch):
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    def get_loss(self, batch, criterion):
        input_ = self.extract_input(batch)
        target = self.extract_target(batch)
        output = self(input_)
        loss = criterion(output, target)
        return loss, {}


class PatchTSTIqr(BasePatchTST):
    data_based_hyperparams = []

    def _init_layers(self, seq_len, pred_len, quantile_mode):
        super()._init_layers(seq_len, pred_len)
        self.quantile_mode = quantile_mode
        self.scaler = IqrScaler()

    def _get_quantiles(self, batch):
        if self.quantile_mode == 'full':
            quantiles = batch.quantiles_full
        elif self.quantile_mode == 'cum':
            quantiles = batch.quantiles_cum
        elif self.quantile_mode == 'rolling':
            quantiles = batch.quantiles_rolling
        else:
            raise ValueError(f'Unknown quantile_mode: {self.quantile_mode}')
        q1s_ = quantiles[:, 0, :]
        q2s_ = quantiles[:, 1, :]
        q3s_ = quantiles[:, 2, :]
        return q1s_, q2s_, q3s_

    def extract_input(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        return self.scaler.scale(
            batch.data[:, -self.seq_len:, :], q1s_=q1s_, q2s_=q2s_, q3s_=q3s_
        )

    def extract_target(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        return self.scaler.scale(
            batch.data_future[:, :self.pred_len], q1s_=q1s_, q2s_=q2s_, q3s_=q3s_
        )

    def predict(self, batch):
        q1s_, q2s_, q3s_ = self._get_quantiles(batch)
        input = self.extract_input(batch)
        output = self(input)
        return self.scaler.rescale(output, q1s_=q1s_, q2s_=q2s_, q3s_=q3s_)
