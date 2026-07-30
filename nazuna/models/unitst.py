from nazuna.models._base import BasicBaseModel
from nazuna.models.common import BatchSeriesNorm, MultiheadAttention, Patchifier
import torch


class _DispatcherAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, dropout_aw=0.1):
        super().__init__()
        self.aggregate = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.distribute = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)

    def forward(self, x, dispatcher, prev_attn_scores=None):
        # x: (B, N, d_model), dispatcher: (B, k, d_model)
        prev_agg, prev_dist = \
            prev_attn_scores if prev_attn_scores is not None else (None, None)
        d, agg_scores = self.aggregate(  # (B, k, d_model)
            dispatcher, x, x, prev_attn_scores=prev_agg,
        )
        out, dist_scores = self.distribute(  # (B, N, d_model)
            x, d, d, prev_attn_scores=prev_dist,
        )
        return out, (agg_scores, dist_scores)


class _UniTSTBlock(torch.nn.Module):
    def __init__(
        self, d_model, n_heads, d_ff, dropout_aw=0.1, dropout_sa=0.1,
        dropout_ff=(0.0, 0.1), use_dispatcher=True, norm_first=False,
    ):
        super().__init__()
        self.use_dispatcher = use_dispatcher
        self.norm_first = norm_first
        if use_dispatcher:
            self.attn = _DispatcherAttention(d_model, n_heads, dropout_aw=dropout_aw)
        else:
            self.attn = MultiheadAttention(d_model, n_heads, dropout_aw=dropout_aw)
        self.dropout_sa = torch.nn.Dropout(dropout_sa)
        self.norm_0 = BatchSeriesNorm(d_model)
        self.ff = torch.nn.ModuleList([
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_ff[0]),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout_ff[1]),
        ])
        self.norm_1 = BatchSeriesNorm(d_model)

    def _forward_ff(self, x):
        x = self.ff[0](x)  # (..., d_ff)
        x = self.ff[1](x)
        x = self.ff[2](x)
        x_f1_debug = x.detach().clone()
        x = self.ff[3](x)  # (..., d_model)
        x = self.ff[4](x)
        return x, x_f1_debug

    def forward(self, x, dispatcher, prev_attn_scores=None):
        if not self.norm_first:
            x_save = x
            if self.use_dispatcher:
                x, attn_scores = self.attn(
                    x, dispatcher, prev_attn_scores=prev_attn_scores,
                )
            else:
                x, attn_scores = self.attn(x, x, x, prev_attn_scores=prev_attn_scores)
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
            if self.use_dispatcher:
                x, attn_scores = self.attn(
                    x, dispatcher, prev_attn_scores=prev_attn_scores,
                )
            else:
                x, attn_scores = self.attn(x, x, x, prev_attn_scores=prev_attn_scores)
            x = self.dropout_sa(x)
            x = x_save + x
            x_save = x
            x = self.norm_1(x)
            x, x_f1_debug = self._forward_ff(x)
            x = x_save + x
        return x, attn_scores, x_f1_debug


class UniTSTLike(BasicBaseModel):
    """
    !!! info "Original Research"
        This model is based on the following research:
        > Juncheng Liu, Chenghao Liu, Gerald Woo, Yiwei Wang, Bryan Hooi,
          Caiming Xiong, and Doyen Sahoo.
          "UniTST: Effectively Modeling Inter-Series and Intra-Series Dependencies
          for Multivariate Time Series Forecasting."
          Transactions on Machine Learning Research (TMLR), 2025.
          [Paper](https://openreview.net/forum?id=p3y5q4cvzV) |
          [arXiv](https://arxiv.org/abs/2406.04975)

        The official source code was not publicly available at the time of writing,
        so this implementation follows the description in the paper. Activation
        function and dropout placement are not specified in the paper and follow
        choices common in PatchTST-style models.
    """
    optimization_part_names = ['emb', 'pos', 'dispatcher', 'out', 'body']

    def _setup(
        self, seq_len: int, pred_len: int, c_in: int,
        patch_len: int = 16, stride: int = 8, padding_patch: str | None = 'end',
        d_model: int = 128, n_heads: int = 8, d_ff: int = 256, e_layers: int = 2,
        use_dispatcher: bool = True, n_dispatchers: int = 8,
        dispatcher_per_block: bool = False, dispatcher_init_normal: bool = False,
        dropout_emb: float = 0.1, dropout_aw: float = 0.1, dropout_sa: float = 0.1,
        dropout_ff: tuple[float, float] = (0.0, 0.2), res_attention: bool = False,
        norm_first: bool = False, norm_out: bool = True, z_scale: float = -1.0,
        w_scale: bool = False, w_scale_coef: float = 1.0,
        scaler_cls: type | None = None, scaler_params: dict | None = None,
        prep_type: str = 'none',
        use_revin: bool = True, revin_affine: bool = False, revin_eps: float = 1e-5,
        use_lc: bool = False,
        lc_end_epoch: int | None = None, lc_rate: float | None = None,
    ) -> None:
        assert seq_len >= patch_len >= stride, 'Expected seq_len >= patch_len >= stride'
        assert d_model % n_heads == 0, 'Expected d_model to be divisible by n_heads'
        assert d_model // n_heads >= 4, 'Expected head_dim >= 4'
        super()._setup(
            seq_len, pred_len, scaler_cls, scaler_params, prep_type=prep_type,
            use_revin=use_revin, revin_eps=revin_eps,
            revin_affine=revin_affine, c_in=c_in,
            use_lc=use_lc, lc_end_epoch=lc_end_epoch, lc_rate=lc_rate,
        )

        self.c_in = c_in
        self.patchifier = Patchifier(patch_len, stride, padding_patch)
        self.n_patches = self.patchifier.num_patches(seq_len)

        self.patch_proj = torch.nn.Linear(patch_len, d_model)
        # Learnable 2D positional encoding shared across the batch.
        self.pos_enc = torch.nn.Parameter(
            torch.empty(c_in, self.n_patches, d_model)
        )
        torch.nn.init.uniform_(self.pos_enc, -0.02, 0.02)
        self.dropout_emb = torch.nn.Dropout(dropout_emb)

        self.use_dispatcher = use_dispatcher
        self.dispatcher_per_block = dispatcher_per_block
        if use_dispatcher:
            # Learnable dispatcher embeddings shared across the batch. A leading
            # block axis is added when each block keeps its own dispatcher.
            if dispatcher_per_block:
                self.dispatcher = torch.nn.Parameter(
                    torch.empty(e_layers, n_dispatchers, d_model)
                )
            else:
                self.dispatcher = torch.nn.Parameter(
                    torch.empty(n_dispatchers, d_model)
                )
            if dispatcher_init_normal:
                torch.nn.init.normal_(self.dispatcher)
            else:
                torch.nn.init.uniform_(self.dispatcher, -0.02, 0.02)

        self.blocks = torch.nn.ModuleList([
            _UniTSTBlock(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                dropout_aw=dropout_aw, dropout_sa=dropout_sa, dropout_ff=dropout_ff,
                use_dispatcher=use_dispatcher, norm_first=norm_first,
            )
            for _ in range(e_layers)
        ])
        if w_scale:
            scale = 6 ** 0.5 * w_scale_coef
            with torch.no_grad():
                for block in self.blocks:
                    block.ff[0].weight.mul_(scale)
                    block.ff[3].weight.mul_(scale)
        self.res_attention = res_attention
        self.z_scale = z_scale

        self.out_proj = torch.nn.Linear(d_model * self.n_patches, self.pred_len)

        # Final normalization for the pre-norm variant, added last to keep the
        # random initialization of the other layers independent of norm_first.
        self.norm_out = \
            BatchSeriesNorm(d_model) if norm_first and norm_out else None

    def forward(self, x):
        B, L, C = x.shape
        patches = self.patchifier(x)  # (B, C, P, patch_len)
        P = patches.size(2)
        z = self.patch_proj(patches)  # (B, C, P, d_model)
        z = z + self.pos_enc.unsqueeze(0)  # broadcast (1, C, P, d_model)
        z = self.dropout_emb(z)
        z = z.reshape(B, C * P, -1)  # (B, C*P, d_model)

        scores = None
        for i, block in enumerate(self.blocks):
            if not self.use_dispatcher:
                dispatcher = None
            elif self.dispatcher_per_block:
                dispatcher = self.dispatcher[i].unsqueeze(0).expand(B, -1, -1)
            else:
                dispatcher = self.dispatcher.unsqueeze(0).expand(B, -1, -1)
            z, scores, x_f1_debug = block(z, dispatcher, (scores if self.res_attention else None))
            if i == len(self.blocks) - 1:
                self._debug_if_initial_stage(f'x_f1_shape = {tuple(x_f1_debug.shape)}')
                self._debug(f'x_f1_norm = {torch.linalg.vector_norm(x_f1_debug, dim=(1, 2)).mean().item()}')

        if self.norm_out is not None:
            z = self.norm_out(z)
        if self.z_scale > 0:
            z = z * self.z_scale

        self._debug_if_initial_stage(f'x_out_shape = {tuple(z.shape)}')
        self._debug(f'x_out_norm = {torch.linalg.vector_norm(z, dim=(1, 2)).mean().item()}')
        self._finish_initial_debug_stage()

        z = z.reshape(B, C, P, -1)  # (B, C, P, d_model)
        z = z.reshape(B, C, -1)  # (B, C, P * d_model)
        y = self.out_proj(z)  # (B, C, pred_len)
        y = y.transpose(1, 2)  # (B, pred_len, C)
        return y, {}

    def set_optimizers(self, optimizer_groups):
        if set(optimizer_groups.groups) == {'model'}:
            super().set_optimizers(optimizer_groups)
            return
        named_params = [
            ('emb', list(self.patch_proj.parameters())),
            ('pos', [self.pos_enc]),
        ]
        if self.use_dispatcher:
            named_params.append(('dispatcher', [self.dispatcher]))
        else:
            del optimizer_groups.groups['dispatcher']
        named_params.append(('out', list(self.out_proj.parameters())))
        grouped_ids = {id(p) for _, params in named_params for p in params}
        body_params = [p for p in self.parameters() if id(p) not in grouped_ids]
        named_params.append(('body', body_params))
        for name, params in named_params:
            optimizer_groups.set_optimizer(
                name, (p for p in params if p.requires_grad),
            )

    # Part boundaries are hardcoded; 'other' is the catch-all.
    @staticmethod
    def _part_of(key):
        if key.startswith('patch_proj.'):
            return 'emb'
        if key == 'pos_enc':
            return 'pos'
        if key == 'dispatcher':
            return 'dispatcher'
        if key.startswith('out_proj.'):
            return 'out'
        if key.startswith('revin.'):
            return 'revin'
        if key.startswith('blocks.'):
            _, i, sub, *rest = key.split('.')
            if sub == 'attn' and rest and rest[0] in ('aggregate', 'distribute'):
                return f'block{i}.attn.{rest[0]}'
            return f'block{i}.{sub}'
        return 'other'

    @classmethod
    def calc_dists(cls, state_path_0, state_path_1):
        state_0 = torch.load(state_path_0, map_location='cpu')
        state_1 = torch.load(state_path_1, map_location='cpu')

        # BatchNorm running stats are buffers and are excluded.
        buffer_suffixes = ('.running_mean', '.running_var', '.num_batches_tracked')
        diffs = {}
        for key, tensor_0 in state_0.items():
            if key.endswith(buffer_suffixes):
                continue
            diff = (tensor_0 - state_1[key]).reshape(-1)
            diffs.setdefault(cls._part_of(key), []).append(diff)

        result = {}
        for name, parts in diffs.items():
            flat = torch.cat(parts)
            result[name] = {
                'num_params': flat.numel(),
                'dist_l2': flat.norm().item(),
            }
        return result

    @classmethod
    def get_part_params(cls, state_path, part_name):
        state = torch.load(state_path, map_location='cpu')
        buffer_suffixes = ('.running_mean', '.running_var', '.num_batches_tracked')
        return {
            key: tensor
            for key, tensor in state.items()
            if not key.endswith(buffer_suffixes) and cls._part_of(key) == part_name
        }
