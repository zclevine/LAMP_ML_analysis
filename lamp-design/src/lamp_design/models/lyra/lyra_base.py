import numpy as np
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# from flashfftconv import FlashDepthWiseConv1d
from einops import rearrange
#from . import s4d 
from .s4d import S4D
from .rmsnorm import RMSNorm


# ---------------------------------------------------------------------------
# SSM backbone wrappers — normalise every backend to (B, L, D) → (B, L, D)
# ---------------------------------------------------------------------------

class _GRUBackbone(nn.Module):
    """Bidirectionality is handled at the Lyra level; this is a unidirectional GRU."""
    def __init__(self, d_model):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out


class _ConvBackbone(nn.Module):
    """Stack of 1-D convolutions (no recurrence at all)."""
    def __init__(self, d_model, n_layers=2, kernel_size=7, dropout=0.2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Conv1d(d_model, d_model, kernel_size,
                          padding=kernel_size // 2, groups=1),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class _DeltaNetBackbone(nn.Module):
    """Wrap fla.layers.GatedDeltaNet to return a plain tensor.

    fla only supports mode="chunk" for training. The chunk backward kernel's
    shared-memory use scales with the value head dim, and the default
    expand_v=2 overflows this GPU's shared memory (~163KB). We set expand_v=1
    (no value expansion) so the kernel fits; this also keeps the param count
    closer to the other backbones.
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        from fla.layers import GatedDeltaNet
        self.deltanet = GatedDeltaNet(
            hidden_size=d_model, num_heads=num_heads,
            expand_v=1, mode="chunk",
        )

    def forward(self, x):
        out = self.deltanet(x)
        return out[0] if isinstance(out, tuple) else out


def _build_ssm(ssm: str, d_model: int, d_state: int = 64,
               dropout: float = 0.2, transposed: bool = False,
               **kernel_args) -> nn.Module:
    """Factory: return an SSM backbone whose forward is (B,L,D)->(B,L,D)."""

    if ssm == "s4d":
        return S4D(d_model, d_state=d_state, dropout=dropout,
                   transposed=transposed, **kernel_args)

    if ssm == "mamba":
        from mamba_ssm import Mamba
        return Mamba(d_model, d_state=d_state, d_conv=4, expand=2)

    if ssm == "mamba2":
        from mamba_ssm import Mamba2
        d_ssm = 2 * d_model
        headdim = 64
        while headdim > d_ssm or d_ssm % headdim != 0:
            headdim //= 2
        return Mamba2(d_model, d_state=d_state, d_conv=4,
                      expand=2, headdim=headdim)

    if ssm == "mamba3":
        from mamba_ssm.modules.mamba3 import Mamba3
        d_ssm = 2 * d_model
        headdim = 64
        while headdim > d_ssm or d_ssm % headdim != 0:
            headdim //= 2
        return Mamba3(d_model, d_state=d_state, headdim=headdim)

    if ssm == "gru":
        return _GRUBackbone(d_model)

    if ssm == "conv":
        return _ConvBackbone(d_model, dropout=dropout)

    if ssm == "deltanet":
        num_heads = max(1, d_model // 16)
        return _DeltaNetBackbone(d_model, num_heads=num_heads)

    raise ValueError(f"Unknown SSM backend: {ssm!r}. "
                     f"Choose from: s4d, mamba, mamba2, mamba3, gru, conv, deltanet")

dropout_fn = nn.Dropout1d
#torch.backends.cudnn.benchmark = True
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example configuration for Lyra, as used in our protein datasets
# usage: 
#       lyra_instance = Lyra(**lyra_example_proteins_config)
lyra_example_proteins_config = {
    "d_input": 20,
    "d_output": 1,
    "d_model": 64,
    "dropout": 0.2,
}

class PGC(nn.Module):
    def __init__(self,d_model,expansion_factor = 1.0,dropout = 0.0, kernel_size=3,
                 extra_kernel_sizes=None):
        super().__init__()

        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        expanded_dim = int(d_model * expansion_factor)

        self.conv = nn.Conv1d(expanded_dim,
                              expanded_dim,
                              kernel_size=kernel_size,
                              padding= kernel_size//2,
                              groups=expanded_dim)

        # Optional extra parallel depthwise kernels (multi-scale conv). Their
        # outputs are summed with self.conv before gating. Kept in a separate
        # ModuleList so the single-kernel default leaves `self.conv` as the only
        # conv module -> state_dict stays identical and old checkpoints load.
        self.extra_convs = nn.ModuleList([
            nn.Conv1d(expanded_dim, expanded_dim, kernel_size=ks,
                      padding=ks // 2, groups=expanded_dim)
            for ks in (extra_kernel_sizes or [])
        ])

        self.in_proj = nn.Linear(d_model, int(d_model * expansion_factor * 2))
        self.out_norm = nn.RMSNorm(int(d_model), eps=1e-8)
        self.in_norm = nn.RMSNorm(expanded_dim * 2, eps=1e-8)
        self.out_proj = nn.Linear(expanded_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # Input projection and normalization
        xv = self.in_norm(self.in_proj(u))

        # Split projected input into two parts: x and v
        x, v = xv.chunk(2, dim=-1)

        # Depthwise convolution on x (+ any parallel multi-scale kernels)
        xt = x.transpose(-1, -2)
        x_conv = self.conv(xt)
        for c in self.extra_convs:
            x_conv = x_conv + c(xt)
        x_conv = x_conv.transpose(-1, -2)

        # Gating mechanism
        gate = v * x_conv

        # Output projection and normalization
        x_out = self.out_norm(self.out_proj(gate))

        return x_out



# class PGC(nn.Module):
#     """
#     Parallel Gated Convolution module with FFT-based convolution.
#     This module projects the input, applies a gated FFT convolution,
#     and projects back to the original dimension.
#     """
#     def __init__(self, d_model, expansion_factor=1.0, dropout=0.0, kernel_size=3):
#         super().__init__()
#         self.d_model = d_model
#         self.expansion_factor = expansion_factor
#         self.dropout = dropout
        
#         # Calculate expanded dimension
#         expanded_dim = int(d_model * expansion_factor)
        
#         # Input projection and normalization
#         self.in_proj = nn.Linear(d_model, expanded_dim * 2)
#         self.in_norm = RMSNorm(expanded_dim * 2, eps=1e-8)
        
#         # Regular convolution for initialization
#         self.conv = nn.Conv1d(expanded_dim, expanded_dim, kernel_size=kernel_size, padding=kernel_size//2, groups=expanded_dim)
        
#         # Flash FFT Conv layer, passing weights and bias from self.conv
#         self.flash_conv = FlashDepthWiseConv1d(expanded_dim, kernel_size=kernel_size, padding=kernel_size//2,
#                                               weights=self.conv.weight, bias=self.conv.bias)
        
#         # Output projection and normalization
#         self.out_proj = nn.Linear(expanded_dim, d_model)
#         self.out_norm = RMSNorm(d_model, eps=1e-8)
        
#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)
        
        
#     def forward(self, u):
#         # Input projection and normalization
#         xv = self.in_norm(self.in_proj(u))
        
#         # Split into x and v for gating
#         x, v = xv.chunk(2, dim=-1)
        
#         # Apply Flash FFT Convolution
#         x_feature_mixed = self.flash_conv(rearrange(x, 'b t f -> b f t').contiguous())
#         x_feature_mixed = rearrange(x_feature_mixed, 'b f t -> b t f')
        
#         # Apply gating with v
#         gated_output = v * x_feature_mixed
        
#         # Output projection and normalization
#         out = self.out_norm(self.out_proj(gated_output))
        
#         return out




class Lyra(nn.Module):
    """
    Lyra neural network architecture combining PGC modules with S4D for sequence modeling.
    
    Architecture:
    1. Input encoding
    2. Two PGC modules with different expansion factors
    3. S4D layer with residual connection
    4. Global pooling and decoding
    
    Parameters:
    -----------
    d_input : int
        Dimension of input features
    d_output : int
        Dimension of output features
    d_model : int
        Internal model dimension
    d_state : int, default=64
        State dimension for S4D layer
    dropout : float, default=0.2
        Dropout rate
    transposed : bool, default=False
        Whether the input is transposed for S4D
    **kernel_args : 
        Additional arguments for S4D kernel
    """
    def __init__(self, d_input, d_output, d_model, d_state=64, dropout=0.2, transposed=False, pgc_kernel_size=3, bidirectional=False, 
            attention=False, n_heads=4, attn_dropout=0.1, ssm="s4d", pgc_kernels=None, pgc_parallel=False, s4_layers=1, **kernel_args):
        super().__init__()

        self.attention = attention
        self.bidirectional = bidirectional
        self.s4_layers = max(1, int(s4_layers))

        # Resolve the PGC conv kernel sizes. `pgc_kernels` may be a list
        # ([3, 5]) or a "-"-joined string ("3-5", convenient for CLI/dir names).
        # The first kernel is the primary (self.conv); the rest are summed as
        # parallel multi-scale kernels. Default = single kernel = legacy PGC.
        if pgc_kernels is None:
            _ks = [pgc_kernel_size]
        elif isinstance(pgc_kernels, str):
            _ks = [int(x) for x in pgc_kernels.split("-") if x != ""]
        else:
            _ks = list(pgc_kernels)
        _primary, _extra = _ks[0], _ks[1:]
        self.pgc_parallel = pgc_parallel

        # Input encoding
        self.encoder = nn.Linear(d_input, d_model)
        
        # PGC modules with different expansion factors
        # First PGC compresses (0.25x), second PGC expands (2x)
        self.pgc1 = PGC(d_model, expansion_factor=0.25, dropout=dropout, kernel_size=_primary, extra_kernel_sizes=_extra)
        self.pgc2 = PGC(d_model, expansion_factor=2, dropout=dropout, kernel_size=_primary, extra_kernel_sizes=_extra)


        self.s4ds = nn.ModuleList([
            _build_ssm(ssm, d_model, d_state=d_state,
                       dropout=dropout, transposed=transposed, **kernel_args)
            for _ in range(self.s4_layers)
        ])
        self.s4_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(self.s4_layers)])
        self.s4d = self.s4ds[0]
        self.norm = self.s4_norms[0]

        # Optional gated-conv branch running in parallel with the SSM; its
        # output is summed with the SSM output before the residual.
        if pgc_parallel:
            self.pgc_par = PGC(d_model, expansion_factor=1.0, dropout=dropout, kernel_size=_primary, extra_kernel_sizes=_extra)

        if attention:
            self.attn_norm   = RMSNorm(d_model)
            self.attn        = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
            self.attn_alpha  = nn.Parameter(torch.tensor(1e-2))
        

        if bidirectional:
            self.pgc1_backward = PGC(d_model, expansion_factor=0.25, dropout=dropout, kernel_size=_primary, extra_kernel_sizes=_extra)
            self.pgc2_backward = PGC(d_model, expansion_factor=2, dropout=dropout, kernel_size=_primary, extra_kernel_sizes=_extra)

            self.s4ds_backward = nn.ModuleList([
                _build_ssm(ssm, d_model, d_state=d_state,
                           dropout=dropout, transposed=transposed, **kernel_args)
                for _ in range(self.s4_layers)
            ])
            self.s4_norms_backward = nn.ModuleList(
                [RMSNorm(d_model) for _ in range(self.s4_layers)])
            self.s4d_backward = self.s4ds_backward[0]
            self.norm_backward = self.s4_norms_backward[0]
            if pgc_parallel:
                self.pgc_par_backward = PGC(d_model, expansion_factor=1.0, dropout=dropout, kernel_size=_primary, extra_kernel_sizes=_extra)

            if attention:
                # --- NEW: attention for backward lane too ---
                self.attn_norm_backward = RMSNorm(d_model)
                self.attn_backward = nn.MultiheadAttention(
                    d_model, n_heads, dropout=attn_dropout, batch_first=True
                )
                self.attn_alpha_backward = nn.Parameter(torch.tensor(1e-2))

            
            # Combine forward and backward
            self.combine = nn.Linear(d_model * 2, d_model)
        
        # output layers
        self.decoder = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)

    def _s4_residual_stack(self, x, s4ds, norms, pgc_par=None, mid_add=None):
        """PGC already applied. Residual S4D layers; optional add after layer 0."""
        for i, (s4d, norm) in enumerate(zip(s4ds, norms)):
            z = norm(x)
            mix = self.dropout(s4d(z))
            if i == 0 and pgc_par is not None:
                mix = mix + self.dropout(pgc_par(z))
            x = mix + x
            if i == 0 and mid_add is not None:
                x = x + mid_add
        return x

    def forward_features(self, u, encoded=False, mid_add=None):
        # Encode input to model dimension (or take already-encoded tokens).
        x = u if encoded else self.encoder(u)

        # forward lane
        x_forward = self.pgc1(x)
        x_forward = self.pgc2(x_forward)
        par = self.pgc_par if self.pgc_parallel else None
        x_forward = self._s4_residual_stack(
            x_forward, self.s4ds, self.s4_norms, pgc_par=par, mid_add=mid_add)

        if self.attention:
            a = self.attn_norm(x_forward)
            a, _ = self.attn(a, a, a, need_weights=False)
            x_forward = x_forward + self.attn_alpha * self.dropout(a)

        if self.bidirectional:
            x_backward = self.pgc1_backward(torch.flip(x, dims=[1]))
            x_backward = self.pgc2_backward(x_backward)
            par_b = self.pgc_par_backward if self.pgc_parallel else None
            mid_b = torch.flip(mid_add, dims=[1]) if mid_add is not None else None
            x_backward = self._s4_residual_stack(
                x_backward, self.s4ds_backward, self.s4_norms_backward,
                pgc_par=par_b, mid_add=mid_b)
            x_backward = torch.flip(x_backward, dims=[1])

            if self.attention:
                ab = self.attn_norm_backward(x_backward)
                ab, _ = self.attn_backward(ab, ab, ab, need_weights=False)
                x_backward = x_backward + self.attn_alpha_backward * self.dropout(ab)

            x = self.combine(torch.cat([x_forward, x_backward], dim=-1))
        else:
            x = x_forward

        pooled = x.mean(dim=1)          # [B, d_model]
        return pooled

    def forward(self, u):
        pooled = self.forward_features(u)  # [B, d_model]
        x = self.decoder(pooled)           # [B, d_output]
        return x


class PureSSM(nn.Module):
    """Vanilla SSM model — no PGC/Lyra scaffolding.

    encoder -> [pre-norm -> SSM -> residual] x n_layers -> mean-pool -> decoder

    This is the stock way an SSM (Mamba/S4D/etc.) is used, so it serves as the
    "is the Lyra scaffolding actually helping?" baseline. Exposes the same
    forward_features() API as Lyra so it can drop into the tabular wrapper.
    """
    def __init__(self, d_input, d_output, d_model, d_state=64, dropout=0.2,
                 ssm="mamba2", n_layers=2, bidirectional=False,
                 transposed=False, **kernel_args):
        super().__init__()
        self.bidirectional = bidirectional
        self.encoder = nn.Linear(d_input, d_model)

        def _make_stack():
            return nn.ModuleList([
                nn.ModuleDict({
                    "norm": RMSNorm(d_model),
                    "ssm": _build_ssm(ssm, d_model, d_state=d_state,
                                      dropout=dropout, transposed=transposed,
                                      **kernel_args),
                    "dropout": nn.Dropout(dropout),
                })
                for _ in range(n_layers)
            ])

        self.layers = _make_stack()
        if bidirectional:
            self.layers_backward = _make_stack()
            self.combine = nn.Linear(2 * d_model, d_model)

        self.decoder = nn.Linear(d_model, d_output)

    @staticmethod
    def _run_stack(x, layers):
        for layer in layers:
            z = layer["norm"](x)
            x = layer["dropout"](layer["ssm"](z)) + x
        return x

    def forward_features(self, u, encoded=False):
        x = u if encoded else self.encoder(u)
        x_forward = self._run_stack(x, self.layers)
        if self.bidirectional:
            x_backward = self._run_stack(torch.flip(x, dims=[1]), self.layers_backward)
            x_backward = torch.flip(x_backward, dims=[1])
            x = self.combine(torch.cat([x_forward, x_backward], dim=-1))
        else:
            x = x_forward
        return x.mean(dim=1)

    def forward(self, u):
        return self.decoder(self.forward_features(u))


class LyraPhysicsCond(nn.Module):
    """Designer-style physics token conditioning on top of Lyra / PureSSM.

    Input tensor is still the Wave-1 physics-token layout
    ``[seq 13 | per-token role | global | optional running]``. The encoder
    only sees the 13 sequence channels (nucleotides + mix + conc). Role and
    global physics go through small MLPs and are added after the encoder:

        x += 0.25 * tanh(role_proj(role))
        x += 0.25 * tanh(glob_proj(global))

    Matching ``lyra_physics_tokens`` in lamp-designer, minus their extra
    global conc/mix (already in the 13-d sequence) and 3′/5′ extras unless
    ``physics_role_dim=8``.
    """

    SEQ_CHANNELS = 13
    SCALE = 0.25

    def __init__(self, lyra_cfg, physics_role_dim=4, physics_running=False):
        super().__init__()
        self.n_role = int(physics_role_dim)
        self.n_glob = 2
        self.n_spatial = 3 if physics_running else 0
        cfg = dict(lyra_cfg)
        cfg["d_input"] = self.SEQ_CHANNELS
        use_pure = cfg.pop("pure_ssm", False)
        self.lyra = PureSSM(**cfg) if use_pure else Lyra(**cfg)
        d_model = int(lyra_cfg["d_model"])
        role_in = self.n_role + self.n_spatial
        self.role_physics_projection = nn.Sequential(
            nn.LayerNorm(role_in),
            nn.Linear(role_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.global_physics_projection = nn.Sequential(
            nn.LayerNorm(self.n_glob),
            nn.Linear(self.n_glob, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        expected = self.SEQ_CHANNELS + self.n_role + self.n_glob + self.n_spatial
        print(
            f"LyraPhysicsCond: encoder d_input={self.SEQ_CHANNELS}, "
            f"tensor d_input>={expected}, role_dim={self.n_role}, "
            f"spatial={self.n_spatial}, scale={self.SCALE}"
        )

    def _physics_delta(self, u):
        i = self.SEQ_CHANNELS
        role = u[..., i:i + self.n_role]
        i += self.n_role
        glob = u[..., i:i + self.n_glob]
        i += self.n_glob
        if self.n_spatial:
            spatial = u[..., i:i + self.n_spatial]
            role = torch.cat([role, spatial], dim=-1)
        delta = self.SCALE * torch.tanh(self.role_physics_projection(role))
        glob_vec = glob[:, 0, :]
        delta = delta + self.SCALE * torch.tanh(
            self.global_physics_projection(glob_vec)
        ).unsqueeze(1)
        return delta

    def _conditioned_tokens(self, u):
        seq = u[..., :self.SEQ_CHANNELS]
        return self.lyra.encoder(seq) + self._physics_delta(u)

    def forward_features(self, u):
        return self.lyra.forward_features(self._conditioned_tokens(u), encoded=True)

    def forward(self, u):
        return self.lyra.decoder(self.forward_features(u))


class LyraPhysicsStaged(LyraPhysicsCond):
    """Sequence-only first S4D residual, then physics, then a second S4D.

    Encoder and PGC + S4D layer 0 see only the 13 sequence channels. After that
    residual, Designer-style physics is added and S4D layer 1 mixes it. This is
    ``--physics_staged``; requires ``s4_layers >= 2`` on the inner Lyra.
    """

    def __init__(self, lyra_cfg, physics_role_dim=4, physics_running=False):
        cfg = dict(lyra_cfg)
        cfg["s4_layers"] = max(int(cfg.get("s4_layers", 2)), 2)
        super().__init__(cfg, physics_role_dim=physics_role_dim,
                         physics_running=physics_running)
        print(f"LyraPhysicsStaged: s4_layers={self.lyra.s4_layers}, "
              f"physics injected after S4D layer 0")

    def forward_features(self, u):
        seq = u[..., :self.SEQ_CHANNELS]
        return self.lyra.forward_features(seq, mid_add=self._physics_delta(u))


class StackedLyraBase(nn.Module):
    """
    Lyra neural network architecture combining PGC modules with S4D for sequence modeling.
    
    Architecture:
    1. Input encoding
    2. Two PGC modules with different expansion factors
    3. S4D layer with residual connection
    4. Global pooling and decoding
    
    Parameters:
    -----------
    d_input : int
        Dimension of input features
    d_output : int
        Dimension of output features
    d_model : int
        Internal model dimension
    d_state : int, default=64
        State dimension for S4D layer
    dropout : float, default=0.2
        Dropout rate
    transposed : bool, default=False
        Whether the input is transposed for S4D
    **kernel_args : 
        Additional arguments for S4D kernel
    """
    def __init__(self, d_input, d_output, d_model, d_state=64, dropout=0.2, transposed=False, pgc_kernel_size=3, bidirectional=False, 
            attention=False, n_heads=4, attn_dropout=0.1, **kernel_args):
        super().__init__()

        self.attention = attention
        self.bidirectional = bidirectional
        
        # Input encoding
        self.encoder = nn.Linear(d_input, d_model)
        
        # PGC modules with different expansion factors
        # First PGC compresses (0.25x), second PGC expands (2x)
        self.pgc1 = PGC(d_model, expansion_factor=0.25, dropout=dropout, kernel_size=pgc_kernel_size)
        self.pgc2 = PGC(d_model, expansion_factor=2, dropout=dropout, kernel_size=pgc_kernel_size)


        # S4D layer for sequence modeling
        self.s4d = S4D(d_model, d_state=d_state, dropout=dropout, transposed=transposed, **kernel_args)
        self.norm = RMSNorm(d_model)

        if attention:
            # --- NEW: tiny attention block (pre-norm + MHSA + scaled residual) ---
            self.attn_norm   = RMSNorm(d_model)
            self.attn        = nn.MultiheadAttention(d_model, n_heads, dropout=attn_dropout, batch_first=True)
            self.attn_alpha  = nn.Parameter(torch.tensor(1e-2))  # small residual scale
        

        if bidirectional:
            # Backward direction components
            self.pgc1_backward = PGC(d_model, expansion_factor=0.25, dropout=dropout, kernel_size=pgc_kernel_size)
            self.pgc2_backward = PGC(d_model, expansion_factor=2, dropout=dropout, kernel_size=pgc_kernel_size)

            self.s4d_backward = S4D(d_model, d_state=d_state, dropout=dropout, transposed=transposed, **kernel_args)
            self.norm_backward = RMSNorm(d_model)

            if attention:
                # --- NEW: attention for backward lane too ---
                self.attn_norm_backward = RMSNorm(d_model)
                self.attn_backward = nn.MultiheadAttention(
                    d_model, n_heads, dropout=attn_dropout, batch_first=True
                )
                self.attn_alpha_backward = nn.Parameter(torch.tensor(1e-2))

            
            # Combine forward and backward
            self.combine = nn.Linear(d_model * 2, d_model)
        
        # output layers
        self.decoder = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, u):
        # Encode input to model dimension
        x = self.encoder(u)

        # forward lane
        x_forward = self.pgc1(x)
        x_forward = self.pgc2(x_forward)
        z_forward = self.norm(x_forward)
        x_forward = self.dropout(self.s4d(z_forward)) + x_forward

        if self.attention:
            a = self.attn_norm(x_forward)
            a, _ = self.attn(a, a, a, need_weights=False)
            x_forward = x_forward + self.attn_alpha * self.dropout(a)

        if self.bidirectional:
            x_backward = self.pgc1_backward(torch.flip(x, dims=[1]))
            x_backward = self.pgc2_backward(x_backward)
            z_backward = self.norm_backward(x_backward)
            x_backward = self.dropout(self.s4d_backward(z_backward)) + x_backward
            x_backward = torch.flip(x_backward, dims=[1])

            if self.attention:
                ab = self.attn_norm_backward(x_backward)
                ab, _ = self.attn_backward(ab, ab, ab, need_weights=False)
                x_backward = x_backward + self.attn_alpha_backward * self.dropout(ab)

            x = self.combine(torch.cat([x_forward, x_backward], dim=-1))
        else:
            x = x_forward

        return x

    def forward(self, u):
        x = self.forward_features(u)
        return x

class StackedLyra(nn.Module):
    def __init__(self, d_input, d_output, d_model, d_state=64, dropout=0.2, transposed=False, pgc_kernel_size=3, bidirectional=False, 
            attention=False, n_heads=4, attn_dropout=0.1, **kernel_args):
        super().__init__()

        self.lyra1 = StackedLyraBase(d_input, d_output, d_model, d_state=d_state, dropout=dropout, transposed=transposed, pgc_kernel_size=3, bidirectional=bidirectional, attention=attention, n_heads=n_heads, attn_dropout=attn_dropout, **kernel_args)
        self.lyra2 = StackedLyraBase(d_model, d_output, d_model, d_state=d_state, dropout=dropout, transposed=transposed, pgc_kernel_size=9, bidirectional=bidirectional, attention=attention, n_heads=n_heads, attn_dropout=attn_dropout, **kernel_args)

        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, u):
        x = self.lyra1(u)
        x = self.lyra2(x)
        pooled = x.mean(dim=1)          # [B, d_model]
        x = self.decoder(pooled)           # [B, d_output]

        return x


class LyraDeep(nn.Module):
    """
    Deep Lyra architecture with skip connections (Nauman's splicing model).
    
    Architecture pattern (per block):
    1. PGC module
    2. Normalization
    3. S4D backbone with residual
    4. Skip connection from 2 blocks back
    
    Parameters:
    -----------
    d_input : int
        Dimension of input features
    d_output : int
        Dimension of output features
    d_model : int
        Internal model dimension
    d_state : int, default=64
        State dimension for S4D layer
    dropout : float, default=0.2
        Dropout rate
    num_blocks : int, default=8
        Number of Lyra blocks to stack
    pgc_expansion : float, default=2.0
        Expansion factor for PGC modules
    pgc_kernel : int, default=11
        Kernel size for PGC convolutions
    bidirectional : bool, default=False
        Whether to use bidirectional processing
    transposed : bool, default=False
        Whether the input is transposed for S4D
    """
    def __init__(self, d_input, d_output, d_model, d_state=64, dropout=0.25, 
                 num_blocks=8, pgc_expansion=2.0, pgc_kernel=11, 
                 bidirectional=False, transposed=False, **kernel_args):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.num_blocks = num_blocks
        
        # Input encoding
        self.encoder = nn.Linear(d_input, d_model)
        
        # Create Lyra blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Create S4D backbone
            backbone = S4D(d_model, d_state=d_state, dropout=dropout, 
                          transposed=transposed, **kernel_args)
            
            block = nn.ModuleDict({
                'pgc': PGC(d_model, 
                          expansion_factor=pgc_expansion, 
                          dropout=dropout, 
                          kernel_size=pgc_kernel),
                'norm': RMSNorm(d_model),
                'backbone': backbone,
                'dropout': nn.Dropout(dropout)
            })
            self.blocks.append(block)
        
        # Bidirectional components
        if bidirectional:
            self.blocks_backward = nn.ModuleList()
            for i in range(num_blocks):
                backbone_bwd = S4D(d_model, d_state=d_state, dropout=dropout, 
                                  transposed=transposed, **kernel_args)
                
                block_bwd = nn.ModuleDict({
                    'pgc': PGC(d_model, 
                              expansion_factor=pgc_expansion, 
                              dropout=dropout, 
                              kernel_size=pgc_kernel),
                    'norm': RMSNorm(d_model),
                    'backbone': backbone_bwd,
                    'dropout': nn.Dropout(dropout)
                })
                self.blocks_backward.append(block_bwd)
            
            # Combine forward and backward
            self.combine = nn.Linear(d_model * 2, d_model)
        
        # Output head
        self.decoder = nn.Linear(d_model, d_output)
        
    def forward_direction(self, X, blocks):
        """Process one direction with skip connections."""
        h = X
        
        # Store outputs for skip connections
        skip_connections = []
        
        for i, block in enumerate(blocks):
            # Lyra block: PGC -> Norm -> Backbone -> Residual
            h = block['pgc'](h)
            z = h
            z = block['norm'](z)
            h = block['dropout'](block['backbone'](z)) + h
            
            # Add skip connection from 2 blocks back
            if i >= 2:
                h = h + skip_connections[i-2]
            
            # Store for future skip connections
            skip_connections.append(h)
        
        return h
    
    def forward(self, X):
        # Encode input
        h = self.encoder(X)
        
        # Forward direction
        h_forward = self.forward_direction(h, self.blocks)
        
        if self.bidirectional:
            # Backward direction (flip input)
            h_backward = self.forward_direction(torch.flip(h, dims=[1]), 
                                               self.blocks_backward)
            # Flip back
            h_backward = torch.flip(h_backward, dims=[1])
            
            # Combine both directions
            h = self.combine(torch.cat([h_forward, h_backward], dim=-1))
        else:
            h = h_forward
        
        # Global pooling and output
        pooled = h.mean(dim=1)  # [B, d_model]
        return self.decoder(pooled)  # [B, d_output]