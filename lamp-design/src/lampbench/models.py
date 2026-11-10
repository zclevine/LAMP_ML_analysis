from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from lyra.pgc import PGC as CanonicalLyraPGC
from lyra.s4d import S4D as CanonicalLyraS4D

from .data import META_FEATURES, PRIMER_TYPES, TABULAR_FEATURES

N_TAB = len(TABULAR_FEATURES)
CONTEXT_TAB_IDX = [TABULAR_FEATURES.index(name) for name in META_FEATURES]
N_CONTEXT_TAB = len(META_FEATURES)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp_min(1)


def reverse_valid(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for i, n in enumerate(lengths.tolist()):
        out[i, :n] = x[i, :n].flip(0)
    return out


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.square().mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


class DropoutNd(nn.Module):
    def __init__(self, p: float, featurewise: bool = True):
        super().__init__()
        self.p = p
        self.featurewise = featurewise

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        shape = ((x.shape[0], x.shape[1], 1) if self.featurewise
                 else (x.shape[0], 1, x.shape[2]))
        mask = (torch.rand(shape, device=x.device) >= self.p).to(x.dtype)
        return x * mask / (1 - self.p)


class S4DKernel(nn.Module):
    def __init__(self, d_model: int, n_state: int = 64):
        super().__init__()
        log_dt = torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        c = torch.randn(d_model, n_state // 2, dtype=torch.cfloat)
        self.c = nn.Parameter(torch.view_as_real(c))
        self.log_dt = nn.Parameter(log_dt)
        self.log_a_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n_state // 2)))
        self.a_imag = nn.Parameter(math.pi * torch.arange(n_state // 2).repeat(d_model, 1))

    def forward(self, length: int):
        dt = self.log_dt.exp()
        c = torch.view_as_complex(self.c)
        a = -self.log_a_real.exp() + 1j * self.a_imag
        dta = a * dt[:, None]
        k = dta[:, :, None] * torch.arange(length, device=a.device)
        c = c * (dta.exp() - 1) / a
        return 2 * torch.einsum("hn,hnl->hl", c, k.exp()).real


class S4D(nn.Module):
    def __init__(self, d_model: int, dropout: float, featurewise_dropout: bool = True):
        super().__init__()
        self.kernel = S4DKernel(d_model)
        self.d = nn.Parameter(torch.randn(d_model))
        self.dropout = DropoutNd(dropout, featurewise_dropout)
        self.out = nn.Sequential(nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=1))

    def forward(self, x):
        u = x.transpose(1, 2)
        length = u.shape[-1]
        k = self.kernel(length)
        y = torch.fft.irfft(
            torch.fft.rfft(u, n=2 * length) * torch.fft.rfft(k, n=2 * length),
            n=2 * length,
        )[..., :length]
        y = y + u * self.d[:, None]
        return self.out(self.dropout(F.gelu(y))).transpose(1, 2)


class PGC(nn.Module):
    def __init__(self, d_model: int, expansion: float, dropout: float, dilation: int = 1):
        super().__init__()
        hidden = max(4, int(d_model * expansion))
        self.in_proj = nn.Linear(d_model, 2 * hidden)
        self.in_norm = RMSNorm(2 * hidden)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation, groups=hidden)
        self.out = nn.Linear(hidden, d_model)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a, gate = self.in_norm(self.in_proj(x)).chunk(2, -1)
        a = self.conv(a.transpose(1, 2)).transpose(1, 2)
        return self.norm(self.dropout(self.out(a * gate)))


class LyraBlock(nn.Module):
    def __init__(self, d_model: int, dropout: float, multiscale: bool = False,
                 pgc_residual: bool = True, featurewise_dropout: bool = True):
        super().__init__()
        # Import the published Lyra primitives directly. The surrounding mask,
        # bidirectionality, pooling, and LAMP head are benchmark adaptations.
        self.pgc1 = CanonicalLyraPGC(d_model, expansion_factor=0.25, dropout=dropout)
        self.pgc2 = CanonicalLyraPGC(d_model, expansion_factor=2.0, dropout=dropout)
        self.multi = PGC(d_model, 1.0, dropout, 3) if multiscale else None
        self.s4 = CanonicalLyraS4D(
            d_model, d_state=64, dropout=dropout,
            featurewise_dropout=featurewise_dropout,
        )
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.pgc_residual = pgc_residual

    def forward(self, x, mask):
        w = mask.unsqueeze(-1).to(x.dtype)
        h = self.pgc2(self.pgc1(x * w))
        if self.multi is not None:
            h = h + self.multi(x * w)
        if self.pgc_residual:
            h = x * w + h
        h = h + self.drop(self.s4(self.norm(h)))
        return h * w


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x, mask):
        score = self.score(x).squeeze(-1).masked_fill(~mask, -1e4)
        return torch.einsum("bl,bld->bd", score.softmax(-1), x)


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, n_tab: int, task: str, dropout: float, fusion: str):
        super().__init__()
        self.task = task
        self.fusion = fusion
        if fusion == "deep_concat":
            self.tab = nn.Sequential(
                nn.LayerNorm(n_tab), nn.Linear(n_tab, 2 * d_model), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.GELU(),
            )
        else:
            self.tab = nn.Sequential(
                nn.LayerNorm(n_tab), nn.Linear(n_tab, d_model), nn.GELU())
        if fusion == "cross_gate":
            self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
            head_in = d_model
        elif fusion == "none":
            head_in = d_model
        else:
            head_in = 2 * d_model
        out_dim = 2 if task == "hurdle" else 1
        self.head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Linear(head_in, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, out_dim),
        )

    def forward(self, seq_repr, tab):
        if self.fusion == "none":
            fused = seq_repr
        else:
            tab_repr = self.tab(tab)
            if self.fusion == "cross_gate":
                gate = self.gate(torch.cat([seq_repr, tab_repr], -1))
                fused = gate * seq_repr + (1 - gate) * tab_repr
            else:
                fused = torch.cat([seq_repr, tab_repr], -1)
        return self.head(fused)


class LegacyPGC(nn.Module):
    """Faithful PGC from the collaborator handoff (including unused dropout)."""
    def __init__(self, d_model: int, expansion: float, dropout: float):
        super().__init__()
        hidden = int(d_model * expansion)
        self.conv = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
        self.in_proj = nn.Linear(d_model, 2 * hidden)
        self.in_norm = nn.RMSNorm(2 * hidden, eps=1e-8)
        self.out_norm = nn.RMSNorm(d_model, eps=1e-8)
        self.out_proj = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        local, gate = self.in_norm(self.in_proj(x)).chunk(2, -1)
        local = self.conv(local.transpose(1, 2)).transpose(1, 2)
        return self.out_norm(self.out_proj(gate * local))


class LegacyLyra(nn.Module):
    """Exact supplied Lyra computation, retaining its padding/reversal behavior."""
    def __init__(self, d_input=13, d_model=64, dropout=0.4, task="positive",
                 with_tabular=False):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.fwd = nn.ModuleList([LegacyPGC(d_model, 0.25, dropout),
                                  LegacyPGC(d_model, 2.0, dropout)])
        self.bwd = nn.ModuleList([LegacyPGC(d_model, 0.25, dropout),
                                  LegacyPGC(d_model, 2.0, dropout)])
        self.s4_fwd, self.s4_bwd = S4D(d_model, dropout), S4D(d_model, dropout)
        self.norm_fwd, self.norm_bwd = RMSNorm(d_model), RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.combine = nn.Linear(2 * d_model, d_model)
        self.with_tabular = with_tabular
        out_dim = 2 if task == "hurdle" else 1
        if with_tabular:
            n_thermo = 34
            self.tab = nn.Sequential(nn.LayerNorm(n_thermo), nn.Linear(n_thermo, d_model),
                                     nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(d_model, d_model), nn.ReLU())
            self.decoder = nn.Sequential(nn.LayerNorm(2 * d_model),
                                         nn.Linear(2 * d_model, d_model), nn.ReLU(),
                                         nn.Dropout(0.1), nn.Linear(d_model, out_dim))
        else:
            self.decoder = nn.Linear(d_model, out_dim)

    def _lane(self, x, pgcs, norm, s4):
        h = pgcs[1](pgcs[0](x))
        return self.drop(s4(norm(h))) + h

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq)
        fwd = self._lane(x, self.fwd, self.norm_fwd, self.s4_fwd)
        # Deliberately flips the full padded tensor, as in the handoff.
        bwd = self._lane(torch.flip(x, [1]), self.bwd, self.norm_bwd, self.s4_bwd)
        bwd = torch.flip(bwd, [1])
        pooled = self.combine(torch.cat([fwd, bwd], -1)).mean(1)
        if self.with_tabular:
            pooled = torch.cat([pooled, self.tab(tab[:, :34])], -1)
        return self.decoder(pooled)


class TabMLP(nn.Module):
    def __init__(self, n_tab=N_TAB, d_model=64, dropout=0.2, task="positive", **_):
        super().__init__()
        out_dim = 2 if task == "hurdle" else 1
        self.net = nn.Sequential(
            nn.LayerNorm(n_tab), nn.Linear(n_tab, 2 * d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, out_dim),
        )

    def forward(self, seq, role, mask, lengths, tab):
        return self.net(tab)


class LyraModel(nn.Module):
    def __init__(self, d_input=13, n_tab=N_TAB, d_model=64, depth=1, dropout=0.2,
                 bidirectional=True, role_embedding=True, masked_pool=True,
                 pool="mean", multiscale=False, fusion="concat", task="positive",
                 shared_bidirectional=False, pgc_residual=True,
                 featurewise_dropout=True, physics_tokens=False, context_tab=False):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.role = nn.Embedding(9, d_model) if role_embedding else None
        self.tab_conditioner = None
        self.physics_tokens = physics_tokens
        self.context_tab = context_tab
        if context_tab:
            n_tab = N_CONTEXT_TAB
        if physics_tokens:
            role_indices = [[TABULAR_FEATURES.index(f"{primer}_{feature}")
                             for feature in ("mismatches", "gc_fraction", "tm",
                                             "dg_ViennaRNA", "mm_3p3", "mm_3p5",
                                             "mm_5p5", "mm_pos_weighted")]
                            for primer in PRIMER_TYPES]
            global_names = ["total_mismatches", "mean_dg_ViennaRNA", "log10_conc",
                            "mm_SS", "mm_BST", "mm_WS"]
            self.register_buffer("physics_role_indices", torch.tensor(role_indices, dtype=torch.long))
            self.register_buffer("physics_global_indices", torch.tensor(
                [TABULAR_FEATURES.index(name) for name in global_names], dtype=torch.long))
            self.role_physics_projection = nn.Sequential(
                nn.LayerNorm(8), nn.Linear(8, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
            self.global_physics_projection = nn.Sequential(
                nn.LayerNorm(len(global_names)), nn.Linear(len(global_names), d_model),
                nn.SiLU(), nn.Linear(d_model, d_model))
        prediction_fusion = fusion
        if fusion == "film_concat":
            # Condition sequence processing on chemistry/thermodynamics while
            # retaining late concatenation for an apples-to-apples fusion test.
            self.tab_conditioner = nn.Sequential(
                nn.LayerNorm(n_tab), nn.Linear(n_tab, 2 * d_model))
            prediction_fusion = "concat"
        self.blocks_fwd = nn.ModuleList([
            LyraBlock(d_model, dropout, multiscale, pgc_residual, featurewise_dropout)
            for _ in range(depth)
        ])
        self.bidirectional = bidirectional
        self.shared_bidirectional = shared_bidirectional
        self.masked_pool = masked_pool
        if bidirectional and not shared_bidirectional:
            self.blocks_bwd = nn.ModuleList([
                LyraBlock(d_model, dropout, multiscale, pgc_residual, featurewise_dropout)
                for _ in range(depth)
            ])
            self.combine = nn.Linear(2 * d_model, d_model)
        self.pool = AttentionPool(d_model) if pool == "attention" else None
        self.pool_kind = pool
        self.terminal_projection = nn.Linear(4 * d_model, d_model) if pool == "terminal" else None
        self.prediction = PredictionHead(
            d_model, n_tab, task, dropout, prediction_fusion)

    def _lane(self, x, mask, lengths, blocks):
        # Exact-length grouping prevents FFT transform length and non-causal
        # convolution boundaries from depending on unrelated padded samples.
        out = torch.zeros_like(x)
        for length in torch.unique(lengths).tolist():
            idx = torch.nonzero(lengths == length, as_tuple=True)[0]
            lane = x.index_select(0, idx)[:, :length]
            lane_mask = torch.ones((len(idx), length), dtype=torch.bool, device=x.device)
            for block in blocks:
                lane = block(lane, lane_mask)
            out[idx, :length] = lane
        return out

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq)
        if self.role is not None:
            x = x + self.role(role)
        if self.physics_tokens:
            # Align each primer's physical summaries to only its own tokens.
            # Separator tokens receive a zero role vector, while assay-level
            # chemistry, concentration, and aggregate thermodynamics condition
            # every valid position.
            role_features = tab[:, self.physics_role_indices]
            separator = torch.zeros((len(tab), 1, role_features.shape[-1]),
                                    dtype=tab.dtype, device=tab.device)
            role_features = torch.cat([role_features, separator], dim=1)
            # Projection commutes with role lookup because every token of a
            # primer receives the same eight physical summaries. Project the
            # nine role rows once, then gather, instead of repeating the MLP at
            # every nucleotide position.
            role_embedding_table = self.role_physics_projection(role_features)
            role_embedding = role_embedding_table.gather(
                1, role.unsqueeze(-1).expand(-1, -1, role_embedding_table.shape[-1]))
            global_embedding = self.global_physics_projection(
                tab[:, self.physics_global_indices]).unsqueeze(1)
            x = x + 0.25 * torch.tanh(role_embedding) + 0.25 * torch.tanh(global_embedding)
        if self.tab_conditioner is not None:
            scale, shift = self.tab_conditioner(tab).chunk(2, -1)
            x = x * (1 + 0.25 * torch.tanh(scale).unsqueeze(1))
            x = x + 0.25 * torch.tanh(shift).unsqueeze(1)
        fwd = self._lane(x, mask, lengths, self.blocks_fwd)
        if self.bidirectional:
            rev = reverse_valid(x, lengths)
            bwd = self._lane(rev, mask, lengths,
                             self.blocks_fwd if self.shared_bidirectional else self.blocks_bwd)
            bwd = reverse_valid(bwd, lengths)
            x = 0.5 * (fwd + bwd) if self.shared_bidirectional else self.combine(
                torch.cat([fwd, bwd], -1))
        else:
            x = fwd
        if self.pool_kind == "terminal":
            mean = masked_mean(x, mask)
            maximum = x.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
            last = x[torch.arange(len(x), device=x.device), lengths.clamp_min(1) - 1]
            pooled = self.terminal_projection(torch.cat([mean, maximum, x[:, 0], last], -1))
        elif self.pool is not None:
            pooled = self.pool(x, mask)
        elif self.masked_pool:
            pooled = masked_mean(x, mask)
        else:
            pooled = x.mean(1)
        if self.context_tab:
            tab = tab[:, CONTEXT_TAB_IDX]
        return self.prediction(pooled, tab)


class CNNModel(nn.Module):
    def __init__(self, d_input=13, n_tab=N_TAB, d_model=64, dropout=0.2, task="positive", **_):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.role = nn.Embedding(9, d_model)
        self.net = nn.ModuleList([
            nn.Conv1d(d_model, d_model, 5, padding=2),
            nn.Conv1d(d_model, d_model, 5, padding=4, dilation=2),
            nn.Conv1d(d_model, d_model, 5, padding=8, dilation=4),
        ])
        self.prediction = PredictionHead(d_model, n_tab, task, dropout, "concat")

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq) + self.role(role)
        w = mask.unsqueeze(-1).to(x.dtype)
        for conv in self.net:
            x = (x + F.gelu(conv((x * w).transpose(1, 2)).transpose(1, 2))) * w
        return self.prediction(masked_mean(x, mask), tab)


class S4DModel(nn.Module):
    def __init__(self, d_input=13, n_tab=N_TAB, d_model=64, dropout=0.2,
                 task="positive", **_):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.role = nn.Embedding(9, d_model)
        self.fwd = nn.ModuleList([
            CanonicalLyraS4D(d_model, dropout=dropout),
            CanonicalLyraS4D(d_model, dropout=dropout),
        ])
        self.bwd = nn.ModuleList([
            CanonicalLyraS4D(d_model, dropout=dropout),
            CanonicalLyraS4D(d_model, dropout=dropout),
        ])
        self.norm_f = nn.ModuleList([RMSNorm(d_model), RMSNorm(d_model)])
        self.norm_b = nn.ModuleList([RMSNorm(d_model), RMSNorm(d_model)])
        self.combine = nn.Linear(2 * d_model, d_model)
        self.prediction = PredictionHead(d_model, n_tab, task, dropout, "concat")

    def _lane(self, x, mask, blocks, norms):
        w = mask.unsqueeze(-1).to(x.dtype)
        for block, norm in zip(blocks, norms):
            x = (x + block(norm(x * w))) * w
        return x

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq) + self.role(role)
        fwd = self._lane(x, mask, self.fwd, self.norm_f)
        rev = reverse_valid(x, lengths)
        bwd = reverse_valid(self._lane(rev, mask, self.bwd, self.norm_b), lengths)
        return self.prediction(masked_mean(self.combine(torch.cat([fwd, bwd], -1)), mask), tab)


class GRUModel(nn.Module):
    def __init__(self, d_input=13, n_tab=N_TAB, d_model=64, dropout=0.2, task="positive", **_):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.role = nn.Embedding(9, d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True, bidirectional=True, num_layers=2,
                          dropout=dropout)
        self.proj = nn.Linear(2 * d_model, d_model)
        self.prediction = PredictionHead(d_model, n_tab, task, dropout, "concat")

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq) + self.role(role)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed, _ = self.gru(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        return self.prediction(masked_mean(self.proj(x), mask[:, :x.shape[1]]), tab)


class TransformerModel(nn.Module):
    def __init__(self, d_input=13, n_tab=N_TAB, d_model=64, dropout=0.2, task="positive",
                 fusion="concat", context_tab=False, **_):
        super().__init__()
        self.context_tab = context_tab
        if context_tab:
            n_tab = N_CONTEXT_TAB
        self.encoder = nn.Linear(d_input, d_model)
        self.role = nn.Embedding(9, d_model)
        layer = nn.TransformerEncoderLayer(d_model, 4, 4 * d_model, dropout, batch_first=True,
                                            norm_first=True, activation="gelu")
        self.net = nn.TransformerEncoder(layer, 3)
        self.prediction = PredictionHead(d_model, n_tab, task, dropout, fusion)

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq) + self.role(role)
        x = self.net(x, src_key_padding_mask=~mask)
        if self.context_tab:
            tab = tab[:, CONTEXT_TAB_IDX]
        return self.prediction(masked_mean(x, mask), tab)


def _preload_mamba_ssm() -> None:
    """Import mamba-ssm without executing a CUDA-ABI-broken package init.

    Torch 2.14 dropped a symbol still referenced by older selective_scan_cuda
    and causal-conv1d wheels. Mamba-2 can still run through the Triton scan.
    """
    import sys
    import types
    from pathlib import Path

    if "mamba_ssm.modules.mamba2" in sys.modules:
        return
    try:
        import mamba_ssm  # noqa: F401
        return
    except ImportError:
        sys.modules.pop("mamba_ssm", None)
    for entry in sys.path:
        candidate = Path(entry) / "mamba_ssm"
        if (candidate / "modules" / "mamba2.py").exists():
            package = types.ModuleType("mamba_ssm")
            package.__path__ = [str(candidate)]
            package.__file__ = str(candidate / "__init__.py")
            package.__package__ = "mamba_ssm"
            sys.modules["mamba_ssm"] = package
            return
    raise RuntimeError("mamba-ssm is required for Mamba models")


class MambaModel(nn.Module):
    def __init__(self, variant: str, d_input=13, n_tab=N_TAB, d_model=64, dropout=0.2,
                 task="positive", bidirectional=True, pooling="mean",
                 shared_bidirectional=False, fusion="concat", context_tab=False, **_):
        super().__init__()
        _preload_mamba_ssm()
        from mamba_ssm.modules.mamba2 import Mamba2
        self.context_tab = context_tab
        if context_tab:
            n_tab = N_CONTEXT_TAB
        self.encoder = nn.Linear(d_input, d_model)
        self.role = nn.Embedding(9, d_model)
        self.variant = variant
        self.bidirectional = bidirectional
        self.shared_bidirectional = shared_bidirectional
        self.pooling = pooling
        if variant == "mamba1":
            from mamba_ssm.modules.mamba_simple import Mamba
            factory = lambda: Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
        elif variant == "mamba2":
            factory = lambda: Mamba2(
                d_model=d_model, d_state=64, d_conv=4, expand=2, headdim=32,
                use_mem_eff_path=False,
            )
        elif variant in {"mamba3_siso", "mamba3_mimo"}:
            from mamba_ssm.modules.mamba3 import Mamba3
            is_mimo = variant.endswith("mimo")
            factory = lambda: Mamba3(
                d_model=d_model, d_state=128, headdim=64,
                is_mimo=is_mimo, mimo_rank=4,
                chunk_size=16 if is_mimo else 64,
                is_outproj_norm=False,
            )
        else:
            raise ValueError(variant)
        self.fwd = nn.ModuleList([factory(), factory()])
        if bidirectional and not shared_bidirectional:
            self.bwd = nn.ModuleList([factory(), factory()])
            self.combine = nn.Linear(2 * d_model, d_model)
        self.norm = RMSNorm(d_model)
        self.prediction = PredictionHead(d_model, n_tab, task, dropout, fusion)
        self.terminal_projection = nn.Linear(4 * d_model, d_model) if pooling == "terminal" else None

    def _lane(self, x, mask, blocks):
        w = mask.unsqueeze(-1).to(x.dtype)
        for block in blocks:
            x = (x + block(self.norm(x))) * w
        return x

    def forward(self, seq, role, mask, lengths, tab):
        x = self.encoder(seq) + self.role(role)
        fwd = self._lane(x, mask, self.fwd)
        if self.bidirectional:
            rev = reverse_valid(x, lengths)
            bwd = reverse_valid(self._lane(
                rev, mask, self.fwd if self.shared_bidirectional else self.bwd), lengths)
            x = 0.5 * (fwd + bwd) if self.shared_bidirectional else self.combine(
                torch.cat([fwd, bwd], -1))
        else:
            x = fwd
        pooled = masked_mean(x, mask)
        if self.pooling == "terminal":
            maximum = x.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
            last = x[torch.arange(len(x), device=x.device), lengths.clamp_min(1) - 1]
            pooled = self.terminal_projection(torch.cat([pooled, maximum, x[:, 0], last], -1))
        if self.context_tab:
            tab = tab[:, CONTEXT_TAB_IDX]
        return self.prediction(pooled, tab)


def build_model(name: str, task: str, d_model: int = 64, dropout: float = 0.2) -> nn.Module:
    common = dict(d_model=d_model, dropout=dropout, task=task)
    if name == "lyra_original":
        return LegacyLyra(d_model=d_model, dropout=0.4, task=task)
    if name == "lyra_original_tabular":
        return LegacyLyra(d_model=d_model, dropout=0.4, task=task, with_tabular=True)
    if name == "lyra_corrected":
        return LyraModel(**common, depth=1, role_embedding=True, masked_pool=True,
                         fusion="concat", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_corrected_crossgate":
        return LyraModel(**common, depth=1, role_embedding=True, masked_pool=True,
                         fusion="cross_gate", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_corrected_tabdeep":
        return LyraModel(**common, depth=1, role_embedding=True, masked_pool=True,
                         fusion="deep_concat", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_corrected_film":
        return LyraModel(**common, depth=1, role_embedding=True, masked_pool=True,
                         fusion="film_concat", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_corrected_nolate":
        return LyraModel(**common, depth=1, role_embedding=True, masked_pool=True,
                         fusion="none", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_physics_tokens":
        return LyraModel(**common, d_input=23, depth=1, role_embedding=True, masked_pool=True,
                         fusion="concat", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_physics_crossgate":
        return LyraModel(**common, d_input=23, depth=1, role_embedding=True, masked_pool=True,
                         fusion="cross_gate", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_physics_nolate":
        return LyraModel(**common, d_input=23, depth=1, role_embedding=True, masked_pool=True,
                         fusion="none", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_paper_mean":
        return LyraModel(**common, depth=1, bidirectional=False, pool="mean",
                         pgc_residual=False, featurewise_dropout=False)
    if name == "lyra_paper_terminal":
        return LyraModel(**common, depth=1, bidirectional=False, pool="terminal",
                         pgc_residual=False, featurewise_dropout=False)
    if name == "lyra_paper_resfix":
        return LyraModel(**common, depth=1, bidirectional=False, pool="terminal",
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_paper_resfix_bidir":
        return LyraModel(**common, depth=1, pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True)
    if name == "lyra_seqonly":
        return LyraModel(**common, depth=1, role_embedding=True, masked_pool=True, fusion="none")
    if name == "lyra_seq_context":
        return LyraModel(**common, d_input=13, depth=1, role_embedding=True, masked_pool=True,
                         fusion="concat", pool="terminal", shared_bidirectional=True,
                         pgc_residual=True, featurewise_dropout=True, context_tab=True)
    if name == "transformer_seq_context":
        return TransformerModel(**common, d_input=13, fusion="concat", context_tab=True)
    if name == "mamba2_seq_context":
        return MambaModel("mamba2", **common, d_input=13, pooling="terminal",
                          bidirectional=True, shared_bidirectional=True,
                          fusion="concat", context_tab=True)
    if name == "lyra_no_role":
        return LyraModel(**common, depth=1, role_embedding=False, masked_pool=True, fusion="concat")
    if name == "lyra_unidirectional":
        return LyraModel(**common, depth=1, bidirectional=False, role_embedding=True,
                         masked_pool=True, fusion="concat")
    if name == "lyra_deep":
        return LyraModel(**common, depth=3, role_embedding=True, masked_pool=True, fusion="concat")
    if name == "lyra_multiscale":
        return LyraModel(**common, depth=2, multiscale=True, fusion="concat")
    if name == "lyra_attnpool":
        return LyraModel(**common, depth=2, pool="attention", fusion="concat")
    if name == "lyra_crossgate":
        return LyraModel(**common, depth=2, pool="attention", fusion="cross_gate")
    if name == "cnn":
        return CNNModel(**common)
    if name == "s4d":
        return S4DModel(**common)
    if name == "bigru":
        return GRUModel(**common)
    if name == "transformer":
        return TransformerModel(**common)
    if name == "transformer_physics_tokens":
        return TransformerModel(**common, d_input=23)
    if name == "mamba2_physics_tokens":
        return MambaModel("mamba2", **common, d_input=23, pooling="terminal",
                          bidirectional=True, shared_bidirectional=True)
    if name == "tab_mlp":
        return TabMLP(**common)
    if name in {"mamba1", "mamba2", "mamba3_siso", "mamba3_mimo"}:
        return MambaModel(name, **common)
    if name in {"mamba2_terminal", "mamba2_terminal_bidir",
                "mamba3_terminal", "mamba3_terminal_bidir"}:
        variant = "mamba2" if name.startswith("mamba2") else "mamba3_siso"
        return MambaModel(variant, **common, pooling="terminal",
                          bidirectional=name.endswith("bidir"), shared_bidirectional=True)
    raise ValueError(name)
