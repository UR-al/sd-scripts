# TSM-LoKr (network_args algo=tsm-lokr; part of the LyCORIS fixed paths, on by default, not with --lycoris_stock):
# LyCORIS' TimeStep Master routing (tsm.py) with LoKr (Kronecker) experts and an optional DoRA magnitude shared by all
# experts. LyCORIS in site-packages is untouched:
# lycoris_algo_fixes.apply_module_patches() adds "tsm-lokr" to LyCORIS' algo table (wrapper.network_module_dict) in
# memory. algo=tsm (LoRA experts) and algo=lokr are not changed by anything in this file.
#
# Per target Linear (base weight W0 [out, in]) and per sample with integer timestep t (set_timestep: round(sigma*1000)):
#   core expert i = idx_0(t), context expert j = idx_s(t) for the other scales (tsm.py:297-315)
#   K_e = s_e * kron(w1_e, w2_e)                      (LyCORIS LoKr factors and scale rule, lokr.py:204-270)
#   stage 1:  W'(t) = D(W0 + K_core,i + sum_s K_ctx,s)
#   stage 2:  W'_b  = D(W0 + K_core,i + sum_s g_bs K_ctx,s),  g_b = router_fc(mean x_b) + timestep_embed[t_b - 1]
#             (no sigmoid, tsm.py:380-387; the router sees the module input pooled per sample)
#   D(V) = V * m / (||V||_row + eps), m = dora_scale [out, 1] (dora_wd=True, wd_on_output=True), else D(V) = V
#   strength mu (network multiplier): W0 + mu * (W' - W0), Forge's blend (weight_adapter/base.py:53-57); mu = 0 runs the
#   base layer exactly. At mu = 1 this is D(V), the same weight LyCORIS LoKr builds (lokr.py:648-666).
# Stage 1 rebuilds the dense merged weight on every call, like the fork's LoKr fast path, and is compiled with the same
# switch (--anima_no_lokr_compile). Stage 2 (experts frozen) never builds a weight per sample: with C = K_core and
# X = K_ctx fixed per (module, interval), ||W0 + C + g X||^2_row = a + 2 g b + g^2 c, so
#   y_b = (m / (sqrt(a + 2 g_b b + g_b^2 c) + eps)) * (x_b (W0 + C)^T + g_b * x_b X^T)
# with x X^T computed from the factors (kron_mv, no dense X).
#
# Saved keys per module (default nn.Module naming): experts.<scale>.<i>.lokr_w1 (or lokr_w1_a/_b), .lokr_w2 (or
# lokr_w2_a/_b), alpha, dora_scale, router_fc.weight/.bias, timestep_embed.weight. Metadata: see metadata_json().

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from library.elementwise_compile import ElementwiseChains
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

ALGO = "tsm-lokr"
SAVED_FORMAT = "tsm-lokr-native"
METADATA_EXPERT_KEY = "ss_tsm_expert"
METADATA_CONFIG_KEY = "ss_tsm_lokr"
FORMAT_VERSION = 1

_TRUE = ("true", "1", "yes", "on")
_FALSE = ("false", "0", "no", "off", "none", "")


def _err(en: str, ko: str) -> ValueError:
    return ValueError(f"tsm-lokr: {en} / tsm-lokr: {ko}")


def _bool(kw: dict, key: str, default: bool) -> bool:
    if key not in kw or kw[key] is None:
        return default
    v = kw[key]
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise _err(f"{key}={v!r} is not a boolean", f"{key}={v!r} 는 참/거짓 값이 아닙니다")
    return bool(v)


def _int(kw: dict, key: str, default: Optional[int]) -> Optional[int]:
    if key not in kw or kw[key] is None or (isinstance(kw[key], str) and kw[key].strip().lower() in ("", "none")):
        return default
    try:
        return int(float(kw[key]))
    except (TypeError, ValueError):
        raise _err(f"{key}={kw[key]!r} is not an integer", f"{key}={kw[key]!r} 는 정수가 아닙니다") from None


def _float(kw: dict, key: str, default: float) -> float:
    if key not in kw or kw[key] is None or (isinstance(kw[key], str) and kw[key].strip() == ""):
        return default
    try:
        return float(kw[key])
    except (TypeError, ValueError):
        raise _err(f"{key}={kw[key]!r} is not a number", f"{key}={kw[key]!r} 는 숫자가 아닙니다") from None


# ======================================================================================================================
# configuration (network_args)
# ======================================================================================================================


@dataclass(frozen=True)
class TsmLokrConfig:
    n_scales: Tuple[int, ...] = (8, 1)
    num_timesteps: int = 1000
    stage: int = 1
    router_input: str = "input"
    gate_init: float = 0.0
    core_factor: int = -1
    context_factor: int = -1
    full_matrix: bool = False
    decompose_both: bool = False
    dora: bool = False
    wd_on_output: bool = True
    offload_states: bool = True
    resident_cores: int = 8
    grouped_accumulation: bool = False

    @property
    def n_core(self) -> int:
        return int(self.n_scales[0])

    def core_index(self, t: int) -> int:
        """Core expert of integer timestep t (tsm.py:297-315)."""
        n = self.n_core
        return max(1, min(math.ceil(t / self.num_timesteps * n), n)) - 1

    def core_index_from_timestep(self, t: float, max_timestep: float = 1000.0) -> int:
        """Core expert of a trainer timestep (sigma * 1000, any float dtype): set_timestep's conversion + core_index."""
        v = int(round(float(t) * self.num_timesteps / float(max_timestep)))
        return self.core_index(max(0, min(self.num_timesteps, v)))


# Knobs LyCORIS would pass to (or apply around) the module that TSM-LoKr v1 does not implement: refused when set to a
# value that does something, instead of being ignored.
_UNSUPPORTED_TRUE = ("bypass_mode", "use_scalar", "use_tucker", "use_cp", "use_conv_cp", "unbalanced_factorization",
                     "rs_lora", "orthogonalize", "orthogonal_init", "nora", "nora_init", "torch_compile",
                     "weight_noise_dynamic_sigma")
_UNSUPPORTED_SET = ("svd_segment", "init_weights", "ggpo_beta", "ggpo_sigma", "weight_noise_sigma")
_UNSUPPORTED_POSITIVE = ("dropout", "rank_dropout", "module_dropout")


def parse_config(kw: dict) -> TsmLokrConfig:
    """Validated TsmLokrConfig from the raw network_args (strings). Raises ValueError (English / Korean)."""
    for key in _UNSUPPORTED_TRUE:
        if _bool(kw, key, False):
            raise _err(f"{key}=True is not supported", f"{key}=True 는 지원하지 않습니다")
    if "disable_conv_cp" in kw and not _bool(kw, "disable_conv_cp", True):
        raise _err("disable_conv_cp=False (Tucker) is not supported", "disable_conv_cp=False(Tucker) 는 지원하지 않습니다")
    for key in _UNSUPPORTED_SET:
        v = kw.get(key)
        if v is not None and not (isinstance(v, str) and v.strip().lower() in ("", "none")):
            raise _err(f"{key} is not supported", f"{key} 는 지원하지 않습니다")
    for key in _UNSUPPORTED_POSITIVE:
        if _float(kw, key, 0.0) > 0:
            raise _err(f"{key} > 0 is not supported (LoKr experts have no dropout)",
                       f"{key} > 0 은 지원하지 않습니다(LoKr 전문가에는 드롭아웃이 없습니다)")

    raw_scales = kw.get("tsm_n_scales", None)
    if raw_scales is None or (isinstance(raw_scales, str) and raw_scales.strip() == ""):
        n_scales = (8, 1)
    elif isinstance(raw_scales, str):
        try:
            n_scales = tuple(int(s.strip()) for s in raw_scales.strip("[]() ").split(",") if s.strip())
        except ValueError:
            raise _err(f"tsm_n_scales={raw_scales!r}: expected integers like 8,1",
                       f"tsm_n_scales={raw_scales!r}: 8,1 처럼 정수를 쉼표로 적어야 합니다") from None
    else:
        n_scales = tuple(int(s) for s in raw_scales)
    if len(n_scales) < 2 or any(n < 1 for n in n_scales):
        raise _err(f"tsm_n_scales={list(n_scales)}: needs a core scale and at least one context scale, each >= 1",
                   f"tsm_n_scales={list(n_scales)}: 코어 1개와 문맥 1개 이상, 각각 1 이상이어야 합니다")
    T = _int(kw, "tsm_num_timesteps", 1000)
    if T < 1:
        raise _err("tsm_num_timesteps must be >= 1", "tsm_num_timesteps 는 1 이상이어야 합니다")
    stage = _int(kw, "tsm_stage", 1)
    if stage not in (1, 2):
        raise _err(f"tsm_stage={stage}: use 1 or 2", f"tsm_stage={stage}: 1 또는 2 만 쓸 수 있습니다")
    router_input = str(kw.get("tsm_router_input", "input") or "input").strip().lower()
    if router_input != "input":
        raise _err(f"tsm_router_input={router_input} has no meaning for full-matrix LoKr experts; use input",
                   "tsm-lokr 에서는 tsm_router_input=input 만 쓸 수 있습니다")
    factor = _int(kw, "factor", -1)
    core_factor = _int(kw, "tsm_core_factor", factor)
    context_factor = _int(kw, "tsm_context_factor", factor)
    for name, f in (("factor", factor), ("tsm_core_factor", core_factor), ("tsm_context_factor", context_factor)):
        if f == 0 or f < -1:
            raise _err(f"{name}={f}: use -1 or a positive factor", f"{name}={f}: -1 또는 양수만 쓸 수 있습니다")
    dora = _bool(kw, "dora_wd", False)
    wd_on_output = _bool(kw, "wd_on_output", True)
    if dora and not wd_on_output:
        raise _err("wd_on_output=False is not supported yet", "아직 wd_on_output=True 만 지원합니다")
    n_core = n_scales[0]
    resident = _int(kw, "tsm_resident_cores", n_core)
    if not 1 <= resident <= n_core:
        raise _err(f"tsm_resident_cores={resident}: use 1..{n_core}", f"tsm_resident_cores={resident}: 1..{n_core} 만 쓸 수 있습니다")
    return TsmLokrConfig(
        n_scales=n_scales,
        num_timesteps=T,
        stage=stage,
        router_input=router_input,
        gate_init=_float(kw, "tsm_gate_init", 0.0),
        core_factor=core_factor,
        context_factor=context_factor,
        full_matrix=_bool(kw, "full_matrix", False),
        decompose_both=_bool(kw, "decompose_both", False),
        dora=dora,
        wd_on_output=wd_on_output,
        offload_states=_bool(kw, "tsm_offload_states", True),
        resident_cores=resident,
        grouped_accumulation=_bool(kw, "tsm_grouped_accumulation", False),
    )


# kohya.py passes only its own keyword list to the modules (kohya.py:419-507, :953-956): the parsed config reaches
# TSMLokrModule through this module-level slot, set by lycoris_algo_fixes.prepare_network_kwargs.
ACTIVE_CONFIG: Optional[TsmLokrConfig] = None


def set_active_config(cfg: Optional[TsmLokrConfig]) -> None:
    global ACTIVE_CONFIG
    ACTIVE_CONFIG = cfg


# ======================================================================================================================
# math
# ======================================================================================================================


def kron4(w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """torch.kron(w1 [a, c], w2 [b, d]) -> [a*b, c*d] as one broadcast multiply (bit-identical, fuses under compile)."""
    a, c = w1.shape
    b, d = w2.shape
    return (w1[:, None, :, None] * w2[None, :, None, :]).reshape(a * b, c * d)


def kron_mv(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """x @ (scale * kron(w1, w2))^T without building the Kronecker product: [..., c*d] -> [..., a*b]."""
    a, c = w1.shape
    b, d = w2.shape
    lead = x.shape[:-1]
    X = x.reshape(-1, c, d)
    Z = torch.matmul(w1, torch.matmul(X, w2.t()))  # [N, a, b]: y[i*b+k] = sum_j w1[i,j] sum_l w2[k,l] x[j*d+l]
    out = Z.reshape(*lead, a * b)
    return out * scale if scale != 1.0 else out


def _rebuild_tsm_lokr(W0, m, eps, mu, out_dtype, scales, *factors):
    """Stage-1 merged weight W0 + mu * (D(W0 + sum_e s_e kron(w1_e, w2_e)) - W0), cast to out_dtype.

    W0 bf16/fp32 base weight, factors fp32: W0 + delta is the exact upcast LyCORIS' LoKr uses (lokr.py:640-645)."""
    delta = None
    for k in range(len(scales)):
        term = kron4(factors[2 * k], factors[2 * k + 1])
        if scales[k] != 1.0:
            term = term * scales[k]
        delta = term if delta is None else delta + term
    V = W0 + delta
    if m is not None:
        D = V * (m / (torch.linalg.vector_norm(V, dim=1, keepdim=True) + eps))
    else:
        D = V
    if mu != 1.0:
        D = W0 + mu * (D - W0)
    return D.to(out_dtype)


_TSM_REBUILD = ElementwiseChains("TSM-LoKr+DoRA weight rebuild", rebuild=_rebuild_tsm_lokr)


def enable_compiled_rebuild(enabled: bool) -> None:
    _TSM_REBUILD.enable(enabled)


def row_norm_terms(W0: torch.Tensor, C: torch.Tensor, Xs: Sequence[torch.Tensor]):
    """Stage-2 row-norm terms (fp32): a = ||W0+C||^2_row [out], b_j = <W0+C, X_j>_row [n, out],
    c_jk = <X_j, X_k>_row [n, n, out], so ||W0 + C + sum_j g_j X_j||^2_row = a + 2 sum g_j b_j + sum g_j g_k c_jk."""
    dt = torch.promote_types(torch.promote_types(W0.dtype, C.dtype), torch.float32)  # fp32 (fp64 in fp64 tests)
    M = W0.to(dt) + C.to(dt)
    Xs = [X.to(dt) for X in Xs]
    a = (M * M).sum(1)
    b = torch.stack([(M * X).sum(1) for X in Xs])
    c = torch.stack([torch.stack([(Xj * Xk).sum(1) for Xk in Xs]) for Xj in Xs])
    return a, b, c


# ======================================================================================================================
# module
# ======================================================================================================================


class LokrExpert(nn.Module):
    """One LoKr expert: LyCORIS' parameter names and its full/low-rank and scale rules (lokr.py:204-270, Linear)."""

    def __init__(self, out_dim: int, in_dim: int, factor: int, lora_dim: int, alpha: float, decompose_both: bool,
                 full_matrix: bool, lora_name: str = ""):
        super().__init__()
        from lycoris.modules.lokr import factorization_with_warning

        a, b = factorization_with_warning(out_dim, factor, lora_name, "output")
        c, d = factorization_with_warning(in_dim, factor, lora_name, "input")
        self.factor_shape = ((a, b), (c, d))
        self.use_w1 = not (decompose_both and lora_dim < max(a, c) / 2 and not full_matrix)
        self.use_w2 = bool(full_matrix or lora_dim >= max(b, d) / 2)
        if self.use_w1:
            self.lokr_w1 = nn.Parameter(torch.empty(a, c))
            nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            self.lokr_w1_a = nn.Parameter(torch.empty(a, lora_dim))
            self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, c))
            nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))
        if self.use_w2:
            self.lokr_w2 = nn.Parameter(torch.zeros(b, d))  # LyCORIS: constant 0 without use_scalar
        else:
            self.lokr_w2_a = nn.Parameter(torch.empty(b, lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.zeros(lora_dim, d))
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        # LyCORIS: scale 1 when both factors are full (alpha := lora_dim), else alpha / lora_dim
        self.scale = 1.0 if (self.use_w1 and self.use_w2) else float(alpha) / float(lora_dim)

    def factors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w1 = self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b
        return w1, w2


def _bilingual_not_implemented(what: str) -> NotImplementedError:
    return NotImplementedError(
        f"tsm-lokr: {what} has no single static weight (the weight depends on the timestep)"
        f" / tsm-lokr: {what} 는 타임스텝마다 가중치가 달라 하나의 고정 가중치로 만들 수 없습니다"
    )


class TSMLokrModule:  # replaced below by _define_module() (LycorisBaseModule is imported lazily)
    pass


def _define_module():
    from lycoris.modules.base import LycorisBaseModule
    from lycoris.modules.tsm import get_tsm_timestep

    class _TSMLokrModule(LycorisBaseModule):
        name = ALGO
        support_module = {"linear"}
        weight_list = []
        weight_list_det = []
        _tsm_family = True

        def __init__(self, lora_name, org_module: nn.Module, multiplier=1.0, lora_dim=4, alpha=1, dropout=0.0,
                     rank_dropout=0.0, module_dropout=0.0, use_tucker=False, **kwargs):
            cfg = ACTIVE_CONFIG
            if cfg is None:
                raise ValueError(
                    "tsm-lokr modules are built only by the fixed LyCORIS paths (library/lycoris_algo_fixes.py, the"
                    " default; not with --lycoris_stock) with network_args algo=tsm-lokr"
                    " / tsm-lokr 모듈은 고친 LyCORIS 경로(library/lycoris_algo_fixes.py, 기본값; --lycoris_stock 이면 안 됨)와"
                    " algo=tsm-lokr 로만 만들 수 있습니다"
                )
            super().__init__(lora_name, org_module, multiplier, 0.0, 0.0, 0.0, False, False)
            if self.module_type != "linear":
                raise ValueError(f"{lora_name}: tsm-lokr supports Linear layers only ({self.module_type})"
                                 f" / tsm-lokr 는 Linear 레이어만 지원합니다")
            self.cfg = cfg
            out_dim, in_dim = int(org_module.out_features), int(org_module.in_features)
            self.in_dim, self.out_dim, self.isconv = in_dim, out_dim, False
            self.tsm_n_scales = list(cfg.n_scales)
            self.tsm_num_timesteps = int(cfg.num_timesteps)
            self.tsm_router_input = "input"
            self.mask_group_id = 0
            self.lora_dim = int(lora_dim)
            if isinstance(alpha, torch.Tensor):
                alpha = float(alpha.detach().float())
            alpha = float(lora_dim) if alpha is None or alpha == 0 else float(alpha)
            self.experts = nn.ModuleList()
            for s, n in enumerate(self.tsm_n_scales):
                f = cfg.core_factor if s == 0 else cfg.context_factor
                self.experts.append(nn.ModuleList(
                    LokrExpert(out_dim, in_dim, f, self.lora_dim, alpha, cfg.decompose_both, cfg.full_matrix, lora_name)
                    for _ in range(n)
                ))
            all_full = all(e.use_w1 and e.use_w2 for sc in self.experts for e in sc)
            # as LyCORIS LoKr stores it (lokr.py:262-270): lora_dim when both factors are full
            self.register_buffer("alpha", torch.tensor(float(lora_dim) if all_full else alpha))
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
            n_ctx = len(self.tsm_n_scales) - 1
            self.router_fc = nn.Linear(in_dim, n_ctx, bias=True)
            self.timestep_embed = nn.Embedding(self.tsm_num_timesteps, n_ctx)
            nn.init.zeros_(self.router_fc.weight)
            nn.init.zeros_(self.router_fc.bias)
            nn.init.zeros_(self.timestep_embed.weight)
            self.wd = bool(cfg.dora)
            self.wd_on_output = True
            if self.wd:
                w = org_module.weight.detach().cpu().clone().float()
                self._dora_norm_dims = (1,)
                self.register_buffer("_dora_eps", torch.tensor(torch.finfo(torch.float32).eps), persistent=False)
                self.dora_scale = nn.Parameter(torch.linalg.vector_norm(w, dim=1, keepdim=True))
                self.dora_scale._is_dora_scale = True
            else:
                self.dora_scale = None
            self._idx_cache: Dict[int, Tuple[int, ...]] = {}
            self._norm_cache: Dict[tuple, tuple] = {}
            self._weights_version = 0
            self._warned_no_t = False
            self.set_stage(cfg.stage)
            if cfg.stage == 2 and cfg.gate_init != 0.0:
                with torch.no_grad():
                    self.router_fc.bias.fill_(cfg.gate_init)

        # ------------------------------------------------------------------------------------------------ stage

        def set_stage(self, stage: int) -> None:
            self.tsm_stage = int(stage)
            train_experts = self.tsm_stage == 1
            for sc in self.experts:
                for e in sc:
                    for p in e.parameters():
                        p.requires_grad_(train_experts)
            if self.dora_scale is not None:
                self.dora_scale.requires_grad_(train_experts)
            self.router_fc.requires_grad_(not train_experts)
            self.timestep_embed.requires_grad_(not train_experts)

        def load_weight_hook(self, module, incompatible_keys):
            self._weights_version += 1
            self._norm_cache.clear()
            if self.tsm_stage == 2 and self.cfg.gate_init != 0.0:
                with torch.no_grad():
                    if not (torch.any(self.router_fc.weight != 0) or torch.any(self.router_fc.bias != 0)
                            or torch.any(self.timestep_embed.weight != 0)):
                        self.router_fc.bias.fill_(self.cfg.gate_init)  # stage 2 starts at the stage-1 model

        # ------------------------------------------------------------------------------------------- timesteps

        def _expert_indices(self, t: int) -> Tuple[int, ...]:
            idx = self._idx_cache.get(t)
            if idx is None:
                T = self.tsm_num_timesteps
                idx = tuple(max(1, min(math.ceil(t / T * n), n)) - 1 for n in self.tsm_n_scales)
                if len(self._idx_cache) > 4096:
                    self._idx_cache.clear()
                self._idx_cache[t] = idx
            return idx

        def _timesteps(self) -> Tuple[int, ...]:
            from library import lycoris_algo_fixes as laf

            ints = laf._TSM_SAMPLE_T.get(self.mask_group_id)
            if ints:
                return tuple(ints)
            t = get_tsm_timestep(self.mask_group_id)
            if t is None:
                if self.training and torch.is_grad_enabled():
                    raise RuntimeError(
                        f"{self.lora_name}: tsm-lokr forward without a timestep; the trainer must call"
                        " lycoris_algo_fixes.set_timestep (NetworkTrainer.apply_tlora_mask) before the forward"
                        " / tsm-lokr: 타임스텝 없이 학습 forward 가 호출됐습니다(set_timestep 훅이 필요합니다)"
                    )
                if not self._warned_no_t:
                    self._warned_no_t = True
                    logger.warning(f"tsm-lokr {self.lora_name}: no timestep set; using t = T (highest-noise experts)")
                t = self.tsm_num_timesteps
            return (int(t),)

        # ------------------------------------------------------------------------------------------- forward

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            mu = self.multiplier
            if mu == 0.0:
                return self.org_forward(x, *args, **kwargs)
            ints = self._timesteps()
            device_type = x.device.type
            autocast = torch.is_autocast_enabled(device_type)
            if autocast:
                out_dtype = torch.get_autocast_dtype(device_type)
            else:
                out_dtype = self._cached_dtype
                x = x.to(self._cached_dtype)  # as LyCORIS LoKr's forward (lokr.py:850)
            W0 = self.get_org_weight_for_compute(x.device).detach()
            bias = self.get_org_bias_for_compute(x.device)
            if bias is not None:
                bias = bias.to(x.dtype)
            if len(ints) > 1:
                if x.dim() < 2 or x.shape[0] != len(ints):
                    raise RuntimeError(
                        f"{self.lora_name}: {len(ints)} timesteps for a batch of {x.shape[0] if x.dim() else 0} rows"
                        f" / 타임스텝 {len(ints)}개와 배치 행 수가 다릅니다"
                    )
                groups: Dict[Tuple[int, ...], List[int]] = {}
                for row, t in enumerate(ints):
                    groups.setdefault(self._expert_indices(t), []).append(row)
            else:
                groups = {self._expert_indices(ints[0]): None}
            outs, order = [], []
            for idx, rows in groups.items():
                if rows is None:
                    xs, ts = x, [ints[0]] * (x.shape[0] if x.dim() >= 2 else 1)
                elif len(groups) == 1:
                    xs, ts = x, [ints[r] for r in rows]
                elif rows == list(range(rows[0], rows[0] + len(rows))):
                    xs, ts = x.narrow(0, rows[0], len(rows)), [ints[r] for r in rows]
                else:
                    xs = x.index_select(0, torch.tensor(rows, dtype=torch.long).to(x.device))
                    ts = [ints[r] for r in rows]
                if self.tsm_stage == 1:
                    outs.append(self._stage1(xs, W0, bias, idx, mu, out_dtype))
                else:
                    outs.append(self._stage2(xs, W0, bias, idx, ts, mu, out_dtype))
                if rows is not None:
                    order.extend(rows)
            if len(outs) == 1:
                return outs[0]
            out = torch.cat(outs, 0)
            inv = [0] * len(order)
            for pos, r in enumerate(order):
                inv[r] = pos
            if inv != list(range(len(inv))):
                out = out.index_select(0, torch.tensor(inv, dtype=torch.long).to(out.device))
            return out

        def _stage1(self, x, W0, bias, idx, mu, out_dtype):
            factors, scales = [], []
            for s, e in enumerate(idx):
                ex = self.experts[s][e]
                factors.extend(ex.factors())
                scales.append(ex.scale)
            m = self.dora_scale if self.wd else None
            eps = self._dora_eps if self.wd else None
            W = _TSM_REBUILD.get(x)["rebuild"](W0, m, eps, float(mu), out_dtype, tuple(scales), *factors)
            return F.linear(x, W, bias)

        def _norm_terms(self, idx, W0):
            key = (idx, W0.device, self._weights_version)
            terms = self._norm_cache.get(key)
            if terms is None:
                with torch.no_grad():
                    core = self.experts[0][idx[0]]
                    C = kron4(*core.factors()) * core.scale
                    Xs = []
                    for s in range(1, len(idx)):
                        ex = self.experts[s][idx[s]]
                        Xs.append(kron4(*ex.factors()) * ex.scale)
                    terms = row_norm_terms(W0, C, Xs)
                if len(self._norm_cache) > 4 * max(1, self.tsm_n_scales[0]):
                    self._norm_cache.clear()
                self._norm_cache[key] = terms
            return terms

        def _gates(self, x, ts):
            B = x.shape[0]
            acc = torch.promote_types(x.dtype, torch.float32)
            z = x.detach().reshape(B, -1, self.in_dim).mean(1, dtype=acc)  # per-sample pooling, fp32 accumulation
            with torch.autocast(x.device.type, enabled=False):
                w = self.router_fc.weight
                g = F.linear(z.to(w.dtype), w, self.router_fc.bias)
                E = self.timestep_embed.weight
                T = self.tsm_num_timesteps
                rows = [min(max(int(t) - 1, 0), T - 1) for t in ts]
                if all(r == rows[0] for r in rows):
                    emb = E[rows[0]].unsqueeze(0)  # Python-int index: no host sync
                else:
                    emb = torch.stack([E[r] for r in rows])
                return g + emb.to(g.dtype)  # [B, n_ctx] fp32

        def _stage2(self, x, W0, bias, idx, ts, mu, out_dtype):
            core = self.experts[0][idx[0]]
            ctx = [self.experts[s][idx[s]] for s in range(1, len(idx))]
            with torch.no_grad():
                w1c, w2c = core.factors()
                What = (W0 + kron4(w1c, w2c) * core.scale if core.scale != 1.0 else W0 + kron4(w1c, w2c)).to(out_dtype)
            y1 = F.linear(x, What)
            y2 = [kron_mv(x, *ex.factors(), ex.scale) for ex in ctx]
            g = self._gates(x, ts)  # [B, n_ctx]
            bshape = (x.shape[0],) + (1,) * (x.dim() - 1)
            mix = y1
            for j, yj in enumerate(y2):
                mix = mix + g[:, j].view(bshape) * yj
            if self.wd:
                a, b, c = self._norm_terms(idx, W0)
                with torch.autocast(x.device.type, enabled=False):  # fp32 row norms (no bf16 matmul)
                    n2 = a.unsqueeze(0) + 2.0 * (g @ b)  # [B, out]
                    n2 = n2 + torch.einsum("bj,bk,jko->bo", g, g, c)
                    s = self.dora_scale.view(1, -1).to(n2.dtype) / (torch.sqrt(n2) + self._dora_eps)
                D = mix * s.view((x.shape[0],) + (1,) * (x.dim() - 2) + (self.out_dim,))
            else:
                D = mix
            if mu != 1.0:
                base = y1 - kron_mv(x, w1c, w2c, core.scale)  # W0 x = (W0 + C) x - C x
                D = base + mu * (D - base)
            D = D.to(y1.dtype)
            if bias is not None:
                D = D + bias
            return D

        # ------------------------------------------------------------------------------- static weight API

        def get_diff_weight(self, multiplier=1, shape=None, device=None):
            raise _bilingual_not_implemented("get_diff_weight")

        def get_merged_weight(self, multiplier=1, shape=None, device=None):
            raise _bilingual_not_implemented("get_merged_weight")

        def merge_to(self, multiplier=1.0):
            raise _bilingual_not_implemented("merge_to")

        @torch.no_grad()
        def apply_max_norm(self, max_norm, device=None):
            raise _bilingual_not_implemented("apply_max_norm (scale_weight_norms)")

        @classmethod
        def make_module_from_state_dict(cls, lora_name, orig_module, *weights):
            raise _bilingual_not_implemented("make_module_from_state_dict (dim_from_weights / merge tools)")

        def bypass_forward(self, x, scale=1):
            raise _bilingual_not_implemented("bypass_forward")

    _TSMLokrModule.__name__ = "TSMLokrModule"
    _TSMLokrModule.__qualname__ = "TSMLokrModule"
    return _TSMLokrModule


def get_module_class():
    """The TSMLokrModule class (a LycorisBaseModule subclass, built on first use)."""
    global TSMLokrModule
    if not isinstance(TSMLokrModule, type) or not getattr(TSMLokrModule, "_tsm_family", False):
        TSMLokrModule = _define_module()
    return TSMLokrModule


def is_tsm_lokr(module) -> bool:
    return type(module).__name__ == "TSMLokrModule" and getattr(module, "_tsm_family", False)


# ======================================================================================================================
# metadata / summary
# ======================================================================================================================

_DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}


def dtype_name(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    return _DTYPE_NAMES.get(dtype, "fp32")


def metadata_json(module, timestep_dtype: str) -> str:
    cfg: TsmLokrConfig = module.cfg
    return json.dumps({
        "version": FORMAT_VERSION,
        "n_scales": list(cfg.n_scales),
        "num_timesteps": int(cfg.num_timesteps),
        "stage": int(module.tsm_stage),
        "router_input": "input",
        "factors": [int(cfg.core_factor), int(cfg.context_factor)],
        "full_matrix": bool(cfg.full_matrix),
        "decompose_both": bool(cfg.decompose_both),
        "dora": "out" if cfg.dora else "none",
        "timestep_dtype": timestep_dtype,
        "timestep_scale": 1000,
        "pooling": "sample",
        "multiplier": "forge-blend",
        "gate_init": float(cfg.gate_init),
    }, separators=(",", ":"))


def size_summary(modules: Sequence[nn.Module]) -> dict:
    """Parameter counts and byte sizes (fp32 weights, AdamW states, fp16 file) per expert kind."""
    if not modules:
        return {}
    cfg = modules[0].cfg
    core = sum(p.numel() for m in modules for p in m.experts[0][0].parameters())
    ctx = sum(p.numel() for m in modules for s in range(1, len(m.experts)) for p in m.experts[s][0].parameters())
    total = sum(p.numel() for m in modules for p in m.parameters())
    gib = 1 / 2 ** 30
    return {
        "modules": len(modules),
        "core_expert_params": core,
        "context_expert_params": ctx,
        "total_params": total,
        "weights_fp32_gib": round(total * 4 * gib, 3),
        "core_state_gib_each": round(core * 8 * gib, 3),
        "file_fp16_gib": round(total * 2 * gib, 3),
        "save_state_gib": round(total * 12 * gib, 3),
    }


def as_dict(cfg: TsmLokrConfig) -> dict:
    return asdict(cfg)
