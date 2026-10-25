# LyCORIS fixed/optimised code paths: in-memory fixes (and faster, same-math forwards) for LyCORIS algorithms that are
# broken or slow on this fork's trainers (site-packages untouched). Importing this module patches nothing; the trainer
# (train_network.py _lycoris_fix_setup) turns them on BY DEFAULT for network_module=lycoris.kohya.
#   --lycoris_stock (config lycoris_stock = true): LyCORIS exactly as installed; this module is not imported at all.
#   --lycoris_algo_fixes (the old opt-in flag, config lycoris_algo_fixes = true): still accepted, changes nothing;
#             with both, --lycoris_stock wins. Other network modules (networks.lora, ...) are never touched. Every
#             trainer built on NetworkTrainer.train() (train_network.py: Anima incl. anima_train_addift.py, MOSS, SDXL,
#             FLUX, SD3, Lumina, HunyuanImage) gets the fixed paths. Scripts that build or merge LyCORIS networks
#             themselves keep stock LyCORIS: the LECO trainers (anima_train_leco.py, sdxl_train_leco.py,
#             train_leco.py), gen_img / merge tools, and train_network.py's own --base_weights merge (it runs before
#             the patches are applied).
# "The flag" / "with the flag" below means these fixed paths (the default); "stock" / "without the flag" means LyCORIS
# as installed (--lycoris_stock). Saved files carry ss_lycoris_algo_fixes = "1", ss_saved_format, ss_lycoris_algo.
#
# Covered algos (FIX_ALGOS) and what goes wrong in stock LyCORIS (3.x as installed in the venv, --lycoris_stock):
#   loha      get_diff_weight applies alpha/dim twice (loha.py:349-350: merges/merge tools get scale^2);
#             make_module_from_state_dict ignores wd_on_output (loha.py:256-264); bypass dtype mismatch under bf16.
#   abba      svd_lowrank(q=dim//2) crashes when dim//2 > min(out, in) (abba.py:216); get_merged_weight applies the
#             multiplier twice (abba.py:371); bypass caches the Khatri-Rao factors once (abba.py:386-388: stale weights,
#             no gradient after a no-grad forward, or "backward through the graph a second time"); no Forge adapter.
#   gora      init divided by the step count twice (gora_utils.py:549 and :791), base-weight .grad kept for the whole run
#             (the hook returns the gradient, gora_utils.py:58/:785), string min/max ranks (kohya.py:304-305).
#   ralora    the pre-pass calls forward_fn(model, batch) (ralora_utils.py:542) inside @torch.no_grad (locon.py:1604, so
#             loss.backward() at :551 has no grad_fn); base weights keep requires_grad (ralora_utils.py:516); block
#             layers' new ParameterLists are created on the CPU; RaLoRA-Pro saves alpha != the trained scale
#             (locon.py:1811-1823); Forge cannot read the block keys (locon.py:2018-2043).
#   lora2     string lora2_nu_target (kohya.py:400 -> lora2.py:144); the save cuts the tensors to rank D (lora2.py:
#             372-373, resume/reload breaks) and leaves Lambda out (Forge applies the LoRA ~10x too strong).
#   diag-oft, boft   bf16 dtype crash where autocast is off (Anima adaLN; diag_oft.py:302, boft.py:358). Speed: rebuild
#             mode builds the rotated weight per call and runs a dW' GEMM in backward, the Cayley inverse syncs with
#             the host; on Linear layers the flag rotates the base output blockwise instead (same math, no dW').
#   tlora     bf16 dtype crash (tlora.py:691-693, :636-637); no Forge adapter.
#   Speed (gora, ralora, lora2, tlora): rebuild mode builds the dense delta weight (out x in) per module per call,
#             adds it to an fp32 copy of the base weight (+ the DoRA norm over that) and runs the merged GEMM, whose
#             backward is a dense weight-gradient GEMM as large as the forward; T-LoRA runs a second full GEMM for its
#             delta. LoRA² reads nu with .item() per call (a host sync); T-LoRA copies the trainer's CPU timestep mask
#             to the device per call (a blocking copy). The flag runs the same function factored (see "LoRA family:
#             factored forward"): base GEMM + thin down/up matmuls, the DoRA row norms from the factors, the mask
#             copied once per step.
#   loha/abba speed: the rebuild forward builds the dense Hadamard delta on every call and needs the merged weight's
#             gradient; the flag uses the exact Khatri-Rao factorisation (x KA^T) KB^T where cheaper, else the stock ops
#             without the fp32 activation / base-weight copies and LoHa's per-call host sync ("LoHa / ABBA speed").
#   tsm       nothing sets the timestep; one int per batch; stage 2: the router never sees the input in rebuild mode
#             (tsm.py:538-545), gate pooling breaks on >3-D inputs (tsm.py:362-371); no Forge adapter. Speed: rebuild
#             mode builds a dense delta weight per module per call, bypass runs one matmul pair per expert; the flag
#             runs the base GEMM with the concatenated active experts' delta as its C input and a hand-written backward
#             (_LowRankLinear / _TSMLinear on the concatenated factors; a mixed-timestep chunk as ONE masked call per
#             module, _TSMMaskedLinear, instead of one per expert group; network_args tsm_fused=False keeps org_forward +
#             a separate delta, tsm_mixed=False the per-group loop, tsm_lowrank_fp32 picks fp32 expert projections -
#             default stage 2 only, tsm_expert_fn=True the experimental one-Function path _TSMExpertLinear).
#   lora, locon   PiSSA (init_weights=pissa): saved at half strength (alpha kept for the doubled rank, locon.py:922-941),
#             with a pissa_converted key Forge does not read; --resume / --network_weights failed (locon.py:898-962);
#             with alpha != dim the base model changed at step 0 (locon.py:404-407). Speed: see "factored training
#             forward" (no merged weight and no weight-gradient matmul per call; not for DoRA).
#   ortholora the saved distilled LoRA (ortholora.py:323-353) drops dora_scale and is bf16; --resume and
#             --network_weights failed (missing S_p/S_q/bases, unexpected lora_up/lora_down). Speed: factored forward.
#   dylora    bf16 dtype crash (dylora.py:162-168); checkpoint recompute drew another random rank (Python random,
#             dylora.py:115); sampling/merges used a random rank; Forge strength 1/block_size (alpha / (b+1) blocks,
#             dylora.py:112); bypass ignored scale and multiplier (dylora.py:142-150); --resume / --network_weights
#             failed (lora_up/lora_down vs up_list/down_list); DoRA silently ignored.
#   lokr      DoRA at multiplier != 1 keeps the whole dW ((W0 + dW) (m (s - 1) + 1), lokr.py:651/:672/:851): multiplier
#             0 (differential output preservation, iLECO, ADDifT targets) is not the base model. The flag interpolates
#             W0 + m (W1 - W0) like Forge / sam-extra; multiplier 1 (every training step) goes to the installed forward
#             (the fork's fast LoKr+DoRA path, lycoris_fast.py) untouched.
#   glora     dora_wd silently ignored (glora.py:55); bypass applies alpha/rank twice, drops the learnable scalar and
#             puts the base bias into the diff (glora.py:292-354); rs_lora saves alpha, Forge reads alpha / rank
#             (glora.py:126-134). Speed: the rebuild backward forms the full weight gradient (a tokens x out x in matmul)
#             and differentiates the dense W A1 A2 chain; the flag keeps the stock forward and gets the factor
#             gradients from thin fp32 matmuls (same math, output and input gradient bit-identical).
#   ia3       dora_wd silently ignored; bypass scales the base bias (ia3.py:115-122); make_module_from_state_dict takes
#             (weight) but gets (weight, on_input) (TypeError in create_network_from_weights); preset=ia3 names SD
#             layers only (nothing on a DiT: the flag raises). No Forge adapter (kept native).
#   full      the trained tensor W0 + dW is zeros_like(W0) (full.py:83): bf16 on a bf16 base, so AdamW steps under half a
#             bf16 ulp of |W| are lost (lr 2e-5 on Anima 2.9B: 38-91% of the weights never move) -> fp32 master weights;
#             get_diff_weight applies the multiplier twice (full.py:244+250: merges, and the forward at multiplier != 0/1
#             as iLECO/ADDifT set it) and crashes on a bias (:232); rank_dropout crashes (:194-198), module_dropout calls
#             the base forward whose weight apply_to deleted (:139); a second load (--network_weights then --resume) adds
#             the file's dW to the current weight (:178-184). Saved "diff" keys are Forge's own format (diff patch).
#   tsm-lokr  not a LyCORIS algo: TSM routing with LoKr experts and a shared DoRA magnitude (library/tsm_lokr.py), added to
#             LyCORIS' algo table in memory; optional core-expert offload for training (library/tsm_offload.py).
# Every patch is listed in apply_module_patches(). The trainer side (train_network.py) calls, unless --lycoris_stock:
#   apply_module_patches() before the network is built, prepare_network_kwargs() on the network args,
#   after_network_created() right after create_network, set_timestep()/clear_timestep() around each step,
#   gora_ralora_allocation_state()/restore_gora_ralora_allocation() for --resume / --network_weights.
#
# Saved files (after_network_created patches network.save_weights): the .safetensors written for epochs/steps/the final
# model is Forge-loadable — abba -> LoHa keys; lora2 -> plain LoRA (Lambda baked into lora_down, rank D); ralora
# block-diagonal -> plain LoRA (block_diag up/down); tlora -> plain LoRA at the full mask (all ranks, t-independent);
# scale folded into the factors so alpha == rank (exact in fp16). gora, loha, diag-oft, boft, lokr are already Forge
# formats; glora too, with alpha = scale * rank (rs_lora). ortholora -> plain LoRA (+ dora_scale); PiSSA -> plain LoRA
# of rank 2r against the original base; dylora -> its LoRA keys with alpha * block_size; lora/locon stay as they are.
# The accelerate state of ortholora and PiSSA is their native form (native_save / pissa_convert=False), so --resume is
# exact. full -> Forge's "diff" keys (dW in the save dtype). tsm, tsm-lokr and ia3 stay native (Forge has no TSM / IA3
# adapter). The accelerate state (--save_state / --resume) keeps LyCORIS' own state_dict, so training resumes exactly.

from __future__ import annotations

import inspect
import json
import math
import os
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

FIX_ALGOS = {"loha", "abba", "gora", "ralora", "lora2", "diag-oft", "boft", "tlora", "tsm",
             "lora", "locon", "ortholora", "dylora", "lokr", "glora", "ia3", "full", "tsm-lokr"}

# DoRA (dora_wd=True): trained and kept in the Forge file (dora_scale [out,1,...] for wd_on_output=True, [1,in,...] for
# False — the axis Forge picks with dora_scale.shape[0] == out, weight_adapter/base.py:45).
DORA_SUPPORTED = {"loha", "abba", "gora", "ralora", "lora2", "lora", "locon", "ortholora", "lokr", "tsm-lokr"}
# The module classes swallow weight_decompose through **kwargs: no dora_scale is ever created (diag_oft.py:55,
# boft.py:70, tlora.py, tsm.py, dylora.py:35, glora.py:55 (accepted, never used), ia3.py:31, full.py:48), although
# kohya.py:263 logs "DoRA is enabled".
DORA_UNSUPPORTED = {"diag-oft", "boft", "tlora", "tsm", "dylora", "glora", "ia3", "full"}

METADATA_FLAG_KEY = "ss_lycoris_algo_fixes"
METADATA_FORMAT_KEY = "ss_saved_format"
METADATA_ALGO_KEY = "ss_lycoris_algo"
METADATA_ALLOC_KEY = "ss_lycoris_alloc"  # same key train_network.py reads for --network_weights
ALLOC_STATE_VERSION = 1

_TRUE = ("true", "1", "yes", "on")


def _str_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUE
    return bool(value)


def enabled(args) -> bool:
    """True when the fixed paths apply (train_network.py _lycoris_fix_setup): a LyCORIS network (network_module
    lycoris.*) without --lycoris_stock. The old --lycoris_algo_fixes flag no longer matters."""
    if getattr(args, "lycoris_stock", False):
        return False
    return str(getattr(args, "network_module", "") or "").startswith("lycoris")


# ======================================================================================================================
# monkeypatch bookkeeping
# ======================================================================================================================

_PATCHES: List[Tuple[Any, str, Any]] = []  # (owner, attribute, original object in owner.__dict__ or None)
_MAP_PATCHES: List[Tuple[dict, str]] = []  # (dict, key) entries added by _install_mapping
_PATCH_LOG: List[str] = []


def _install(owner, name: str, func: Callable, reason: str, kind: str = "function") -> bool:
    current = owner.__dict__.get(name)
    marker = getattr(current, "__func__", current)
    if getattr(marker, "_lycoris_algo_fix", False):
        return False  # already patched (idempotent)
    func._lycoris_algo_fix = True
    new = classmethod(func) if kind == "classmethod" else func
    _PATCHES.append((owner, name, current))
    setattr(owner, name, new)
    owner_name = getattr(owner, "__qualname__", None) or getattr(owner, "__name__", str(owner))
    _PATCH_LOG.append(f"{owner_name}.{name}: {reason}")
    return True


def _install_mapping(mapping: dict, key: str, value: Any, reason: str) -> bool:
    """mapping[key] = value if the key is free (idempotent); removed again by _revert_module_patches()."""
    if key in mapping:
        return False
    mapping[key] = value
    _MAP_PATCHES.append((mapping, key))
    _PATCH_LOG.append(f"network_module_dict[{key!r}]: {reason}")
    return True


def _original(owner, name: str):
    """The unpatched object (owner.__dict__ entry before our patch, or the inherited attribute)."""
    for o, n, orig in _PATCHES:
        if o is owner and n == name:
            return orig
    return owner.__dict__.get(name)


def _revert_module_patches() -> None:
    """Undo apply_module_patches() (tests; the trainer never reverts)."""
    while _PATCHES:
        owner, name, orig = _PATCHES.pop()
        if orig is None:
            try:
                delattr(owner, name)
            except AttributeError:
                pass
        else:
            setattr(owner, name, orig)
    while _MAP_PATCHES:
        mapping, key = _MAP_PATCHES.pop()
        mapping.pop(key, None)
    _PATCH_LOG.clear()


def patched_targets() -> List[str]:
    return list(_PATCH_LOG)


# ======================================================================================================================
# small helpers
# ======================================================================================================================


def _org_weight(module) -> torch.Tensor:
    return module.org_module[0].weight


def _all_loras(network) -> List[nn.Module]:
    seen, out = set(), []
    for attr in ("text_encoder_loras", "unet_loras", "loras"):
        for lora in getattr(network, attr, None) or []:
            if id(lora) not in seen:
                seen.add(id(lora))
                out.append(lora)
    return out


def _scalar_value(module) -> float:
    s = getattr(module, "scalar", None)
    if isinstance(s, torch.Tensor) and s.numel() == 1:
        return float(s.detach().float().cpu())
    return 1.0


def _out_dim(module) -> int:
    return int(module.shape[0])


def _cast_if_upcast(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """t.to(dtype) only when that is an exact upcast (bf16/fp16 -> fp32): never lowers precision."""
    if t.dtype != dtype and t.is_floating_point() and torch.promote_types(t.dtype, dtype) == dtype:
        return t.to(dtype)
    return t


# ======================================================================================================================
# LoHa
# ======================================================================================================================


def _loha_apply_scalars(self, w: torch.Tensor) -> torch.Tensor:
    """The scalar multiplications of LohaModule._forward_rebuild_core (loha.py:441-452), on w."""
    st = getattr(self, "scalar_type", "none")
    if st == "scalar":
        return w * self.scalar.to(w.device)
    if st == "row":
        return w * self.row_scalar.to(w.device).view(-1, *([1] * (w.dim() - 1)))
    if st == "column":
        return w * self.col_scalar.to(w.device).view(1, -1, *([1] * (w.dim() - 2)))
    if st == "row_column":
        r = self.row_scalar.to(w.device).view(-1, *([1] * (w.dim() - 1)))
        c = self.col_scalar.to(w.device).view(1, -1, *([1] * (w.dim() - 2)))
        return w * r * c
    return w


def _loha_get_diff_weight(self, multiplier=1, shape=None, device=None):
    # get_weight already applies alpha/dim (functional/loha.py gamma=scale); the stock version multiplies by
    # self.scale once more (loha.py:349-350). Same delta as the training forward (loha.py:437-456).
    target_shape = shape if shape is not None else self.shape
    diff = _loha_apply_scalars(self, self.get_weight(target_shape)) * multiplier
    if shape is not None:
        diff = diff.reshape(shape)
    if device is not None:
        diff = diff.to(device)
    return diff, None


def _wd_on_output_from(dora_scale, orig_module) -> bool:
    # LyCORIS: wd_on_output=True -> [out, 1, ...]; False -> [1, in, ...]. Forge decides the same way (base.py:45).
    if dora_scale is None:
        return True
    return int(dora_scale.shape[0]) == int(orig_module.weight.shape[0])


def _loha_make_module_from_state_dict(cls, lora_name, orig_module, w1a, w1b, w2a, w2b, t1, t2, alpha, dora_scale):
    module = cls(
        lora_name,
        orig_module,
        1,
        w1b.size(0),
        float(alpha),
        use_tucker=t1 is not None,
        weight_decompose=dora_scale is not None,
        wd_on_output=_wd_on_output_from(dora_scale, orig_module),
    )
    module.hada_w1_a.copy_(w1a)
    module.hada_w1_b.copy_(w1b)
    module.hada_w2_a.copy_(w2a)
    module.hada_w2_b.copy_(w2b)
    if t1 is not None:
        module.hada_t1.copy_(t1)
        module.hada_t2.copy_(t2)
    if dora_scale is not None:
        module.dora_scale.copy_(dora_scale)
    return module


def _bypass_dtype(x: torch.Tensor) -> torch.dtype:
    """The dtype F.linear(x, diff.to(x.dtype)) runs in: autocast's (it casts fp32/bf16/fp16), else x's."""
    dev = x.device.type
    if torch.is_autocast_enabled(dev) and x.dtype in (torch.float32, torch.bfloat16, torch.float16):
        return torch.get_autocast_dtype(dev)
    return x.dtype


def _loha_bypass_forward_diff(self, x, scale=1):
    plain = (self.module_type == "linear" and not (self.training and self.rank_dropout)
             and not getattr(self, "use_orthogonal_weights", False))
    if plain:
        dt = _bypass_dtype(x)
        rank = self.hada_w1_a.shape[1] * self.hada_w2_a.shape[1]
        if _hada_factored_cheaper(self, x, "loha", rank, False, self.hada_w1_a.dtype != dt, bypass=True):
            # (x KA^T) KB^T == x dW^T without the dense delta (see "LoHa / ABBA speed"), in the dtype F.linear runs in
            kb, ka = _loha_kr(self, scale)
            return self.drop(_HadaDelta.apply(x.to(dt), ka, kb, None, kb.dtype))
    # loha.py:390-405 plus the cast: the fp32 diff weight met a bf16 input where autocast is off (Anima adaLN).
    diff_weight = _loha_apply_scalars(self, _loha_weight(self) if plain else self.get_weight(self.shape)) * scale
    return self.drop(self._call_op(x, diff_weight.to(x.dtype)))


# ======================================================================================================================
# ABBA
# ======================================================================================================================


def _abba_initialize_weights(self):
    orig = _original(_abba_cls(), "initialize_weights")
    if self.r1 == 0 or self.r2 == 0:
        return orig(self)
    W0 = self.org_module[0].weight.data
    w2d_shape = (W0.size(0), W0[0].numel())
    q = min(self.r1, *w2d_shape)
    if q == self.r1:
        return orig(self)  # the stock path works: unchanged
    # svd_lowrank needs q <= min(out, in) (abba.py:216 crashed). Pair 1 = the best rank-q approximation of W0, the extra
    # r1-q ranks start like LoRA (down Kaiming, up 0) — B1A1 is already exact at q = min(out, in).
    W = W0.clone().to(dtype=torch.float32).view(*w2d_shape)
    U, S, V = torch.svd_lowrank(W, q=q, niter=10)
    s = torch.sqrt(S)
    up1 = torch.zeros(w2d_shape[0], self.r1)
    up1[:, :q] = U * s.unsqueeze(0)
    down1 = torch.empty(self.r1, w2d_shape[1])
    torch.nn.init.kaiming_uniform_(down1, a=math.sqrt(5))
    down1[:q] = s.unsqueeze(1) * V.T
    self.lora_up1.weight.data.copy_(up1.view(self.lora_up1.weight.shape))
    self.lora_down1.weight.data.copy_(down1.view(self.lora_down1.weight.shape))
    torch.nn.init.kaiming_uniform_(self.lora_down2.weight, a=math.sqrt(5))
    torch.nn.init.constant_(self.lora_up2.weight, 0)
    logger.info(
        f"ABBA {self.lora_name}: rank {self.r1} > min(out, in) = {q}; SVD init uses rank {q}, the other {self.r1 - q}"
        " ranks of pair 1 start at zero"
    )


def _abba_get_merged_weight(self, multiplier=1, shape=None, device=None):
    diff, _ = self.get_diff_weight(multiplier, shape, device)  # already includes the multiplier (abba.py:355)
    weight = self.get_org_weight_for_compute(device).to(self.dtype, non_blocking=True)
    if self.wd:
        merged = self.apply_weight_decompose(weight + diff, multiplier)
    else:
        merged = weight + diff  # abba.py:371 multiplied by the multiplier a second time
    return merged, None


def _abba_bypass_forward_diff(self, x, scale=1):
    if self.r1 == 0 or self.r2 == 0:
        return 0
    current_scale = self.scale * scale
    if not self.isconv:
        # Rebuild the Khatri-Rao factors on every call (abba.py:386-388 cached them once: stale after each optimizer
        # step, no gradient path after a no-grad forward), used in the input dtype (bf16 where autocast is off).
        # scalar * scale * multiplier is folded into B1 (out x r1) instead of scaling the (tokens x out) output.
        kb, ka = _abba_kr(self, scale)
        dt = x.dtype
        return self.drop(F.linear(F.linear(x, ka.to(device=x.device, dtype=dt)), kb.to(device=x.device, dtype=dt)))
    diff_weight = self.make_weight(device=x.device) * current_scale
    # no bias here: org_forward already added it (abba.py:417-421 added it twice)
    return self.drop(self._call_op(x, diff_weight.to(x.dtype)))


def _abba_make_module_from_state_dict(cls, lora_name, orig_module, up1, down1, up2, down2, alpha, dora_scale):
    lora_dim = down1.size(0) + down2.size(0)
    module = cls(
        lora_name,
        orig_module,
        1.0,
        lora_dim=lora_dim,
        alpha=float(alpha),
        use_tucker=False,
        weight_decompose=dora_scale is not None,
        wd_on_output=_wd_on_output_from(dora_scale, orig_module),
    )
    module.lora_up1.weight.data.copy_(up1)
    module.lora_down1.weight.data.copy_(down1)
    module.lora_up2.weight.data.copy_(up2)
    module.lora_down2.weight.data.copy_(down2)
    if dora_scale is not None:
        module.dora_scale.copy_(dora_scale)
    return module


def _abba_cls():
    from lycoris.modules.abba import AbbaModule

    return AbbaModule


def _loha_cls():
    from lycoris.modules.loha import LohaModule

    return LohaModule


# ======================================================================================================================
# LoHa / ABBA speed: the Hadamard delta as a Khatri-Rao low-rank product
# ======================================================================================================================
# (B1 A1) * (B2 A2) == KB KA with KB[i, (a, b)] = B1[i, a] B2[i, b] (out x r1*r2) and KA[(a, b), j] = A1[a, j] A2[b, j]
# (r1*r2 x in): the delta has rank r1*r2 (LoHa dim 16: 256; ABBA dim 16: 8*8 = 64). The stock rebuild forward
# (bypass_mode=False) builds it densely on every call (again in the checkpoint recompute, and twice more in HadaWeight's
# backward), adds it to an fp32 copy of the base weight, runs F.linear on an fp32 copy of the activation (autocast casts
# both straight back to bf16) and its backward needs the gradient of the whole merged weight (one more tokens x in x out
# matmul); LoHa's get_weight also makes a blocking host-to-device copy per call (torch.tensor(scale, device=cuda),
# loha.py:306). With the flag, each linear rebuild-mode call takes the cheaper of two exact paths (_hada_factored_cheaper):
#   factored: x W0^T + (x KA^T) KB^T (same math, no dense delta, no merged-weight gradient), DoRA's norm of each row (or
#     column) of W0 + KB KA as ||W0_i||^2 + 2 (W0 KA^T)_i . KB_i + KB_i (KA KA^T) KB_i^T, DoRA's scale applied to the
#     output (wd_on_output) or the input instead of the weight; scale, scalars and multiplier folded into B1 (A1 for
#     column scalars). Rounding no coarser than the stock path (see _HadaDelta / _HadaDoRAOut).
#   dense: the stock float ops, bit-identical, minus the fp32 activation / base-weight copies and the host sync.
# The factored path wins for ABBA (rank r^2/4) and for LoHa at small token counts; LoHa (rank r^2, fp32 adapter gradients
# like its stock backward) at ~4k tokens and DoRA keeps the dense path for square layers.

_FLOATS = (torch.float32, torch.bfloat16, torch.float16, torch.float64)
# Path choice (_hada_factored_cheaper): both paths give the same result, this only picks the faster one, by a roofline
# estimate of what each adds to one training step (gradient checkpointing: forward twice) on an RTX 5090: matmul
# throughput (bf16 large / bf16 rank-sized / fp32) and memory bandwidth (FLOP/s, bytes/s), kernel launches. The byte
# counts per weight / activation / rank-sized element were taken from a per-op trace of both paths (scratch gpu_model.py).
_MM_BF16, _MM_BF16_THIN, _MM_FP32, _BW, _LAUNCH = 150e12, 100e12, 60e12, 1.4e12, 4e-6
_DENSE_BYTES = {("loha", False): 125.0, ("loha", True): 185.0, ("abba", False): 140.0, ("abba", True): 224.0}


def _hada_factored_cheaper(self, x: torch.Tensor, algo: str, rank: int, wd: bool, fp32_grads: bool,
                           bypass: bool = False) -> bool:
    """Factored adds: the rank-sized matmuls (forward x2, backward; in fp32 for fp32_grads, plus the mid recompute),
    DoRA's W0 KA^T (forward x2, backward), activation- and rank-sized passes, DoRA's extra kernels. Dense adds: the merged
    weight's gradient (bypass: the delta is a full matmul, 4 of them) and the dense passes. N = tokens of this call."""
    out_f, in_f = int(self.shape[0]), int(self.shape[1])
    n = x.numel() // in_f
    nr = float(n) * rank
    thin = 4.0 * nr * (in_f + out_f) + 2.0 * nr * in_f
    nbytes = 4.0 * n * (in_f + out_f) + 60.0 * rank * (in_f + out_f)
    t = 0.0
    if fp32_grads:
        t += 4.0 * nr * (in_f + out_f) / _MM_FP32
        nbytes += 6.0 * n * (in_f + out_f)
    else:
        thin += 4.0 * nr * (in_f + out_f)
    if wd:
        thin += 6.0 * out_f * in_f * rank
        nbytes += (24.0 * n * out_f if self.wd_on_output else 28.0 * n * in_f) + 4.0 * out_f * in_f + 150.0 * out_f * rank
        t += 45 * _LAUNCH
    t += thin / _MM_BF16_THIN + nbytes / _BW
    dense = (8.0 if bypass else 2.0) * n * out_f * in_f / _MM_BF16
    dense += _DENSE_BYTES[(algo, bool(wd))] * out_f * in_f / _BW
    return t <= dense


def _khatri_rao(B1, A1, B2, A2) -> Tuple[torch.Tensor, torch.Tensor]:
    """KB (out x r1*r2), KA (r1*r2 x in) with KB @ KA == (B1 @ A1) * (B2 @ A2)."""
    kb = (B1.unsqueeze(2) * B2.unsqueeze(1)).reshape(B1.shape[0], -1)
    ka = (A1.unsqueeze(1) * A2.unsqueeze(0)).reshape(-1, A1.shape[1])
    return kb, ka


def _loha_kr(self, multiplier=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factors of get_weight(shape) * scalars [* multiplier] (loha.py:470-486, :442-457): gamma = alpha/dim and the scalars
    on B1 (rows) / A1 (columns)."""
    b1, a1 = self.hada_w1_a * float(self.scale), self.hada_w1_b
    st = getattr(self, "scalar_type", "none")
    if st == "scalar":
        b1 = b1 * self.scalar
    elif st in ("row", "row_column"):
        b1 = b1 * self.row_scalar.view(-1, 1)
    if st in ("column", "row_column"):
        a1 = a1 * self.col_scalar.view(1, -1)
    if multiplier is not None:
        b1 = b1 * multiplier
    return _khatri_rao(b1, a1, self.hada_w2_a, self.hada_w2_b)


def _abba_kr(self, multiplier) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factors of make_weight * scale * multiplier (abba.py:418; make_weight includes the scalar, :348)."""
    f = self.scalar * float(self.scale) * multiplier
    return _khatri_rao(self.lora_up1.weight * f, self.lora_down1.weight, self.lora_up2.weight, self.lora_down2.weight)


def _hada_fast_dtype(self, x: torch.Tensor, uses_dropout: bool) -> Optional[torch.dtype]:
    """The dtype F.linear runs in for this rebuild-mode call (autocast's, else the module's) when the flag's forward
    applies, else None (stock forward: conv, bypass, LyCORIS' own torch_compile, dropouts in training, orthogonalize,
    float8 base weights)."""
    if (self.module_type != "linear" or self.bypass_mode or "_forward_rebuild_core" in self.__dict__
            or getattr(self, "use_orthogonal_weights", False)
            or (self.training and (self.rank_dropout or self.module_dropout or (uses_dropout and self.dropout)))):
        return None
    w, b = self.org_module[0].weight, self.org_module[0].bias
    if w is None or w.dtype not in _FLOATS or (b is not None and b.dtype not in _FLOATS):
        return None  # --fp8_base (train_network.py): float8 has no type promotion; the stock path casts it with .to()
    cd = self._cached_dtype
    if x.dtype not in _FLOATS or cd not in _FLOATS:
        return None
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        # autocast leaves fp64 alone: an fp64 module would not run in dt
        return dt if dt in (torch.bfloat16, torch.float16) and cd != torch.float64 else None
    return cd


def _hada_cast(t: torch.Tensor, cd: torch.dtype, dt: torch.dtype) -> torch.Tensor:
    """t as the stock forward hands it to F.linear: .to(module dtype), then autocast's cast to dt. One cast when the
    first is exact (bf16/fp16 -> fp32) or dt is the module dtype: the same values without the fp32 copy."""
    if t.dtype != cd and cd != dt and torch.promote_types(t.dtype, cd) != cd:
        t = t.to(cd)
    return t.to(dt)


def _hada_dora_norm(self, w0: torch.Tensor, kb: torch.Tensor, ka: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    """||W0 + KB KA|| per output row (wd_on_output) or per input column, in fp32 (or the wider dora_scale dtype),
    without KB KA: ||W0_i||^2 + 2 (W0 KA^T)_i . KB_i + KB_i (KA KA^T) KB_i^T. The W0 matmul runs in dt (bf16 under
    autocast, as the stock merged weight's matmul did), the small rank x rank terms in fp32 (a bf16 dora_scale under
    --full_bf16 would round each term on its own)."""
    nd = torch.promote_types(self.dora_scale.dtype, torch.float32)
    with torch.autocast(w0.device.type, enabled=False):
        kbn, kan = kb.to(nd), ka.to(nd)
        if self.wd_on_output:
            base = torch.linalg.vector_norm(w0, dim=1, dtype=nd).square()
            cross = (F.linear(w0.to(dt), ka.to(dt)).to(nd) * kbn).sum(1)
            quad = ((kbn @ (kan @ kan.t())) * kbn).sum(1)
        else:
            base = torch.linalg.vector_norm(w0, dim=0, dtype=nd).square()
            cross = (torch.mm(kb.to(dt).t(), w0.to(dt)).to(nd) * kan).sum(0)
            quad = (((kbn.t() @ kbn) @ kan) * kan).sum(0)
        # clamp: a zero row stays finite (vector_norm's gradient at 0 is 0, sqrt's is inf)
        return torch.sqrt((base + 2 * cross + quad).clamp_min(torch.finfo(nd).tiny))


def _mul_into(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """a * b computed in the promoted dtype, written in dtype in one pass (one rounding, no fp32 temporary)."""
    return torch.mul(a, b, out=torch.empty(torch.broadcast_shapes(a.shape, b.shape), dtype=dtype, device=a.device))


def _col_dot(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """sum_n a[n, i] * b[n, i] in dtype: the products in dtype without copying a or b to it (an expanded dtype zero makes
    addcmul compute in dtype). A bf16 einsum would round each sum to bf16 (DoRA scale gradient ~20% less accurate)."""
    zero = torch.zeros((), dtype=dtype, device=a.device).expand(a.shape)
    return torch.addcmul(zero, a, b).sum(0)


class _HadaDelta(torch.autograd.Function):
    """up = (x KA^T) KB^T (+ bias), x and bias in the compute dtype dt (bf16 under autocast), KA, KB in the network dtype.

    grad_dtype is the dtype of the adapter-gradient products: the network dtype for LoHa (its stock HadaWeight backward
    runs in fp32, outside autocast; a bf16 chain would round mid and the two gradient products), dt for ABBA (its stock
    make_weight backward runs in bf16 under autocast). The base x W0^T is a plain addmm with up as the addend (one rounding
    of base + delta), outside this function: gradient checkpointing's early stop then skips that matmul in the recompute
    (a custom function is always re-run whole)."""

    @staticmethod
    def forward(ctx, x, ka, kb, bias, grad_dtype):
        dt = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            x2 = x.reshape(-1, x.shape[-1])
            mid = torch.mm(x2, ka.to(dt).t())
            up = F.linear(mid, kb.to(dt), bias)
        ctx.save_for_backward(x, ka, kb, mid if grad_dtype == dt else None)
        ctx.grad_dtype = grad_dtype
        return up.view(*x.shape[:-1], up.shape[-1])

    @staticmethod
    def backward(ctx, g):
        x, ka, kb, mid = ctx.saved_tensors
        gd, dt = ctx.grad_dtype, x.dtype
        grad_x = grad_ka = grad_kb = None
        with torch.autocast(x.device.type, enabled=False):
            x2 = x.reshape(-1, x.shape[-1])
            g_g = g.reshape(-1, g.shape[-1]).to(gd)
            grad_mid = torch.mm(g_g, kb.to(gd))  # (N, R)
            if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                x_g = x2.to(gd)
                if ctx.needs_input_grad[1]:
                    grad_ka = torch.mm(grad_mid.t(), x_g).to(ka.dtype)
                if ctx.needs_input_grad[2]:
                    if mid is None:
                        mid = torch.mm(x_g, ka.to(gd).t())  # recomputed in the gradient dtype
                    grad_kb = torch.mm(g_g.t(), mid).to(kb.dtype)
            if ctx.needs_input_grad[0]:
                grad_x = torch.mm(grad_mid.to(dt), ka.to(dt)).view(x.shape)
        return grad_x, grad_ka, grad_kb, None, None


class _HadaDoRAOut(torch.autograd.Function):
    """y = (x W0^T + (x KA^T) KB^T) * sc + bias: DoRA's per-output-row scale on the output (axis wd_on_output).

    One function because sc's gradient sum_n g_n * (x W^T)_n needs the unscaled product (saved, so the recompute re-runs
    the base matmul, as the stock path's merged-weight gradient costs one); the scale is one fused pass, the input gradient
    one addmm (base + delta rounded once). grad_dtype as in _HadaDelta."""

    @staticmethod
    def forward(ctx, x, w0, bias, ka, kb, sc, grad_dtype):
        dt = x.dtype
        with torch.autocast(x.device.type, enabled=False):
            x2 = x.reshape(-1, x.shape[-1])
            mid = torch.mm(x2, ka.to(dt).t())
            z = torch.addmm(torch.mm(mid, kb.to(dt).t()), x2, w0.t())  # x W^T, unscaled
            y = torch.empty_like(z)
            if bias is None:
                torch.mul(z, sc, out=y)
            else:
                torch.addcmul(bias, z, sc, out=y)
        ctx.save_for_backward(x, w0, ka, kb, sc, z, mid if grad_dtype == dt else None)
        ctx.grad_dtype = grad_dtype
        return y.view(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, g):
        x, w0, ka, kb, sc, z, mid = ctx.saved_tensors
        gd, dt = ctx.grad_dtype, x.dtype
        need_x, need_ka, need_kb, need_sc = (ctx.needs_input_grad[i] for i in (0, 3, 4, 5))
        grad_x = grad_ka = grad_kb = grad_sc = None
        with torch.autocast(x.device.type, enabled=False):
            x2 = x.reshape(-1, x.shape[-1])
            g2 = g.reshape(-1, g.shape[-1]).to(dt)
            gs = _mul_into(g2, sc, dt)  # gradient of the unscaled x W^T, for the base matmul
            gs_g = gs if gd == dt else _mul_into(g2, sc, gd)  # and for the adapter (unrounded in fp32)
            grad_mid = torch.mm(gs_g, kb.to(gd))  # (N, R)
            if need_ka or need_kb:
                x_g = x2.to(gd)
                if need_ka:
                    grad_ka = torch.mm(grad_mid.t(), x_g).to(ka.dtype)
                if need_kb:
                    if mid is None:
                        mid = torch.mm(x_g, ka.to(gd).t())  # recomputed in the gradient dtype
                    grad_kb = torch.mm(gs_g.t(), mid).to(kb.dtype)
            if need_sc:
                grad_sc = _col_dot(g2, z, sc.dtype)
            if need_x:
                grad_x = torch.addmm(torch.mm(grad_mid.to(dt), ka.to(dt)), gs, w0).view(x.shape)
        return grad_x, None, None, grad_ka, grad_kb, grad_sc, None


def _hada_linear_forward(self, x: torch.Tensor, kb: torch.Tensor, ka: torch.Tensor, dt: torch.dtype, multiplier,
                         grad_dtype: torch.dtype):
    """The stock rebuild forward F.linear(x, W, b), W = W0 + KB KA or apply_weight_decompose(W0 + KB KA, multiplier)
    (loha.py:481-487, abba.py:420-427), as x W0^T + (x KA^T) KB^T: no dense delta, no merged-weight gradient. The input,
    base weight and bias in dt as the stock F.linear saw them."""
    cd = self._cached_dtype
    w0 = _hada_cast(self.get_org_weight_for_compute(x.device).detach(), cd, dt)  # stock: .data (no grad into the base)
    bias = self.get_org_bias_for_compute(x.device)
    bc = None if bias is None else _hada_cast(bias, cd, dt)
    xc = _hada_cast(x, cd, dt)
    if self.wd:
        sc = self.dora_scale.reshape(-1) / (_hada_dora_norm(self, w0, kb, ka, dt) + self._dora_eps)
        sc = multiplier * (sc - 1) + 1  # apply_weight_decompose (loha.py:383-386)
        if self.wd_on_output:  # (x W^T) * sc == x (W * sc[:, None])^T, the bias unscaled
            return _HadaDoRAOut.apply(xc, w0, bc, ka, kb, sc, grad_dtype)
        # x (W * sc[None, :])^T == (x * sc) W^T. One rounded copy per consumer: their gradients then meet in fp32 (one
        # bf16 sum would round the input and scale gradients once more than the stock merged-weight path)
        xs = xc * sc
        xc, xd = xs.to(dt), xs.to(dt)
    else:
        xd = xc
    up = _HadaDelta.apply(xd, ka, kb, bc, grad_dtype)
    out_f = w0.shape[0]
    return torch.addmm(up.reshape(-1, out_f), xc.reshape(-1, xc.shape[-1]), w0.t()).view(up.shape)


def _loha_weight(self) -> torch.Tensor:
    """get_weight(self.shape) of a linear LoHa (no tucker / orthogonalize / rank dropout) with gamma made on the CPU:
    loha.py:306 makes it with device=cuda, a blocking host-to-device copy per call. Same values (a 0-dim operand)."""
    from lycoris.functional.loha import diff_weight

    gamma = torch.tensor(self.scale, dtype=self.hada_w1_b.dtype)
    w = diff_weight(self.hada_w1_b, self.hada_w1_a, self.hada_w2_b, self.hada_w2_a, None, None, gamma=gamma)
    return w.reshape(self.shape)


def _hada_dense_forward(self, x: torch.Tensor, diff: torch.Tensor, dt: torch.dtype, multiplier, mult_on_diff: bool):
    """The stock rebuild forward (loha.py:504-487, abba.py:449-427) with the same float ops, without the fp32 copies of
    the activation and the base weight that autocast casts straight back (like library/lycoris_fast.py for LoKr):
    W0 + diff promotes bf16 -> fp32 as W0.to(fp32) + diff does, and the merged weight is cast to dt once."""
    cd = self._cached_dtype
    w0 = self.get_org_weight_for_compute(x.device).detach()
    bias = self.get_org_bias_for_compute(x.device)
    if mult_on_diff:
        diff = diff * multiplier
    merged = (w0 if torch.promote_types(w0.dtype, cd) == cd else w0.to(cd)) + diff
    if self.wd:
        merged = self.apply_weight_decompose(merged, multiplier)
    bc = None if bias is None else _hada_cast(bias, cd, dt)
    return F.linear(_hada_cast(x, cd, dt), merged.to(dt), bc)


def _loha_forward(self, x, *args, **kwargs):
    dt = _hada_fast_dtype(self, x, uses_dropout=False)  # the LoHa rebuild forward has no dropout (loha.py:504-513)
    if dt is None:
        return _original(_loha_cls(), "forward")(self, x, *args, **kwargs)
    mult = self.multiplier_buf
    pd = self.hada_w1_a.dtype
    if _hada_factored_cheaper(self, x, "loha", self.hada_w1_a.shape[1] * self.hada_w2_a.shape[1], self.wd, pd != dt):
        kb, ka = _loha_kr(self, None if self.wd else mult)  # DoRA: the multiplier only enters the decomposition
        return _hada_linear_forward(self, x, kb, ka, dt, mult, pd)
    diff = _loha_apply_scalars(self, _loha_weight(self).to(self._cached_dtype))
    return _hada_dense_forward(self, x, diff, dt, mult, not self.wd)


_ABBA_DORA_FUSED = True  # tests: False = the _hada_* paths only, as before


def _abba_dora_dense(self, x: torch.Tensor, dt: torch.dtype, mult) -> Optional[torch.Tensor]:
    """ABBA rebuild with DoRA per output row, apply_weight_decompose(W0 + KB KA, mult) (abba.py:416-427), through
    _DenseDoRAOut (B = KB with scalar, scale and multiplier in it, A = KA, s = 1) when _lf_dora_dense picks the dense
    form for this call (on CUDA: everywhere but very wide inputs; on the CPU never), else None (the _hada_* paths).
    Measured on the RTX 5090 at Anima shapes (dim 16, bf16 autocast, checkpointing, ms per call for 2048->2048 x 4096
    tokens / 2048->8192 / adaLN 256->6144 x 1 token): the stock rebuild 2.13 / 6.17 / 1.83, _HadaDoRAOut 2.82 / 5.22 /
    2.93 (host-bound: ~145 launches per call), this 2.08 / 5.44 / 1.58; an Anima-like block (16 modules + attention)
    44.7 ms stock, 42.3 split, 37.9 with this. Input and bias in dt as the stock F.linear saw them; the base weight as
    the stock merge saw it, W0.to(module dtype) (abba.py:461), not rounded to dt: _DenseDoRAOut rounds W' once, and an
    fp32 base rounded to bf16 first made the output ~20% and the dora_scale gradient ~25% less accurate than stock."""
    cd = self._cached_dtype
    xc = _hada_cast(x, cd, dt)
    x2 = xc.reshape(-1, xc.shape[-1])
    out_f, in_f = int(self.shape[0]), int(self.shape[1])
    if not _lf_dora_dense(x2.shape[0], out_f, in_f, self.r1 * self.r2, dt, x2.device):
        return None
    kb, ka = _abba_kr(self, mult)
    w0 = self.get_org_weight_for_compute(x.device).detach()
    if torch.promote_types(w0.dtype, cd) != cd:
        w0 = w0.to(cd)  # a lossy cast the stock makes too; an exact one is left to _DenseDoRAOut's cast to nd
    bias = self.get_org_bias_for_compute(x.device)
    bc = None if bias is None else _hada_cast(bias, cd, dt)
    y = _DenseDoRAOut.apply(x2, w0, bc, ka, kb, self.dora_scale.to(x.device), mult, self._dora_eps, 1.0, dt)
    return y.view(*x.shape[:-1], y.shape[-1])


def _abba_forward(self, x):
    dt = _hada_fast_dtype(self, x, uses_dropout=True)
    if dt is None or self.r1 == 0 or self.r2 == 0:
        return _original(_abba_cls(), "forward")(self, x)
    mult = self.multiplier_buf
    if self.wd and self.wd_on_output and _ABBA_DORA_FUSED:
        y = _abba_dora_dense(self, x, dt, mult)
        if y is not None:
            return y
    if _hada_factored_cheaper(self, x, "abba", self.r1 * self.r2, self.wd, False):
        kb, ka = _abba_kr(self, mult)  # the ABBA delta carries the multiplier with and without DoRA (abba.py:418)
        return _hada_linear_forward(self, x, kb, ka, dt, mult, dt)
    diff = self.make_weight(x.device).to(self._cached_dtype) * self.scale * mult
    return _hada_dense_forward(self, x, diff, dt, mult, False)


# ======================================================================================================================
# LoCon family loaders (create_network_from_weights: --dim_from_weights, --base_weights, merge tools)
# ======================================================================================================================


def _locon_make_module_from_state_dict(cls, lora_name, orig_module, up, down, mid, alpha, dora_scale, *extra):
    # get_module passes every weight_list entry (8: + pissa_A_init, pissa_B_init, pissa_converted, locon.py:30-39) to a
    # function that took 5 (locon.py:491-493): TypeError for every LoCon/GoRA file. Extra entries are unused here.
    module = cls(
        lora_name,
        orig_module,
        1.0,
        down.size(0),
        float(alpha),
        use_tucker=mid is not None,
        weight_decompose=dora_scale is not None,
        wd_on_output=_wd_on_output_from(dora_scale, orig_module),
    )
    module.lora_up.weight.data.copy_(up)
    module.lora_down.weight.data.copy_(down)
    if mid is not None:
        module.lora_mid.weight.data.copy_(mid)
    if dora_scale is not None:
        module.dora_scale.copy_(dora_scale)
    return module


def _ralora_make_module_from_state_dict(cls, lora_name, orig_module, up, down, mid, alpha, dora_scale, *extra):
    # weight_list has 8 + 128 block entries; block files (n_split > 1) load through network.load_weights instead.
    if up is None or down is None:
        raise ValueError(
            f"RaLoRA {lora_name}: block-diagonal file; load it with network.load_weights (or save it with"
            " the fixed LyCORIS paths (the default, not --lycoris_stock), which write a plain LoRA)"
        )
    module = cls(
        lora_name,
        orig_module,
        1.0,
        down.size(0) if down.dim() >= 2 else 0,
        float(alpha),
        use_tucker=mid is not None,
        weight_decompose=dora_scale is not None,
        wd_on_output=_wd_on_output_from(dora_scale, orig_module),
    )
    module.lora_up.weight.data.copy_(up)
    module.lora_down.weight.data.copy_(down)
    if mid is not None:
        module.lora_mid.weight.data.copy_(mid)
    if dora_scale is not None:
        module.dora_scale.copy_(dora_scale)
    return module


# ======================================================================================================================
# LoRA²
# ======================================================================================================================


def _lora2_custom_state_dict(self):
    # Full lora_dim tensors (lora2.py:372-373 cut them to the effective rank D, so a state or weights file could not be
    # loaded back into the same network once D < lora_dim). Scalars baked like LoConModule.custom_state_dict.
    destination = {
        "alpha": self.alpha,
        "lora2_nu": self.lora2_nu,
        "lora_up.weight": self._apply_row_scalar(self.lora_up.weight),
        "lora_down.weight": self._apply_col_scalar(self.lora_down.weight),
    }
    if self.wd:
        destination["dora_scale"] = self.dora_scale
    return destination


def _lora2_make_module_from_state_dict(cls, lora_name, orig_module, up, down, nu, alpha, dora_scale):
    d = down.size(0)
    module = cls(
        lora_name,
        orig_module,
        1.0,
        lora_dim=d,
        alpha=float(alpha),
        weight_decompose=dora_scale is not None,
        wd_on_output=_wd_on_output_from(dora_scale, orig_module),
        lora2_nu_target=d,
    )
    module.lora_up.weight.data[:, :d].copy_(up)
    module.lora_down.weight.data[:d].copy_(down)
    module.lora2_nu.data.copy_(nu)
    if dora_scale is not None:
        module.dora_scale.copy_(dora_scale)
    return module


# ======================================================================================================================
# GoRA / RaLoRA pre-pass
# ======================================================================================================================


def _adapt_forward_fn(forward_fn: Callable, base_weights: List[torch.Tensor], state: dict) -> Callable:
    """forward_fn for the pre-pass loops: accepts (batch) or (model, batch) whichever the caller passes and the trainer's
    function takes; runs with grad enabled (precompute_and_init is @torch.no_grad, locon.py:1604/2418); frees each
    base weight's .grad right after it is accumulated (the record hooks return the gradient, so it would stay)."""
    try:
        params = list(inspect.signature(forward_fn).parameters.values())
        var_pos = any(p.kind == p.VAR_POSITIONAL for p in params)
        n_pos = sum(1 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)
    except (TypeError, ValueError):
        var_pos, n_pos = True, 0

    def free_grad(p):
        p.grad = None

    def adapted(*call_args):
        if not state["hooked"]:
            for w in base_weights:
                w.grad = None
                if w.requires_grad and hasattr(w, "register_post_accumulate_grad_hook"):
                    state["handles"].append(w.register_post_accumulate_grad_hook(free_grad))
            state["hooked"] = True
        else:
            for w in base_weights:
                w.grad = None
        if var_pos or n_pos >= len(call_args):
            fn_args = call_args
        else:
            fn_args = call_args[len(call_args) - max(n_pos, 1):]
        with torch.enable_grad():
            return forward_fn(*fn_args)

    return adapted


def _swap_forward_fn(args: tuple, kwargs: dict, index: int, wrap: Callable) -> Tuple[tuple, dict]:
    if "forward_fn" in kwargs:
        kwargs = dict(kwargs)
        kwargs["forward_fn"] = wrap(kwargs["forward_fn"])
    elif len(args) > index:
        args = list(args)
        args[index] = wrap(args[index])
        args = tuple(args)
    return args, kwargs


def _modules_arg(args: tuple, kwargs: dict) -> list:
    return list(kwargs.get("modules", args[0] if args else []) or [])


def _gora_precompute_gradients(*args, **kwargs):
    from lycoris.modules import gora_utils

    orig = _original(gora_utils, "gora_precompute_gradients")
    modules = _modules_arg(args, kwargs)
    base = [_org_weight(m) for m in modules]
    state = {"hooked": False, "handles": []}
    args, kwargs = _swap_forward_fn(args, kwargs, 2, lambda fn: _adapt_forward_fn(fn, base, state))
    try:
        return orig(*args, **kwargs)
    finally:
        for h in state["handles"]:
            h.remove()
        for w in base:
            w.grad = None
            w.requires_grad_(False)


def _set_iters_one(modules: Iterable, only_if_frozen: bool) -> None:
    """grad_stored is already the mean (gora_utils.py:788-792, ralora_utils.py:576-580): make the later
    '/ iters' divisions (gora_utils.py:277 and :549, ralora_utils.py:324 and :428) no-ops."""
    for m in modules:
        w = _org_weight(m)
        if getattr(w, "grad_stored", None) is None or not hasattr(w, "iters"):
            continue
        if only_if_frozen and w.requires_grad:
            continue  # GoRA's in-loop convergence check: grad_stored is still the sum there
        w.iters = 1


def _gora_allocate_ranks(modules, *args, **kwargs):
    from lycoris.modules import gora_utils

    # After the accumulation loop the hooks are gone and requires_grad is False (gora_utils.py:782-786).
    _set_iters_one(modules, only_if_frozen=True)
    return _original(gora_utils, "allocate_ranks")(modules, *args, **kwargs)


def _gora_dynamic_init(modules, *args, **kwargs):
    from lycoris.modules import gora_utils

    _set_iters_one(modules, only_if_frozen=True)
    return _original(gora_utils, "gora_dynamic_init")(modules, *args, **kwargs)


def _ralora_get_allocated_rank(modules, *args, **kwargs):
    from lycoris.modules import ralora_utils

    _set_iters_one(modules, only_if_frozen=False)  # only called after the averaging (ralora_utils.py:576-580)
    return _original(ralora_utils, "get_allocated_rank")(modules, *args, **kwargs)


def _ralora_compute_n_split_allocations(modules, *args, **kwargs):
    from lycoris.modules import ralora_utils

    _set_iters_one(modules, only_if_frozen=False)
    return _original(ralora_utils, "compute_n_split_allocations")(modules, *args, **kwargs)


def _freeze_unused_ralora_dense(modules: Iterable) -> int:
    """RaLoRA block layers (n_split > 1) keep their dense lora_up/lora_down, which the block forward never uses
    (locon.py:1832-1990) and which are never saved: freeze them so they stay out of the optimizer."""
    n = 0
    for m in modules:
        if type(m).__name__ == "RaLoRAModule" and int(getattr(m, "n_split", 1) or 1) > 1:
            for p in (m.lora_up.weight, m.lora_down.weight):
                if p.requires_grad:
                    p.requires_grad_(False)
                    n += 1
    return n


def _ralora_precompute_gradients(*args, **kwargs):
    from lycoris.modules import ralora_utils

    orig = _original(ralora_utils, "ralora_precompute_gradients")
    modules = _modules_arg(args, kwargs)
    base = [_org_weight(m) for m in modules]
    was_trainable = {id(p): (p, p.requires_grad) for m in modules for p in m.parameters()}
    state = {"hooked": False, "handles": []}
    args, kwargs = _swap_forward_fn(args, kwargs, 2, lambda fn: _adapt_forward_fn(fn, base, state))
    try:
        return orig(*args, **kwargs)
    finally:
        for h in state["handles"]:
            h.remove()
        for w in base:  # ralora_utils.py:516 left requires_grad on: a dense base-weight gradient on every step
            w.grad = None
            w.requires_grad_(False)
        for p, flag in was_trainable.values():  # :523-525 froze the adapters; new block params stay trainable
            p.requires_grad_(flag)
        _freeze_unused_ralora_dense(modules)


def _ralora_dynamic_init(self, avg_rank: int, rank: int, n_split: int = 1):
    from lycoris.modules.locon import RaLoRAModule

    _original(RaLoRAModule, "dynamic_init")(self, avg_rank, rank, n_split)
    # The block ParameterLists are created with torch.empty/zeros on the CPU in fp32 (locon.py:1773-1796): keep them on
    # the module's device and dtype like every other adapter weight.
    ref = self.lora_up.weight
    for p in list(self._mini_lora_A) + list(self._mini_lora_B):
        if p.device != ref.device or p.dtype != ref.dtype:
            p.data = p.data.to(device=ref.device, dtype=ref.dtype)


def _ralora_load_weight_prehook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    from lycoris.modules.locon import RaLoRAModule

    a0 = f"{prefix}lora_down_block0.weight"
    n_key = f"{prefix}n_split"
    if a0 in state_dict and self.n_split > 1:
        n = int(state_dict[n_key].item()) if n_key in state_dict else 1
        a_keys = [f"{prefix}lora_down_block{i}.weight" for i in range(n)]
        b_keys = [f"{prefix}lora_up_block{i}.weight" for i in range(n)]
        if (
            n == self.n_split
            and all(k in state_dict for k in a_keys + b_keys)
            and all(state_dict[a_keys[i]].shape == self._mini_lora_A[i].shape for i in range(n))
            and all(state_dict[b_keys[i]].shape == self._mini_lora_B[i].shape for i in range(n))
        ):
            # Same structure (restore_gora_ralora_allocation ran first): copy in place so the optimizer keeps the very
            # Parameters it was built with. The stock hook (locon.py:2045-2120) always makes new Parameters.
            with torch.no_grad():
                for i in range(n):
                    self._mini_lora_A[i].copy_(state_dict.pop(a_keys[i]))
                    self._mini_lora_B[i].copy_(state_dict.pop(b_keys[i]))
            state_dict.pop(n_key, None)
            if isinstance(self.scalar, nn.Parameter):
                self.scalar.data.fill_(1.0)
            elif hasattr(self, "scalar"):
                self.scalar.fill_(1.0)
            for k in a_keys + b_keys + [n_key]:
                if k in unexpected_keys:
                    unexpected_keys.remove(k)
            self.__dict__["_laf_blocks_loaded"] = True  # the post hook drops the ParameterLists' own missing keys
            return
    return _original(RaLoRAModule, "load_weight_prehook")(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    )


def _ralora_load_weight_hook(self, module, incompatible_keys):
    from lycoris.modules.locon import RaLoRAModule

    _original(RaLoRAModule, "load_weight_hook")(self, module, incompatible_keys)
    if self.n_split > 1:
        # Block layers keep their unused dense lora_up/lora_down (never saved for n_split > 1, locon.py:2018-2043), and
        # the block weights arrive as lora_*_block{i} keys, not under the ParameterLists' own names.
        own = (f"{self.lora_name}.lora_up.weight", f"{self.lora_name}.lora_down.weight")
        blocks_loaded = self.__dict__.pop("_laf_blocks_loaded", False)
        mini = f"{self.lora_name}._mini_lora_"
        incompatible_keys.missing_keys[:] = [
            k for k in incompatible_keys.missing_keys
            if not (k.endswith(own) or (blocks_loaded and mini in k))
        ]


# ======================================================================================================================
# Diag-OFT / BOFT / T-LoRA dtype
# ======================================================================================================================


def _make_oft_rebuild_core(cls_getter: Callable):
    def _forward_rebuild_core(self, x, org_weight, bias):
        # forward() casts x to the network dtype (fp32) but the rebuilt weight goes back to the base dtype (bf16):
        # F.linear fails where autocast is off (Anima adaLN; diag_oft.py:302, boft.py:358). Upcasting the base weight
        # first is exact and gives the same result under autocast (F.linear casts to bf16 either way).
        org_weight = _cast_if_upcast(org_weight, x.dtype)
        return _original(cls_getter(), "_forward_rebuild_core")(self, x, org_weight, bias)

    return _forward_rebuild_core


def _diag_oft_cls():
    from lycoris.modules.diag_oft import DiagOFTModule

    return DiagOFTModule


def _boft_cls():
    from lycoris.modules.boft import ButterflyOFTModule

    return ButterflyOFTModule


def _oft_get_r(self):
    # diag_oft.py:121-135 / boft.py:134-149 without host syncs: torch.inverse checks the blocks for singularity with a
    # blocking copy (linalg_inv -> _linalg_check_errors), and `if q_norm > constraint` reads the norm on the host. inv_ex
    # runs the same kernel (same bits); the where selects the same values.
    I = self.I.to(device=self.oft_blocks.device, dtype=self.oft_blocks.dtype)
    q = self.oft_blocks - self.oft_blocks.transpose(-1, -2)
    normed_q = q
    if self.constraint > 0:
        q_norm = torch.norm(q) + 1e-8
        normed_q = torch.where(q_norm > self.constraint, q * self.constraint / q_norm, q)
    q_float = normed_q.float()
    I_float = I.float()
    return (I_float + q_float) @ torch.linalg.inv_ex(I_float - q_float).inverse


# Speed (Linear layers, rebuild and bypass): both algos rotate the rows of W (diag_oft.py:293-297: W' = R^T W per block;
# boft.py:337-353: butterfly stages over the rows), so F.linear(x, W', b) = rot(F.linear(x, W)) + b. The rebuild path
# built W' per call (fp32 copies of W; BOFT m permute + einsum rounds over W; again in the checkpoint recompute) and, W'
# requiring grad, ran a full dW' = dy^T x GEMM in backward only to reach oft_blocks. The flag keeps the base GEMMs and
# rotates the output: y = z + z D per block (D = R - I is small, so bf16 keeps it instead of rounding it away inside
# W'), one strided baddbmm each way. BOFT merges its stages into a few dense blocks (_OFT_MAX_BLOCK channels). With a
# bias the rotation runs in fp32 and the bias is added before the one rounding (F.linear adds it in its epilogue).

_OFT_MAX_BLOCK = 256
_AUTOCAST_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


def _blocks(t: torch.Tensor, nb: int, S: int) -> torch.Tensor:
    return t.view(-1, nb, S).transpose(0, 1)  # [nb, rows, S] view of a contiguous [..., nb*S]


class _BlockResidualMM(torch.autograd.Function):
    """y = z + z @ D on contiguous blocks of the last axis: z [..., nb*S] contiguous, D [nb, S, S] in z's dtype.
    Strided batched matmuls on views of z/y (no permute copies); call with autocast off."""

    @staticmethod
    def forward(ctx, z, D):
        nb, S, _ = D.shape
        y = torch.empty_like(z, memory_format=torch.contiguous_format)
        zv = _blocks(z, nb, S)
        torch.baddbmm(zv, zv, D, out=_blocks(y, nb, S))
        ctx.save_for_backward(z, D)
        return y

    @staticmethod
    def backward(ctx, gy):
        z, D = ctx.saved_tensors
        nb, S, _ = D.shape
        gz = gD = None
        with torch.autocast(gy.device.type, enabled=False):
            gy = gy.contiguous()
            gyv = _blocks(gy, nb, S)
            if ctx.needs_input_grad[0]:
                gz = torch.empty_like(gy)
                torch.baddbmm(gyv, gyv, D.transpose(1, 2), out=_blocks(gz, nb, S))
            if ctx.needs_input_grad[1]:
                gD = torch.bmm(_blocks(z, nb, S).transpose(1, 2), gyv)
        return gz, gD


def _finish(y: torch.Tensor, bias: Optional[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
    return (y if bias is None else y + bias.to(y.dtype)).to(dtype)


def _diag_oft_rotate(self, z: torch.Tensor, scale, rebuild: bool, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """rot(z) + bias of diag_oft.py:293-304 (rebuild: rank_drop, rescale) / :252-261 (bypass: mask 1, no rescale, no
    bias: stock rotates the whole base output)."""
    r = self._apply_multiplicative_dropout(self.get_r())
    d = (self.rank_drop(r * scale) if rebuild else r * scale) - scale * self.I  # R - I, R the stock einsum operand
    cd = z.dtype
    if self.rescaled:
        # w = rescale * (R^T W): the factor folded into R's columns. In fp32: D then holds rescale - 1 (not small), and
        # the rescale gradient would rest on single bf16-rounded entries of the D gradient.
        d = (d + self.I) * self.rescale.reshape(self.block_num, 1, self.block_size) - self.I
    if self.rescaled or bias is not None:
        cd = torch.promote_types(cd, torch.float32)
    return _finish(_BlockResidualMM.apply(z.to(cd), d.to(cd)), bias, z.dtype)


def _boft_chunks(m: int, r_b: int) -> List[Tuple[int, int]]:
    """Stage ranges [lo, hi) merged into one dense block each (r_b * 2^(hi-lo) channels <= _OFT_MAX_BLOCK), balanced."""
    smax = max(1, int(math.log2(max(_OFT_MAX_BLOCK // r_b, 2))))
    n = max(1, -(-m // smax))
    sizes = [m // n + (1 if k < m % n else 0) for k in range(n)]
    out, lo = [], 0
    for s in sizes:
        out.append((lo, lo + s))
        lo += s
    return out


def _boft_stage(inp: torch.Tensor, bi: torch.Tensor, i: int, b: int) -> torch.Tensor:
    # boft.py:344-353: stage i on the rows of inp
    k = 2**i * (b // 2)
    inp = inp.unflatten(0, (-1, 2, k)).transpose(1, 2).flatten(0, 2).unflatten(0, (-1, b))
    inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
    return inp.flatten(0, 1).unflatten(0, (-1, k, 2)).transpose(1, 2).flatten(0, 2)


def _boft_rotate(self, z: torch.Tensor, scale, rebuild: bool, bias: Optional[torch.Tensor]) -> torch.Tensor:
    """rot(z) + bias of boft.py:337-360 (rebuild) / :281-302 (bypass: no bias). Channel c = jj + r_b*u
    (jj < r_b = b/2); stage i mixes jj with bit i of u, so stages [lo, hi) together mix jj and bits [lo, hi): dense
    blocks of r_b*2^(hi-lo) channels, contiguous for lo = 0, after one permutation (copy) of the channels otherwise.
    Each chunk's block matrix is the stock stage code applied to identity columns (fp32, autocast off); rescale is
    folded into the last chunk's rows. Few rows (adaLN: one per sample): the stock stages run on z itself (fp32)."""
    m, b = self.boft_m, self.boft_b
    r_b = b // 2
    out = z.shape[-1]
    r = self._apply_multiplicative_dropout(self.get_r())
    I = self.I.to(device=r.device, dtype=r.dtype)
    stages = r * scale + (1 - scale) * I  # boft.py:343 for every stage at once (same elementwise values)
    shape = z.shape
    rows = z.numel() // out
    if rows <= _OFT_MAX_BLOCK:
        cd = torch.promote_types(z.dtype, r.dtype)
        y, stages = z.reshape(rows, out).t().to(cd), stages.to(cd)
        for i in range(m):
            y = _boft_stage(y, stages[i], i, b)
        if self.rescaled:
            y = y * self.rescale.reshape(out, 1)
        return _finish(y.t(), bias, z.dtype).reshape(shape)
    chunks = _boft_chunks(m, r_b)
    cd = z.dtype if bias is None else torch.promote_types(z.dtype, torch.float32)
    y = z.reshape(rows, out).to(cd)
    for n, (lo, hi) in enumerate(chunks):
        S, Bb, Clo = r_b << (hi - lo), 1 << (hi - lo), 1 << lo
        Ghi = out // (S * Clo)
        eye = torch.eye(S, device=r.device, dtype=r.dtype)
        X = eye.view(1, Bb, 1, r_b, S).expand(Ghi, Bb, Clo, r_b, S).reshape(out, S)
        for i in range(lo, hi):
            X = _boft_stage(X, stages[i], i, b)
        Mt = X.view(Ghi, Bb, Clo, r_b, S).transpose(1, 2).reshape(Ghi * Clo, S, S).transpose(1, 2)  # [blk, in, out]
        if self.rescaled and n == len(chunks) - 1:
            Mt = Mt * self.rescale.reshape(Ghi, Bb, Clo, r_b).transpose(1, 2).reshape(Ghi * Clo, 1, S)
        D = (Mt - eye).to(cd)
        if lo == 0:
            y = _BlockResidualMM.apply(y, D)
        else:  # channels (Ghi, Bb, Clo, jj) -> (Ghi, Clo, Bb, jj): the chunk's blocks contiguous, and back
            yp = y.view(rows, Ghi, Bb, Clo, r_b).transpose(2, 3).reshape(rows, out)
            yp = _BlockResidualMM.apply(yp, D)
            y = yp.view(rows, Ghi, Clo, Bb, r_b).transpose(2, 3).reshape(rows, out)
    return _finish(y, bias, z.dtype).view(shape)


def _make_oft_forward(cls_getter: Callable, rotate: Callable):
    def forward(self, x: torch.Tensor, *args, **kwargs):
        # not torch.compiled (compile_forward wraps the instance's _forward_rebuild_core), not an empty batch
        fast = self.module_type == "linear" and "_forward_rebuild_core" not in self.__dict__ and x.numel() > 0
        if fast and self.bypass_mode and rotate is _diag_oft_rotate:
            fast = not self.rescaled and not (self.dropout != 0 and self.training)  # stock masks / rescale formula
        if not fast:
            return _original(cls_getter(), "forward")(self, x, *args, **kwargs)
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x)
        dev = x.device.type
        bias = None
        if self.bypass_mode:
            z = self.org_forward(x).contiguous()  # stock rotates the base output, bias included
            scale = self.multiplier
        else:
            weight = self.get_org_weight_for_compute(x.device)
            bias = self.get_org_bias_for_compute(x.device)
            if not (torch.is_autocast_enabled(dev) and x.dtype in _AUTOCAST_FLOATS and weight.dtype in _AUTOCAST_FLOATS
                    and self._cached_dtype in _AUTOCAST_FLOATS):
                # no autocast (Anima adaLN): forward()'s cast + the bf16 fix (_make_oft_rebuild_core). Under autocast
                # F.linear casts x and W to its dtype anyway, so their fp32 round trip is skipped (same values).
                x = x.to(self._cached_dtype)
                weight = _cast_if_upcast(weight, x.dtype)
                bias = None if bias is None else bias.to(x.dtype)
            z = F.linear(x, weight)
            scale = self.multiplier_buf
            if bias is not None:
                bias = bias.to(z.dtype)  # the value F.linear(x, W', b) adds (autocast casts it to bf16)
        with torch.autocast(dev, enabled=False):
            return rotate(self, z, scale, not self.bypass_mode, bias)

    return forward


def _tlora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # tlora.py:672-694 with the delta weight in the input dtype (was self.dtype = fp32 -> F.linear dtype error with a
    # bf16 input where autocast is off; under autocast F.linear casts to bf16 either way).
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)
    if self.bypass_mode or self._has_batched_mask() or (
        self.isconv and (any(k != 1 for k in self.shape[2:]) or self.kw_dict.get("groups", 1) > 1)
    ):
        return self.bypass_forward(x, scale=self.multiplier)
    base = self.org_forward(x, *args, **kwargs)
    diff_weight, _ = self.get_diff_weight(multiplier=self.multiplier, device=x.device)
    diff_weight = diff_weight.to(x.dtype)
    delta = self.op(x, diff_weight, None, **self.kw_dict)
    return base + delta


def _tlora_bypass_forward_diff(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    # tlora.py:627-661 with lambda in the input dtype (fp32 lambda promoted the bf16 activations, tlora.py:636-653).
    device = x.device
    dtype = x.dtype
    mask = self._get_mask(device)
    lam = (self.lambda_layer.to(device) * mask).to(dtype)
    lam_base = (self.base_lambda.to(device) * mask).to(dtype)
    if self.isconv:
        q_out = self.down_op(x, self.q_layer.weight.to(dtype), None, **self.kw_dict_down)
        curr_out = self.up_op(q_out * self._reshape_lam_for_broadcast(lam, q_out), self.p_layer.weight.to(dtype), None,
                              **self.kw_dict_up)
        q_base_out = self.down_op(x, self.base_q.to(dtype), None, **self.kw_dict_down)
        base_out = self.up_op(q_base_out * self._reshape_lam_for_broadcast(lam_base, q_base_out), self.base_p.to(dtype),
                              None, **self.kw_dict_up)
    else:
        q_out = self.down_op(x, self.q_layer.weight.to(dtype), None)
        curr_out = self.up_op(q_out * self._reshape_lam_for_broadcast(lam, q_out), self.p_layer.weight.to(dtype), None)
        q_base_out = self.down_op(x, self.base_q.to(dtype), None)
        base_out = self.up_op(q_base_out * self._reshape_lam_for_broadcast(lam_base, q_base_out), self.base_p.to(dtype),
                              None)
    diff = curr_out - base_out
    return self.drop(diff * self.scalar.to(device=device, dtype=dtype) * self.scale * scale)


# ======================================================================================================================
# TSM
# ======================================================================================================================

_TSM_SAMPLE_T: Dict[int, List[int]] = {}  # group id -> per-sample integer timesteps of the current batch

# --anima_block_compile (library/anima_block_compile.py): set by the trainer before the network is created. The
# compiled Block graphs trace the LyCORIS module forwards, so (1) after apply_to the adapted Linears' forwards are
# re-bound through closures (rebind_forwards: dynamo 2.7 builds a bound method's __self__ without a source and then
# cannot read org_module[0]), (2) set_timestep hoists the TSM expert choice / stage-2 embedding row onto the modules
# (_hoist_tsm_step) so the graphs read a per-step tuple / index tensor instead of the integer timestep, (3) a chunk whose
# rows carry different timesteps runs eager (mixed_timestep_chunk; its expert union changes per chunk).
BLOCK_COMPILE = False


def set_block_compile(enabled: bool) -> None:
    global BLOCK_COMPILE
    BLOCK_COMPILE = bool(enabled)


def rebind_forwards(network) -> int:
    """Re-bind every adapted Linear's instance forward (LyCORIS apply_to: `org_module.forward = self.forward`, a bound
    method) to a closure over the LyCORIS module. Same call, same results; the closure cell gives dynamo a source for
    the module, so the compiled Block graph can read `self.org_module[0]` (a plain list) - with the bound method every
    adapted Linear is a graph break. The closure names its module as __wrapped_lora__ (NOT __self__: dynamo refuses to
    inline any function carrying a __self__ attribute, "Inline attempt with __self__"), and lycoris_fast's sample-time
    merged weight cache reads that name next to a bound method's __self__. Idempotent."""
    n = 0
    for lora in _all_loras(network):
        org = getattr(lora, "org_module", None)
        lin = org[0] if isinstance(org, (list, tuple)) and org else None
        if lin is None:
            continue
        bound = lin.__dict__.get("forward")
        if bound is None or getattr(bound, "__wrapped_lora__", None) is lora:
            continue  # not patched by this module, or already re-bound
        if getattr(bound, "__self__", None) is not lora:
            continue  # someone else's patch (RamTorch streaming forward, a cached merge): leave it

        def _make(module):
            def forward(x, *args, **kwargs):
                return module.forward(x, *args, **kwargs)

            forward.__wrapped_lora__ = module
            return forward

        lin.forward = _make(lora)
        n += 1
    return n


def mixed_timestep_chunk() -> bool:
    """True while the current micro-step's rows carry different TSM timesteps (set_timestep stored them per group): the
    block-compile wrapper runs such a chunk through the eager Block._forward (the masked path's expert union changes
    with the chunk, so a compiled graph would be re-traced per combination)."""
    return any(len(set(v)) > 1 for v in _TSM_SAMPLE_T.values())


def _hoist_tsm_step(modules, ints: List[int]) -> None:
    """Block compile: store this micro-step's expert choice (and the stage-2 embedding row as an index tensor) on each
    TSM module, read by _tsm_experts / _get_timestep / _tsm_compute_gates while tracing (a single timestep); for a
    mixed-timestep chunk clear them (the chunk runs eager, see mixed_timestep_chunk)."""
    single = len(set(ints)) == 1
    for m in modules:
        d = m.__dict__
        if not single:
            d.pop("_hx_experts", None)
            d.pop("_hx_tidx", None)
            d.pop("_hx_idx", None)
            continue
        t = ints[0]
        if type(m).__name__ == "TSMModule":
            d["_hx_experts"] = _tsm_experts(m, t)
            if int(getattr(m, "tsm_stage", 1)) == 2 and hasattr(m, "timestep_embed"):
                w = m.timestep_embed.weight
                t_idx = min(max(int(t) - 1, 0), int(m.tsm_num_timesteps) - 1)
                prev = d.get("_hx_tidx")
                if prev is not None and prev.device == w.device:
                    prev.fill_(t_idx)  # in place: the compiled graph keeps the tensor identity (no re-guarding)
                else:
                    d["_hx_tidx"] = torch.tensor([t_idx], dtype=torch.long, device=w.device)
            else:
                d.pop("_hx_tidx", None)
        else:  # TSMLokrModule (library/tsm_lokr.py): its own expert selection
            idx_fn = getattr(m, "_expert_indices", None)
            if idx_fn is not None:
                d["_hx_idx"] = idx_fn(t)
_TSM_EXPERTS: Dict[Tuple[Any, int, Tuple[int, ...]], Tuple[int, ...]] = {}  # (t, T, n_scales) -> expert per scale
_TSM_PLANS: Dict[Any, Any] = {}  # per-step grouping of a mixed-timestep batch (device index tensors)


def _tsm_experts(self, t) -> Tuple[int, ...]:
    """Active expert per scale for timestep t (tsm.py:297-315), computed once per (t, T, n_scales). Under block compile
    the choice set_timestep hoisted onto the module (_hx_experts): the graph then guards on the expert tuple (one entry
    per expert choice) instead of on the integer timestep (one per distinct value) and never mutates this cache."""
    if torch.compiler.is_compiling():
        hx = self.__dict__.get("_hx_experts")
        if hx is not None:
            return hx
    key = (t, self.tsm_num_timesteps, tuple(self.tsm_n_scales))
    idx = _TSM_EXPERTS.get(key)
    if idx is None:
        if len(_TSM_EXPERTS) > 8192:
            _TSM_EXPERTS.clear()
        idx = _TSM_EXPERTS[key] = tuple(self._get_expert_index(t, s) for s in range(len(self.tsm_n_scales)))
    return idx


def _tsm_get_timestep(self) -> int:
    """TSMModule._get_timestep. Under block compile with a hoisted step (one timestep per chunk) the graph needs no
    integer timestep - _tsm_experts reads the hoisted expert choice and _tsm_compute_gates the hoisted embedding row - so
    a constant is returned instead of the global dict's value, which would be a guard per distinct timestep value."""
    from lycoris.modules.tsm import TSMModule

    if torch.compiler.is_compiling() and "_hx_experts" in self.__dict__:
        if int(getattr(self, "tsm_stage", 1)) == 1 or "_hx_tidx" in self.__dict__:
            return 0
    return _original(TSMModule, "_get_timestep")(self)


def _tsm_compute_gates(self, bottleneck, input_features, timestep):
    from lycoris.modules.tsm import TSMModule

    if not self.isconv:
        if self.tsm_router_input == "bottleneck":
            src, width = bottleneck, self.lora_dim
        else:
            src, width = input_features, self.in_dim
        if isinstance(src, torch.Tensor) and src.dim() >= 2 and src.shape[-1] == width:
            # Pool every leading axis (batch, frames, H, W, tokens) to (width,): tsm.py:362-371 kept axis 1 for >2-D
            # inputs, which on Anima's 5-D activations is the frame axis (crash "1x1 and 256x1" at :379).
            # fp32 accumulation without an fp32 copy of the activations (fp64 activations keep fp64, so the fp64 tests
            # compare paths exactly; the GPU's bf16 / fp32 activations pool in fp32 as before).
            z = src.detach().mean(dim=tuple(range(src.dim() - 1)), dtype=torch.promote_types(src.dtype, torch.float32))
            device = z.device
            fc_w = self.router_fc.weight.to(device)
            fc_b = self.router_fc.bias.to(device)
            embed_w = self.timestep_embed.weight.to(device)
            fc_out = F.linear(z.to(fc_w.dtype), fc_w, fc_b)
            if torch.compiler.is_compiling():
                hx_idx = self.__dict__.get("_hx_tidx")  # block compile: the row as a per-step index tensor (set_timestep)
                if hx_idx is not None:
                    return fc_out + embed_w.index_select(0, hx_idx.to(embed_w.device)).squeeze(0)
            t_idx = min(max(int(timestep) - 1, 0), self.tsm_num_timesteps - 1)
            # the row tsm.py:384 picks, by a Python int: torch.tensor(t, device=cuda) is a blocking copy per call
            return fc_out + embed_w[t_idx]
    return _original(TSMModule, "_compute_gates")(self, bottleneck, input_features, timestep)


def _tsm_delta(self, x: torch.Tensor, t, multiplier) -> torch.Tensor:
    """TSM delta for one timestep with one down and one up matmul (no dropout).

    The active experts' factors are concatenated: sum_j B_j A_j x == [B_1 .. B_m] [A_1; ..; A_m] x, the stage-1 sum of
    tsm.py:419-429 (bypass) and of tsm.py:494-516 + :607 (rebuild, without building the dense delta weight).
    scalar * scale * multiplier is folded into the concatenated up factor (out x m*rank) instead of scaling the
    (tokens x out) output. Stage 2 (tsm.py:441-482): G_j B_j A_j x == B_j (G_j A_j x), so the context columns of the
    bottleneck are multiplied by their gates (out of place) before the up matmul; the core columns feed the router as
    the core bottleneck did. Only the active experts are in the graph, so only they get gradients (none in stage 2,
    where set_stage froze them)."""
    experts = [self.experts[s][e] for s, e in enumerate(_tsm_experts(self, t))]
    dtype = x.dtype
    down = torch.cat([ex["down"].weight for ex in experts], 0).to(dtype)
    up = torch.cat([ex["up"].weight for ex in experts], 1)
    up = (up * (self.scalar.to(up.device) * (self.scale * multiplier))).to(dtype)
    if self.isconv:
        mid = self.down_op(x, down, None, **self.kw_dict_down)
    else:
        mid = F.linear(x, down)
    if self.tsm_stage == 2:
        ranks = [ex["down"].weight.shape[0] for ex in experts]
        axis = 1 if self.isconv else mid.dim() - 1
        gates = self._compute_gates(mid.narrow(axis, 0, ranks[0]), x, t)  # (num_context,)
        if all(r == ranks[0] for r in ranks):
            g = torch.cat([gates.new_ones(1), gates]).unsqueeze(1).expand(-1, ranks[0]).reshape(-1)
        else:
            g = torch.cat([gates.new_ones(ranks[0])] + [gates[j].expand(r) for j, r in enumerate(ranks[1:])])
        if self.isconv:
            g = g.view([1, -1] + [1] * (mid.dim() - 2))
        mid = (mid * g).to(mid.dtype)  # product in fp32 (the gates are fp32), one rounding as in tsm.py:471
    if self.isconv:
        return self.up_op(mid, up, None, **self.kw_dict_up)
    return F.linear(mid, up)


def _tsm_group_plan(self, ints: List[int], device):
    """Samples grouped by active experts (stage 1) or by timestep (stage 2: the timestep embedding differs per t), as
    [(t, rows)], plus the permutation back to batch order (None: already in order). rows is (start, length) for a
    contiguous run (narrow, no copy) or a device index tensor. Cached for the step: torch.tensor(..., device=cuda) is a
    blocking copy, which the old code made per group per module call."""
    key = (tuple(ints), self.tsm_stage, self.tsm_num_timesteps, tuple(self.tsm_n_scales), device)
    plan = _TSM_PLANS.get(key)
    if plan is not None:
        return plan
    groups: "OrderedDict[Any, Tuple[int, List[int]]]" = OrderedDict()
    for i, t in enumerate(ints):
        groups.setdefault(_tsm_experts(self, t) if self.tsm_stage == 1 else t, (t, []))[1].append(i)
    entries, order = [], []
    for t, idx in groups.values():
        if idx == list(range(idx[0], idx[0] + len(idx))):
            entries.append((t, (idx[0], len(idx))))
        else:
            entries.append((t, torch.tensor(idx, dtype=torch.long).to(device)))
        order.extend(idx)
    inverse = None
    if order != sorted(order):
        inv = [0] * len(order)
        for pos, i in enumerate(order):
            inv[i] = pos
        inverse = torch.tensor(inv, dtype=torch.long).to(device)
    if len(_TSM_PLANS) > 64:
        _TSM_PLANS.clear()
    plan = _TSM_PLANS[key] = (entries, inverse)
    return plan


def _tsm_grouped_delta(self, x: torch.Tensor, ints: List[int], multiplier) -> torch.Tensor:
    entries, inverse = _tsm_group_plan(self, ints, x.device)
    parts = []
    for t, rows in entries:
        xs = x.narrow(0, rows[0], rows[1]) if isinstance(rows, tuple) else x.index_select(0, rows)
        parts.append(_tsm_delta(self, xs, t, multiplier))
    delta = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
    return delta if inverse is None else delta.index_select(0, inverse)


def _tsm_split_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """org_forward(x) + the expert delta as a separate tensor (the fused path's fallback: conv, RamTorch base weights,
    delta dropout, dtype mixes the stock forward would not run either)."""
    base = self.org_forward(x, *args, **kwargs)
    ints = _TSM_SAMPLE_T.get(self.mask_group_id)
    if ints is not None and len(ints) > 1 and x.dim() >= 2 and x.shape[0] == len(ints) and len(set(ints)) > 1:
        # one expert (stage 1) / one timestep embedding (stage 2) per sample instead of one int per batch
        delta = _tsm_grouped_delta(self, x, ints, self.multiplier)
    else:
        # Stage 2 always takes the input path: rebuild mode feeds the router zeros (tsm.py:538-545), router_fc.weight
        # would never get a gradient. Stage 1 rebuild (tsm.py:604-608) is the same math without the dense delta weight.
        delta = _tsm_delta(self, x, self._get_timestep(), self.multiplier)
    if self.bypass_mode or self.tsm_stage == 2:
        delta = self.drop(delta)  # bypass dropout (tsm.py:588); the stage-1 rebuild forward has none (any batch)
    return base + delta.to(base.dtype)


# Fused TSM forward (2026-10-25): y = x W0^T + C, C = (x A^T) B^T + bias with A / B the concatenated active experts: the
# base GEMM takes the expert delta as its C input and the input gradient comes back the same way (addmm), with a
# hand-written backward. Against _tsm_split_forward that drops, per module call (forward, checkpoint recompute,
# backward), the separate delta tensor, the base + delta add, the second input-gradient add and ~10 autograd nodes. GPU
# A/B 2026-10-25 (RTX 5090, Anima 2.9B, @bjunbae 4-tier batch 1 x accumulation 8, dim 32, unsloth offload on, 16 steps):
# fused 12.64 vs split 12.48 vs stock LyCORIS 17.61 s/step, peak 21.7 GiB for both fixed paths - the fused call was
# speed-neutral there (the step is not bound by the TSM modules' launches); the ~30% over stock comes from the split path
# already. It stays the default for its numerics: under bf16 autocast it is as close to fp64 as the split path in stage 1
# and ~2x closer in stage 2 (router gradients). Same math; the output is rounded once instead of base and delta each
# (fp64: identical to the split path). Stage 2: G_j B_j A_j x == (B_j G_j)(A_j x), so the gates (per-batch scalars)
# scale the context columns of B; the router gets its gradient through that product. A mixed-timestep batch runs one
# fused call per expert group over its rows (the plan of _tsm_group_plan).
# Second round (2026-10-25, the GPU profile: a 1024 / 512 micro-step has a ~1.1 s host floor of which the 640 module
# calls are ~0.5 s of Python, not launches): the factor path (_tsm_factor_linear, the default) builds A = cat(downs) and
# B = cat(ups) * s (* gates) as autograd ops and hands them to _LowRankLinear (bf16 low-rank) / _TSMLinear (fp32
# low-rank), runs the stage-2 router through autograd on top of a second core GEMM, and its backward slices B's gradient
# into per-expert COLUMN blocks that AccumulateGrad clones. _TSMExpertLinear (below) takes the expert weights (and the
# router) themselves: one autograd node per call, the concatenation / scale / router / gate fold / casts inside it, the
# expert and router gradients returned contiguous in each parameter's layout. CPU, 32 modules in one checkpointed chain
# (Anima checkpoints per Block), per module call-set (forward + checkpoint recompute + backward): aten ops under bf16
# autocast (bf16 base, fp32 experts, m = 2) stage 1 43 -> 43 (fp32 low-rank 50 -> 44), stage 2 85 -> 58 (90 -> 60);
# Python time (fp32, 1 thread, Linear 64 -> 64 rank 4) stage 1 0.219 -> 0.197 ms (0.239 -> 0.210), stage 2
# 0.393 -> 0.294 ms (0.397 -> 0.294). On the GPU (the A/B setting above, cap 0.85) it held ~1.3 GiB more at the
# 2048-tier peak, ran out of memory where the factor path did not, and the run that got through was slower (16.1 vs
# 11.2 s/step), so the factor path stays the default and the Function is the opt-in tsm_expert_fn=True (_TSM_EXPERT_FN).
# The fp64 op counts of the test cases: test_lycoris_algo_fixes.py, test_tsm_fused_forward_is_taken_and_dispatches_less.
_TSM_FUSED = True  # network_args tsm_fused=False: _tsm_split_forward for every call
# network_args tsm_lowrank_fp32: True = the function runs x A^T, mid B^T and the expert gradients g^T mid / gmid^T x in
# fp32 with autocast off (the token reductions a bf16 GEMM rounds to 3 digits) and rounds the delta once into the bf16
# base GEMM; False = everything in the autocast dtype (as _LowRankLinear does for the LoRA family). Default (not given):
# stage 2 only - the user's choice 2026-10-25 after the GPU A/B: fp32 cost +11% step time and +3 GiB (torch 2.7.1 has no
# bf16-in / fp32-out mm, so the token-sized tensors are copied to fp32) for a stage-1 gain AdamW mostly normalises away
# (fp64 gradient error 0.86% -> 0.83% on the CPU toy), while stage 2 (router gates) went 4.2% -> 0.6% and has no
# expert-gradient GEMMs to pay for. The base GEMM and the input gradient stay bf16 either way.
_TSM_LOWRANK_FP32: Optional[bool] = None
_TSM_ONES1: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}  # the core expert's gate (1) per device / dtype


def _tsm_lowrank_fp32(self) -> bool:
    return _TSM_LOWRANK_FP32 if _TSM_LOWRANK_FP32 is not None else int(getattr(self, "tsm_stage", 1)) == 2


def _tsm_ones1(like: torch.Tensor) -> torch.Tensor:
    key = (like.device, like.dtype)
    one = _TSM_ONES1.get(key)
    if one is None:  # once per device / dtype, not torch.tensor(...) from Python data per call (a blocking copy on CUDA)
        one = _TSM_ONES1[key] = torch.ones(1, device=like.device, dtype=like.dtype)
    return one


class _NoCtx:
    """A no-op context (contextlib.nullcontext without the import): the autocast-off block of _TSMExpertLinear is
    entered only when autocast is on and would recast the fp32 low-rank GEMMs."""
    __slots__ = ()

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NOCTX = _NoCtx()


_TSM_COLIDX: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}  # (m, r, device) -> [0]*r + [1]*r + .. (m r,)


def _tsm_colidx(m: int, r: int, device) -> torch.Tensor:
    """Column -> expert index of the concatenated B, built once per (m, r, device) (no tensor from Python data per call)."""
    key = (m, r, device)
    idx = _TSM_COLIDX.get(key)
    if idx is None:
        idx = _TSM_COLIDX[key] = torch.arange(m, device=device).repeat_interleave(r)
    return idx


class _TSMExpertLinear(torch.autograd.Function):
    """x (t, in) -> x W0^T + C, C = (x A^T) B^T + bias, with A = [down_0; ..; down_{m-1}] (m r, in) and
    B = [up_0 .. up_{m-1}] * s (* the router gate per context expert in stage 2) (out, m r) of the m active experts,
    given as the expert weights themselves: concatenation, scale, router, gate fold and casts happen here, with no
    autograd node each. ld: the dtype of the low-rank GEMMs (x A^T, mid B^T, g^T mid, gmid^T x) and of the saved x -
    cdt itself (the bf16 variant: every value rounded where _LowRankLinear rounded it) or fp32 / fp64 (tsm_lowrank_fp32:
    the token reductions in fp32 and the delta rounded once into the cdt base GEMM, as _TSMLinear does). s: a Python
    float (scalar * scale * multiplier; k None) or the scalar Parameter / a buffer written in place (k = scale *
    multiplier applied here; the Parameter gets its gradient). Stage 2 (rw given): the router of tsm.py:325-390 as
    _tsm_compute_gates runs it - the tokens of the core bottleneck (the first r columns of mid; core=True) or of the
    input pooled in fp32, gates = rw z + rb + ew[ti] (ti the clamped t - 1, a Python int), and the context columns of B
    carry gate * s; rw / rb / ew get their gradients (ew's dense, one row set, as select's backward did). Saved: x in ld
    (the fp32 LayerNorm output itself for most Anima modules with the fp32 variant), W0 in cdt, A, B, mid (small); the
    unscaled B, the gate vector, z only when the scalar or the router need gradients. Backward: gb = g^T mid (out, m r)
    once; the up gradients leave it as one contiguous (m, out, r) buffer (cast, scale and relayout in one kernel) unbound
    into m tensors with the parameters' own strides - no column slices for AccumulateGrad to clone; the down gradients
    are the contiguous row blocks of ga = gmid^T x; gx = (gmid A -> cdt) + g W0 as one addmm. GEMMs nobody needs are
    skipped (stage 2: frozen experts -> no ga, gb only for the gate / scalar gradient; no input gradient -> no gmid)."""

    @staticmethod
    def forward(ctx, x, W0, bias, s, k, rw, rb, ew, ti, core, cdt, ld, *weights):
        m = len(weights) // 2
        downs, ups = weights[0::2], weights[1::2]
        out, r = ups[0].shape
        dev = x.device.type
        # autocast would recast the ld GEMMs to its dtype: switched off only when that differs (the bf16 variant feeds
        # it bf16 operands already; torch.autocast's enter/exit is ~1.7 us, x3 per call-set)
        noac = ld != cdt and torch.is_autocast_enabled(dev)
        with torch.autocast(dev, enabled=False) if noac else _NOCTX:
            A = downs[0] if m == 1 else torch.cat(downs, 0)
            Bf = ups[0] if m == 1 else torch.cat(ups, 1)  # (out, m r): column block j = up_j
            sk = s if k is None else s * k
            xl, Al = x.to(ld), A.to(ld)
            xc = xl if ld == cdt else x.to(cdt)
            W0c = W0.to(cdt)
            mid = torch.mm(xl, Al.t())
            gv = z = None
            if rw is None:
                Bs = Bf if k is None and sk == 1.0 else Bf * sk
            else:
                src = mid.narrow(1, 0, r) if core else x
                # pooling over the tokens (tsm.py:362) in promote(dtype, fp32), as _tsm_compute_gates
                z = src.mean(0, dtype=torch.promote_types(src.dtype, torch.float32)).to(rw.dtype)
                gates = torch.addmv(rb, rw, z) + ew[ti]  # (m - 1,): FC(z) + the embedding row (tsm.py:380-389)
                gv = torch.cat((_tsm_ones1(gates), gates))  # (m,): 1 for the core expert, the gate per context expert
                Bs = Bf * (gv * sk)[_tsm_colidx(m, r, gates.device)]
            Bl = Bs.to(ld)
            Cl = torch.mm(mid, Bl.t()) if bias is None else torch.addmm(bias.to(ld), mid, Bl.t())
            y = torch.addmm(Cl.to(cdt), xc, W0c.t())
        need = ctx.needs_input_grad
        keep = need[3] or need[5] or need[6] or need[7]  # scalar / router gradients: the unscaled B, gv, z
        ctx.save_for_backward(xl, W0c, Al, Bl, mid, Bf if keep else None, gv if keep else None,
                              sk if k is not None else None, z if keep else None, ew if need[7] else None)
        ctx.meta = (m, r, x.dtype, downs[0].dtype, ups[0].dtype, cdt, sk if k is None else None, k, noac, ti)
        return y

    @staticmethod
    def backward(ctx, g):
        xl, W0c, Al, Bl, mid, Bf, gv, skt, z, ew = ctx.saved_tensors
        m, r, xdt, adt, bdt, cdt, skf, k, noac, ti = ctx.meta
        with torch.autocast(g.device.type, enabled=False) if noac else _NOCTX:
            sk = skf if skt is None else skt
            need = ctx.needs_input_grad
            need_x, need_s, need_rw, need_rb, need_ew, need_w = need[0], need[3], need[5], need[6], need[7], need[12:]
            need_g = need_rw or need_rb or need_ew
            need_up, need_down = any(need_w[1::2]), any(need_w[0::2])
            gl = g.to(xl.dtype)
            gx = gs = grw = grb = gew = None
            gw = [None] * (2 * m)
            if need_up or need_s or need_g:
                gb = torch.mm(gl.t(), mid).to(bdt)  # (out, m r): the gradient of the scaled, gated B
                out = gb.shape[0]
                if need_s or need_g:
                    red = (gb.view(out, m, r) * Bf.view(out, m, r)).sum((0, 2))  # (m,) sum_j gb_j . up_j
                    if need_g:
                        gg = (red.narrow(0, 1, m - 1) * sk).to(z.dtype)  # the gates' gradient
                        if need_rw:
                            grw = torch.outer(gg, z)
                        if need_rb:
                            grb = gg
                        if need_ew:
                            gew = torch.select_scatter(torch.zeros_like(ew), gg, 0, ti)
                    if need_s:
                        gs = ((red.sum() if gv is None else torch.dot(red, gv.to(red.dtype))) * k).to(skt.dtype)
                if need_up:
                    buf = torch.empty((m, out, r), dtype=bdt, device=gb.device)  # each [j] has up_j's own strides
                    torch.mul(gb.view(out, m, r).transpose(0, 1), sk if gv is None else (gv * sk).view(m, 1, 1), out=buf)
                    for j, gu in enumerate(buf.unbind(0)):
                        if need_w[2 * j + 1]:
                            gw[2 * j + 1] = gu
            if need_x or need_down:
                gmid = torch.mm(gl, Bl)  # (t, m r) in ld
                if need_down:
                    ga = torch.mm(gmid.t(), xl).to(adt)  # (m r, in): row block j = down_j's gradient, contiguous
                    for j, gd in enumerate(ga.view(m, r, -1).unbind(0)):
                        if need_w[2 * j]:
                            gw[2 * j] = gd
                if need_x:
                    gx = torch.addmm(torch.mm(gmid, Al).to(cdt), g.to(cdt), W0c).to(xdt)
        return (gx, None, None, gs, None, grw, grb, gew, None, None, None, None, *gw)


def _tsm_fused_plan(self, x: torch.Tensor) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], torch.dtype]]:
    """(W0, bias, compute dtype) when the fused forward applies to this call, else None (the split forward runs)."""
    if not _TSM_FUSED or self.isconv or self.module_type != "linear" or not x.is_floating_point() or x.dim() < 2:
        return None
    if self.dropout and self.training and (self.bypass_mode or self.tsm_stage == 2):
        return None  # self.drop applies to the delta alone there (the stage-1 rebuild forward has no dropout)
    if self.__dict__.get("is_ramtorch_org", False):  # RamTorch streams the base weight in its own forward / backward
        return None
    org = self.org_module[0]
    w, b = org.weight, org.bias  # not RamTorch here: no get_org_weight_for_compute transfer
    if w is None or w.requires_grad or (b is not None and b.requires_grad):
        return None
    if w.dtype not in _FLOATS or (b is not None and b.dtype not in _FLOATS):
        return None  # --fp8_base: the stock org_forward casts it
    lowp = _lowp_autocast(x.device.type)
    if lowp is None and (w.dtype != x.dtype or (b is not None and b.dtype != x.dtype)):
        return None  # outside autocast the split path computes the base in the module dtype and the delta in x's
    ok = self.__dict__.get("_tsm_one_rank")
    if ok is None:  # _TSMExpertLinear views B as (out, m, r): one rank for every expert (tsm.py:258-260 always)
        ok = self._tsm_one_rank = len({ex["down"].weight.shape[0] for sc in self.experts for ex in sc}) == 1
    return (w, b, x.dtype if lowp is None else lowp) if ok else None


def _tsm_fused_ok(self, x: torch.Tensor) -> bool:
    return _tsm_fused_plan(self, x) is not None


def _tsm_scale(self) -> Tuple[Any, Optional[float]]:
    """(s, k) for the fused forward (_tsm_factor_linear and _TSMExpertLinear). The non-trainable scalar buffer (torch.tensor(1.0), tsm.py:244; apply_max_norm
    tsm.py:807 and load_weight_hook :784-787 write it in place) goes in as one Python float scale * multiplier * value,
    read as 1.0 while untouched and re-read with float() only when its storage or version counter changes (an in-place
    write, a module.to() copy after one) - no host sync and no 0-d tensor op per call. The scalar Parameter (use_scalar=True) goes in as the tensor with
    k = scale * multiplier applied inside the function, whether or not it currently requires grad."""
    sc = self.scalar
    k = float(self.scale) * self.multiplier
    if isinstance(sc, nn.Parameter):  # use_scalar: trained (or frozen) value, never assumed to be 1.0
        return sc, k
    if torch.compiler.is_compiling():
        # block compile: the data_ptr / version cache below is not traceable; hand the 0-d buffer itself to the graph
        # (read at run time, so an apply_max_norm / load_weight_hook write is seen; one fused multiply, no host sync)
        return sc * k, None
    key = (sc.data_ptr(), sc._version)
    cached = self.__dict__.get("_tsm_scalar_cache")
    if cached is None or cached[0] != key:
        # The buffer starts as torch.tensor(1.0) (tsm.py:244) and is not in the state dict: no host sync while it is
        # untouched. load_weight_hook (tsm.py:784-787, run by every load_state_dict: --network_weights / --resume) resets
        # it to 1.0 in place and apply_max_norm (tsm.py:807) scales it - both bump the version - and a module.to() copy
        # afterwards changes the storage: each makes the first call re-read the value once with float(). (A write
        # followed by a copy before the first call would still look untouched; the trainers never move the network
        # after training starts.)
        value = 1.0 if (cached is None and sc._version == 0) else float(sc)
        cached = self.__dict__["_tsm_scalar_cache"] = (key, value)
    return k * cached[1], None


def _tsm_active_weights(self, t) -> List[torch.Tensor]:
    """down_0, up_0, .., down_{m-1}, up_{m-1} of the experts active at t, read through the module dicts (each
    nn.Module.__getattr__ / ModuleList.__getitem__ is ~1 us; 12 of them per call on 640 modules)."""
    scales = self.experts._modules
    ws = []
    for si, e in enumerate(_tsm_experts(self, t)):
        ex = scales[str(si)]._modules[str(e)]._modules
        ws.append(ex["down"]._parameters["weight"])
        ws.append(ex["up"]._parameters["weight"])
    return ws


# network_args tsm_expert_fn=True: _TSMExpertLinear (above) for every call. Default False (GPU A/B 2026-10-25 16:4x, Anima
# 2.9B 4-tier batch 1 x accumulation 8, 16 steps, cap 0.85): the Function held ~1.3 GiB more at the 2048-tier peak
# (max_allocated 25.5 vs 24.3 GiB), ran out of memory at the cap where the factor path did not, and the one run that
# got past step 8 was slower (16.1 vs 11.2 s/step) - the CPU gain of the hand-written path (-10% Python per call) did not
# reach the GPU step. Kept as an opt-in for a later look (where the extra tensor comes from is not found yet).
_TSM_EXPERT_FN = False


class _TSMLinear(torch.autograd.Function):
    """x (t, in) -> x W0^T + C, C = (x A^T) B^T + bias with the low-rank part in ld (fp32, or fp64 for fp64 experts) and
    the frozen base GEMM in cdt (the autocast dtype): the fp32 low-rank variant of the factor path (A, B built outside as
    autograd ops), used when tsm_lowrank_fp32 applies. Saved: x in ld (the fp32 LayerNorm output itself for most Anima
    modules), W0 in cdt, A, B, mid (small). Backward: gb = g^T mid and ga = gmid^T x in ld (fp32 reductions over the
    tokens), gx = (gmid A -> cdt) + g W0 as one addmm."""

    @staticmethod
    def forward(ctx, x, W0, bias, A, B, cdt, ld):
        with torch.autocast(x.device.type, enabled=False):
            xl, Al, Bl = x.to(ld), A.to(ld), B.to(ld)
            W0c = W0.to(cdt)
            mid = torch.mm(xl, Al.t())
            Cl = torch.mm(mid, Bl.t()) if bias is None else torch.addmm(bias.to(ld), mid, Bl.t())
            y = torch.addmm(Cl.to(cdt), x.to(cdt), W0c.t())
        ctx.save_for_backward(xl, W0c, Al, Bl, mid)
        ctx.dtypes = (x.dtype, A.dtype, B.dtype, cdt)
        return y

    @staticmethod
    def backward(ctx, g):
        with torch.autocast(g.device.type, enabled=False):
            xl, W0c, Al, Bl, mid = ctx.saved_tensors
            xdt, adt, bdt, cdt = ctx.dtypes
            need_x, _, _, need_a, need_b, _, _ = ctx.needs_input_grad
            gl = g.to(xl.dtype)
            gx = ga = gb = None
            if need_b:
                gb = torch.mm(gl.t(), mid).to(bdt)
            if need_x or need_a:
                gmid = torch.mm(gl, Bl)  # (t, r) in ld
                if need_a:
                    ga = torch.mm(gmid.t(), xl).to(adt)
                if need_x:
                    gx = torch.addmm(torch.mm(gmid, Al).to(cdt), g.to(cdt), W0c).to(xdt)
        return gx, None, None, ga, gb, None, None


def _tsm_factor_linear(self, x2: torch.Tensor, t, W0, bias, s, k, cdt: torch.dtype, ld: torch.dtype,
                       weights: List[torch.Tensor]) -> torch.Tensor:
    """The factor path (the default): A = cat(downs), B = cat(ups) * scalar * scale * multiplier (* the router gates on
    the context columns in stage 2) as autograd ops, then _LowRankLinear (ld == cdt) or _TSMLinear (fp32 low-rank).
    Stage 2: G_j B_j A_j x == (B_j G_j)(A_j x), per-batch scalars, so a column scale of B; the router gets its gradient
    through that product. x2 (tokens, in) is the router input of this expert group."""
    A = torch.cat(weights[0::2], 0)
    B = torch.cat(weights[1::2], 1) * (s * k if k is not None else s)
    if self.tsm_stage == 2:
        bottleneck = None
        if self.tsm_router_input == "bottleneck":
            with torch.no_grad():  # the core bottleneck the router pools (detached in _tsm_compute_gates anyway)
                bottleneck = F.linear(x2.to(cdt), weights[0].to(cdt))
        gates = self._compute_gates(bottleneck, x2, t)  # (num_context,), fp32
        ranks = [w.shape[0] for w in weights[0::2]]
        g = torch.cat([gates.new_ones(ranks[0])] + [gates[j].expand(r) for j, r in enumerate(ranks[1:])])
        B = B * g.to(B.dtype)
    if ld == cdt:
        return _LowRankLinear.apply(x2, W0, bias, A, B, None, cdt, 1, False)
    return _TSMLinear.apply(x2, W0, bias, A, B, cdt, ld)


def _tsm_fused_linear(self, x2: torch.Tensor, t, W0, bias, s, k, cdt: torch.dtype) -> torch.Tensor:
    """One fused call for the rows x2 (tokens, in) of one expert group at timestep t."""
    weights = _tsm_active_weights(self, t)
    ld = torch.promote_types(weights[0].dtype, torch.float32) if _tsm_lowrank_fp32(self) else cdt
    if not _TSM_EXPERT_FN:
        return _tsm_factor_linear(self, x2, t, W0, bias, s, k, cdt, ld, weights)
    if self.tsm_stage != 2:
        return _TSMExpertLinear.apply(x2, W0, bias, s, k, None, None, None, 0, False, cdt, ld, *weights)
    mods = self._modules  # router_fc / timestep_embed through the dicts (as _tsm_active_weights)
    rf = mods["router_fc"]._parameters
    ti = min(max(int(t) - 1, 0), self.tsm_num_timesteps - 1)  # the embedding row tsm.py:384 picks
    ew = mods["timestep_embed"]._parameters["weight"]
    return _TSMExpertLinear.apply(x2, W0, bias, s, k, rf["weight"], rf["bias"], ew, ti,
                                  self.tsm_router_input == "bottleneck", cdt, ld, *weights)


# Mixed-timestep chunks (2026-10-25, third round): with per-sample timesteps a chunk of n samples has up to n distinct
# active expert sets, and the group loop below ran one fused call per group - on the GPU the step of the small tiers is
# bound by the Python per module call, so batching several samples into one forward bought nothing for TSM (a 4-row
# 1024 chunk cost 4 calls per module, like 4 single-row forwards; measured 1.44 s per image either way). The mixed
# path runs ONE call per module: A / B hold every expert active anywhere in the chunk (their union U, R = |U| x r
# columns), and a per-sample mask over the blocks (1 for the sample's own experts, 0 otherwise; in stage 2 the router
# gate of the sample's timestep on its context block) zeroes the other experts' columns of the bottleneck before the up
# GEMM. Same math as the group loop: sum_j M_bj B_j A_j x_b; an expert outside U is not in the graph (grad None, the
# optimizer skips it as before), one inside U gets exactly the gradient of the rows it is active for (the mask zeroes
# the others). Low-rank GEMMs run over R instead of each group's m r columns (R <= (groups + 1) r: 8% of the base
# GEMM at R = 160, in = 2048). network_args tsm_mixed=False restores the group loop.
_TSM_MIXED = True


class _TSMMaskedLinear(torch.autograd.Function):
    """x (t, in) -> x W0^T + ((x A^T) * M) B^T + bias for a chunk of n samples of tok tokens each (t = n tok): A (R, in)
    and B (out, R) are the concatenated experts active anywhere in the chunk, mask (n, R) picks each sample's own blocks
    (1 / 0, or the router gate in stage 2) and is expanded over the sample's tokens. ld: the low-rank dtype (cdt, or
    fp32 with tsm_lowrank_fp32, as _TSMLinear). Saved: x in ld, W0 in cdt, A, B, the unmasked bottleneck (t, R) and the
    mask. Backward: gb = g^T (mid * M); gmid = (g B) * M; ga = gmid^T x; gx = (gmid A -> cdt) + g W0 as one addmm;
    gmask = sum over the sample's tokens of (g B) * mid (the gates' gradient in stage 2)."""

    @staticmethod
    def forward(ctx, x, W0, bias, A, B, mask, tok, cdt, ld):
        n, R = mask.shape
        with torch.autocast(x.device.type, enabled=False):
            xl, Al, Bl, ml = x.to(ld), A.to(ld), B.to(ld), mask.to(ld)
            W0c = W0.to(cdt)
            mid = torch.mm(xl, Al.t())  # (t, R), every expert of U for every token
            midm = (mid.view(n, tok, R) * ml.unsqueeze(1)).view(-1, R)
            Cl = torch.mm(midm, Bl.t()) if bias is None else torch.addmm(bias.to(ld), midm, Bl.t())
            y = torch.addmm(Cl.to(cdt), x.to(cdt), W0c.t())
        ctx.save_for_backward(xl, W0c, Al, Bl, mid, ml)
        ctx.meta = (x.dtype, A.dtype, B.dtype, mask.dtype, cdt, n, tok)
        return y

    @staticmethod
    def backward(ctx, g):
        with torch.autocast(g.device.type, enabled=False):
            xl, W0c, Al, Bl, mid, ml = ctx.saved_tensors
            xdt, adt, bdt, mdt, cdt, n, tok = ctx.meta
            need_x, _, _, need_a, need_b, need_m = ctx.needs_input_grad[:6]
            R = mid.shape[1]
            gl = g.to(xl.dtype)
            gx = ga = gb = gm = None
            if need_b:
                midm = (mid.view(n, tok, R) * ml.unsqueeze(1)).view(-1, R)
                gb = torch.mm(gl.t(), midm).to(bdt)
            if need_x or need_a or need_m:
                gmid_raw = torch.mm(gl, Bl)  # (t, R) in ld
                if need_m:
                    gm = (gmid_raw * mid).view(n, tok, R).sum(1).to(mdt)  # (n, R)
                gmid = (gmid_raw.view(n, tok, R) * ml.unsqueeze(1)).view(-1, R)
                if need_a:
                    ga = torch.mm(gmid.t(), xl).to(adt)
                if need_x:
                    gx = torch.addmm(torch.mm(gmid, Al).to(cdt), g.to(cdt), W0c).to(xdt)
        return gx, None, None, ga, gb, gm, None, None, None


def _tsm_mixed_plan(self, ints: List[int], device) -> dict:
    """What _tsm_mixed_linear needs for a chunk with these per-sample timesteps, built once per step and shared by every
    module (cached with the group plans, cleared by set_timestep; torch.tensor(..., device=cuda) is a blocking copy):
    the experts active anywhere in the chunk in a fixed order (U), the (n, |U|) 0/1 block mask, each sample's core block;
    stage 2: the distinct timesteps with each sample's group, the group sizes, the embedding rows and each sample's
    context block positions."""
    key = ("mixed", tuple(ints), self.tsm_stage, self.tsm_num_timesteps, tuple(self.tsm_n_scales), device)
    plan = _TSM_PLANS.get(key)
    if plan is not None:
        return plan
    n = len(ints)
    per_row = [_tsm_experts(self, t) for t in ints]
    U = sorted({(s, e) for row in per_row for s, e in enumerate(row)})
    pos = {se: i for i, se in enumerate(U)}
    block = [[0.0] * len(U) for _ in range(n)]
    core, ctx = [], []
    for b, row in enumerate(per_row):
        for s, e in enumerate(row):
            block[b][pos[(s, e)]] = 1.0
        core.append(pos[(0, row[0])])
        ctx.append([pos[(s, e)] for s, e in enumerate(row) if s > 0])
    # single: every sample picks the same experts, so the plain fused call at ints[0] is exact - stage 1 only: in stage
    # 2 the gate depends on each sample's timestep (its embedding row, the pooling over its own group's tokens)
    plan = {"U": U, "nb": len(U), "single": self.tsm_stage != 2 and len(U) == len(self.tsm_n_scales),
            "block_mask": torch.tensor(block, dtype=torch.float32).to(device),
            "core_pos": torch.tensor(core, dtype=torch.long).to(device), "arange": torch.arange(n).to(device)}
    if self.tsm_stage == 2:
        distinct = list(OrderedDict.fromkeys(ints))
        gi = {t: i for i, t in enumerate(distinct)}
        T = self.tsm_num_timesteps
        plan["n_groups"] = len(distinct)
        plan["grp_idx"] = torch.tensor([gi[t] for t in ints], dtype=torch.long).to(device)
        plan["counts"] = torch.tensor([[float(ints.count(t))] for t in distinct], dtype=torch.float32).to(device)
        plan["t_idx"] = torch.tensor([min(max(int(t) - 1, 0), T - 1) for t in distinct], dtype=torch.long).to(device)
        plan["ctx_pos"] = torch.tensor(ctx, dtype=torch.long).to(device)  # (n, n_context)
    if len(_TSM_PLANS) > 64:
        _TSM_PLANS.clear()
    _TSM_PLANS[key] = plan
    return plan


def _tsm_mixed_gates(self, x2: torch.Tensor, A: torch.Tensor, plan: dict, cdt: torch.dtype, n: int, tok: int,
                     r: int) -> torch.Tensor:
    """Stage 2: the router gates per distinct timestep of the chunk, (groups, n_context) fp32 - _tsm_compute_gates'
    pooling (tsm.py:362-389) over each group's tokens, vectorised: the input rows, or the core bottleneck (one no-grad
    thin GEMM over the union A, each sample's core block picked); then FC(z) + the timestep embedding row."""
    pdt = torch.promote_types(cdt, torch.float32)  # the pooling dtype of _tsm_compute_gates
    with torch.no_grad():
        if self.tsm_router_input == "bottleneck":
            midc = F.linear(x2.to(cdt), A.to(cdt))  # (t, R)
            pooled = midc.view(n, tok, -1).mean(1, dtype=pdt)  # (n, R)
            z_rows = pooled.view(n, plan["nb"], r)[plan["arange"], plan["core_pos"]]  # (n, r)
        else:
            z_rows = x2.view(n, tok, -1).mean(1, dtype=pdt)  # (n, in)
        Z = torch.zeros(plan["n_groups"], z_rows.shape[1], dtype=pdt, device=x2.device)
        Z = Z.index_add_(0, plan["grp_idx"], z_rows) / plan["counts"].to(pdt)
    fc_w, fc_b, embed_w = self.router_fc.weight, self.router_fc.bias, self.timestep_embed.weight
    return F.linear(Z.to(fc_w.dtype), fc_w, fc_b) + embed_w[plan["t_idx"]]


def _tsm_mixed_linear(self, x2: torch.Tensor, ints: List[int], W0, bias, s, k, cdt: torch.dtype, tok: int) -> torch.Tensor:
    """One _TSMMaskedLinear call for a chunk whose samples have different timesteps (x2: (n tok, in))."""
    plan = _tsm_mixed_plan(self, ints, x2.device)
    if plan["single"]:  # stage 1, the timesteps differ but every sample picks the same experts: the plain fused call
        return _tsm_fused_linear(self, x2, ints[0], W0, bias, s, k, cdt)
    scales = self.experts._modules
    downs, ups = [], []
    for sc, e in plan["U"]:
        ex = scales[str(sc)]._modules[str(e)]._modules
        downs.append(ex["down"]._parameters["weight"])
        ups.append(ex["up"]._parameters["weight"])
    ld = torch.promote_types(downs[0].dtype, torch.float32) if _tsm_lowrank_fp32(self) else cdt
    A = torch.cat(downs, 0)  # (R, in)
    B = torch.cat(ups, 1) * (s * k if k is not None else s)  # (out, R)
    r = downs[0].shape[0]
    blocks = plan["block_mask"]  # (n, nb) 0 / 1, fp32
    if self.tsm_stage == 2:
        gates = _tsm_mixed_gates(self, x2, A, plan, cdt, len(ints), tok, r)  # (groups, n_context), the router's dtype
        blocks = blocks.to(gates.dtype)
        blocks = blocks * torch.ones_like(blocks).scatter(1, plan["ctx_pos"], gates[plan["grp_idx"]])
    mask = blocks.repeat_interleave(r, dim=1)  # (n, R)
    return _TSMMaskedLinear.apply(x2, W0, bias, A, B, mask, tok, cdt, ld)


def _tsm_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)
    plan = _tsm_fused_plan(self, x)
    if plan is None:
        return _tsm_split_forward(self, x, *args, **kwargs)
    W0, bias, cdt = plan
    s, k = _tsm_scale(self)
    shape = x.shape
    ints = _TSM_SAMPLE_T.get(self.mask_group_id)
    if ints is not None and len(ints) > 1 and shape[0] == len(ints) and len(set(ints)) > 1:
        if _TSM_MIXED and not _TSM_EXPERT_FN:
            y = _tsm_mixed_linear(self, x.reshape(-1, shape[-1]), ints, W0, bias, s, k, cdt, x[0].numel() // shape[-1])
            return y.view(*shape[:-1], y.shape[-1])
        entries, inverse = _tsm_group_plan(self, ints, x.device)
        parts = []
        for t, rows in entries:
            xg = x.narrow(0, rows[0], rows[1]) if isinstance(rows, tuple) else x.index_select(0, rows)
            yg = _tsm_fused_linear(self, xg.reshape(-1, shape[-1]), t, W0, bias, s, k, cdt)
            parts.append(yg.view(*xg.shape[:-1], yg.shape[-1]))
        y = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        return y if inverse is None else y.index_select(0, inverse)
    y = _tsm_fused_linear(self, x.reshape(-1, shape[-1]), self._get_timestep(), W0, bias, s, k, cdt)
    return y.view(*shape[:-1], y.shape[-1])


# ======================================================================================================================
# LoRA family (GoRA / RaLoRA / LoRA² / T-LoRA): factored forward
# ======================================================================================================================
# Rebuild mode (bypass_mode=False) computes F.linear(x, W'), W' = W0 + s*dW (dW = B A rebuilt dense per call, with fp32
# copies of W0 and x that bf16 autocast casts straight back) or, with DoRA, W' = (W0 + s*dW) * k where
# k = mult * (m / (|W0 + s*dW| + eps) - 1) + 1 per output row (wd_on_output=True) or input column (locon.py:873-895).
# Backward then runs a dense weight-gradient GEMM (as large as the forward GEMM) and the rebuild's own backward.
# The same function, factored (s = self.scale; without DoRA the multiplier is folded into the up factor too):
#   y = x W0^T + C,  C = (x A^T)(s B)^T + bias             (the base GEMM takes the low-rank part as its C input)
#   DoRA, rows: y = x (k * W0)^T + (x A^T)(s k B)^T + bias (k * W0: one weight-sized pass; k - 1 from the factors)
# with |W0_i + s (BA)_i|^2 = |W0_i|^2 + 2s B_i.(W0 A^T)_i + s^2 B_i (A A^T) B_i^T: one thin GEMM W0 A^T (bf16 under
# autocast, like the stock delta GEMM), r x r products in the DoRA dtype, no out x in tensor, no dense weight gradient
# (W0 is frozen). Under bf16 autocast the result is rounded as in the stock forward (the weight W0 or k * W0 once, the
# GEMM output once), but the delta is no longer rounded into the bf16 merged weight. DoRA per input column
# would need k * x rounded on top (measured ~20% larger bf16 error), so it keeps the stock dense rebuild, fed from these
# factors. Base weight and input go to the GEMMs as they are under bf16/fp16 autocast (the stock fp32 copies are value
# no-ops there); outside autocast everything is cast to the module dtype first, as the stock forward does.
# DoRA per output row (n == 1; ABBA too): one autograd function per call with a hand-written backward, the factored
# form above (_LowRankDoRA) or the dense rebuild without the fp32 activation copies (_DenseDoRAOut), whichever
# _lf_dora_dense estimates cheaper. On the GPU the step is host-bound (Python + kernel launches per module): measured
# on the RTX 5090 the split factored form (the norm in ordinary autograd ops, ~145 launches per call) made GoRA + DoRA
# 43% slower per step than the stock rebuild although its GPU work was smaller; the dense function wins on all of
# Anima's layers but the MLP's second (8192 wide input).
# RaLoRA blocks (n_split > 1): the same per diagonal block with batched matmuls. LoRA²: Lambda over all lora_dim
# positions, zero past the effective rank D (make_weight slices B, A to D): same values and gradients, D computed on the
# device (compute_effective_rank's nu.item() synced every call). Bypass mode: one factored delta, same as the stock.
# T-LoRA: P diag(lam m) Q - Pb diag(lamb m) Qb = [P lam - Pb lamb, Pb lamb] diag(m, m) [Q; Q - Qb]; both factors carry
# the small trained change, so bf16 no longer rounds the difference of two large products (the stock rebuild builds it
# dense and runs a second full GEMM; bypass subtracts two full products). Conv modules, Tucker, O-LoRA, rank dropout
# while training, GGPO, LyCORIS' torch_compile, a base weight that needs its gradient (the GoRA/RaLoRA pre-pass) and
# (rebuild mode) float8 base weights keep the code paths above.

_ARANGE: Dict[Any, torch.Tensor] = {}
_ONES_MASK: Dict[Any, torch.Tensor] = {}
_LORA2_TARGETS: Dict[Any, torch.Tensor] = {}


def _lowp_autocast(device_type: str) -> Optional[torch.dtype]:
    """The autocast dtype when bf16/fp16 autocast is on for this device type (F.linear/matmul then cast to it)."""
    if torch.is_autocast_enabled(device_type):
        dt = torch.get_autocast_dtype(device_type)
        if dt in (torch.bfloat16, torch.float16):
            return dt
    return None


def _locon_cls():
    from lycoris.modules.locon import LoConModule

    return LoConModule


def _lf_eligible(self, x: torch.Tensor) -> bool:
    if self.module_type != "linear" or self.tucker or getattr(self, "olora", False) or not x.is_floating_point():
        return False
    if self.rank_dropout and self.training:
        return False
    if "_forward_rebuild_core" in self.__dict__:  # network_args torch_compile=True: keep LyCORIS' compiled rebuild
        return False
    if (self._ggpo_enabled and self.training and getattr(self, "combined_weight_norms", None) is not None
            and getattr(self, "grad_norms", None) is not None):
        return False
    w, b = self.org_module[0].weight, self.org_module[0].bias
    if w is None or (not self.bypass_mode and (w.requires_grad or (b is not None and b.requires_grad))):
        return False  # GoRA/RaLoRA pre-pass: the stock path gives W0 its gradient
    if not self.bypass_mode and (w.dtype not in _FLOATS or (b is not None and b.dtype not in _FLOATS)):
        return False  # --fp8_base (train_network.py): the DoRA norm cannot promote float8; the stock path casts it
    kind = type(self).__name__
    if kind == "LoRA2Module" or (kind == "RaLoRAModule" and self.n_split > 1):
        return self.scalar_type in ("none", "scalar")  # those make_weight overrides multiply by self.scalar only
    return True


def _arange1(r: int, device, dtype) -> torch.Tensor:
    key = (r, device, dtype)
    t = _ARANGE.get(key)
    if t is None:
        t = _ARANGE[key] = torch.arange(1, r + 1, dtype=dtype).to(device)  # once per (r, device, dtype)
    return t


def _lora2_lam(self) -> torch.Tensor:
    """Lambda for all lora_dim rank positions, 0 past the effective rank D (lora2.py:233-239 slice B and A to D):
    compute_lambda_diag's values for j <= D, D = clamp(ceil(-ln(1-q)/nu), 1, lora_dim) (lora2_utils.py) in fp64 on
    the device, as Python computes it from nu.item()."""
    nu = self.lora2_nu
    r = int(self.lora_dim)
    idx = _arange1(r, nu.device, nu.dtype)
    lam = torch.exp(-nu * idx) - torch.exp(-nu * (idx + 1))
    with torch.no_grad():
        nu64 = nu.detach().to(torch.float64)
        d = torch.ceil((-math.log(1.0 - self.lora2_quantile)) / nu64).clamp(1, r)
        d = torch.where(nu64 <= 0, torch.full_like(d, r), d)
        keep = _arange1(r, nu.device, torch.float64) <= d
    return torch.where(keep, lam, torch.zeros_like(lam))


def _lf_factors(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """(A, B, n) with dW = B A as make_weight builds it, before self.scale (scalars, NoRA, orthogonalisation, the
    LoCon timestep mask, LoRA²'s Lambda folded in): A (r, in) / B (out, r) for n == 1; stacked diagonal blocks
    A (n, r, in/n) / B (n, out/n, r) for RaLoRA n_split = n > 1 (locon.py:1832-1878)."""
    kind = type(self).__name__
    if kind == "RaLoRAModule" and self.n_split > 1:
        A = torch.stack([self._maybe_nora_down(a) for a in self._mini_lora_A])
        B = torch.stack([self._orthogonalize(b) for b in self._mini_lora_B])
        if self.scalar_type == "scalar":
            B = B * self.scalar.to(B.device)
        return A, B, int(self.n_split)
    if kind == "LoRA2Module":  # lora2.py:225-243: no NoRA / orthogonalisation there
        A = self.lora_down.weight
        B = self.lora_up.weight * _lora2_lam(self).to(self.lora_up.weight.device)
        if self.scalar_type == "scalar":
            B = B * self.scalar.to(B.device)
        return A, B, 1
    A = self._maybe_nora_down(self.lora_down.weight)  # LoConModule._make_weight_single (GoRA, RaLoRA n_split == 1)
    if self.use_timestep_mask and self.training:
        A = A * self._timestep_mask.to(A).view(-1, 1)
    B = self._maybe_orthogonalize(self.lora_up.weight)
    st = self.scalar_type
    if st == "scalar":
        B = B * self.scalar.to(B.device)
    elif st in ("row", "row_column"):
        B = B * self.row_scalar.to(B.device).view(-1, 1)
    if st in ("column", "row_column"):
        A = A * self.col_scalar.to(A.device).view(1, -1)
    return A, B, 1


def _lf_blocks(x2: torch.Tensor, A: torch.Tensor, Bs: torch.Tensor, n: int) -> torch.Tensor:
    """blockdiag(Bs_i A_i) x2 for x2 (t, in) as (t, n, out/n) (a transposed view): one batched matmul pair."""
    t, mi = x2.shape[0], A.shape[-1]
    mid = torch.matmul(x2.reshape(t, n, mi).transpose(0, 1), A.transpose(1, 2))  # (n, t, r)
    return torch.matmul(mid, Bs.transpose(1, 2)).transpose(0, 1)  # (t, n, mo)


def _lf_delta(x: torch.Tensor, A: torch.Tensor, Bs: torch.Tensor, n: int) -> torch.Tensor:
    """blockdiag(Bs_i A_i) x for x (..., in) (bypass)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if n == 1:
        d = F.linear(F.linear(x2, A), Bs)
    else:
        d = _lf_blocks(x2, A, Bs, n).reshape(x2.shape[0], -1)
    return d.view(*shape[:-1], d.shape[-1])


class _LowRankLinear(torch.autograd.Function):
    """x (t, in) -> x W0'^T + C, C = blockdiag(B_i A_i) x + bias (n diagonal blocks; n == 1: B A x), in the compute
    dtype cdt; W0' = W0, or k * W0 rows with DoRA (e = k - 1 given; B then already carries k). This is the stock
    merged-weight GEMM with the low-rank part as its C input: the result is rounded once (and W0' once), as the stock
    rounds W' and the GEMM output, and the delta is no longer rounded into a bf16 weight. W0 and bias are frozen: no
    dense weight gradient. Backward: grad_x = g W0' + (low-rank part) as one GEMM with the small part as C; with DoRA
    grad_e = sum over tokens of g * (y - C) / k (= g * x W0^T; 0 where k == 0) and the up-factor product g^T (x A^T),
    both in e's dtype (the norm path cancels most of the latter outside, through k: a bf16-rounded product left its
    rounding in the small difference, LoRA² nu gradient 2x the stock error). y - C is rounded at the size of y and C:
    a row whose trained delta dwarfs its base row gets a proportionally coarser grad_e in bf16 than the stock path.
    up32 (LoRA²): the up-factor product from x A^T recomputed in fp32 (see _backward)."""

    @staticmethod
    def forward(ctx, x, W0, bias, A, B, e, cdt, n, up32=False):
        xc, W0c, Ac, Bc = x.to(cdt), W0.to(cdt), A.to(cdt), B.to(cdt)  # what bf16 autocast casts them to
        bc = None if bias is None else bias.to(cdt)
        t = xc.shape[0]
        if n == 1:
            mid = F.linear(xc, Ac)
            C = F.linear(mid, Bc, bc)
        else:
            mid = torch.matmul(xc.view(t, n, -1).transpose(0, 1), Ac.transpose(1, 2))  # (n, t, r)
            Ct = Bc.transpose(1, 2)
            Cb = torch.matmul(mid, Ct) if bc is None else torch.baddbmm(bc.view(n, 1, -1), mid, Ct)
            C = Cb.transpose(0, 1).reshape(t, -1)
        el = None if e is None else e.to(cdt)
        W = W0c if el is None else torch.addcmul(W0c, W0c, el.view(-1, 1))  # k * W0 rows, one rounding (as W')
        y = torch.addmm(C, xc, W.t())
        # DoRA keeps the output for grad_e: an in-place edit of it downstream raises autograd's version error
        ctx.save_for_backward(xc, W0c, Ac, Bc, mid, el, e, None if e is None else y, None if e is None else C,
                              A if up32 else None)
        ctx.dtypes = (x.dtype, A.dtype, B.dtype)
        ctx.n = n
        return y

    @staticmethod
    def backward(ctx, g):
        with torch.autocast(g.device.type, enabled=False):  # explicit dtypes (in case backward runs under autocast)
            return _LowRankLinear._backward(ctx, g)

    @staticmethod
    def _backward(ctx, g):
        xc, W0c, Ac, Bc, mid, el, e, y, C, A32 = ctx.saved_tensors
        xdt, adt, bdt = ctx.dtypes
        n = ctx.n
        need_x, _, _, need_a, need_b, need_e, _, _, _ = ctx.needs_input_grad
        g = g.to(xc.dtype)
        t = g.shape[0]
        gx = ga = gb = ge = None
        acc = None if e is None else torch.promote_types(g.dtype, e.dtype)
        gv = g if n == 1 else g.view(t, n, -1).transpose(0, 1)  # (n, t, out/n)
        if need_b:
            if A32 is not None:
                # LoRA²: nu's gradient sums this product over the out rows (through Lambda), where the bf16 rounding of
                # mid, shared by every row, does not average out (nu up to ~3x the stock error at Anima sizes): x A^T
                # again in fp32 from x and the unrounded A (an fp32 copy of x and g, one thin fp32 GEMM more)
                f = torch.promote_types(A32.dtype, torch.float32)
                xv = xc if n == 1 else xc.view(t, n, -1).transpose(0, 1)
                gb = gv.to(f).transpose(-1, -2) @ (xv.to(f) @ A32.to(f).transpose(-1, -2))
            elif e is None:
                gb = gv.transpose(-1, -2) @ mid
            else:
                gb = gv.to(acc).transpose(-1, -2) @ mid.to(acc)
            gb = gb.to(bdt)
        if need_x or need_a:
            gmid = gv @ Bc
            if need_a:
                xv = xc if n == 1 else xc.view(t, n, -1).transpose(0, 1)
                ga = (gmid.transpose(-1, -2) @ xv).to(adt)
            if need_x:
                Cg = gmid @ Ac if n == 1 else (gmid @ Ac).transpose(0, 1).reshape(t, -1)
                W = W0c if el is None else torch.addcmul(W0c, W0c, el.view(-1, 1))
                gx = torch.addmm(Cg, g, W).to(xdt)
        if need_e:
            gf = g.to(acc)
            # k == 0 (dora_scale 0 at multiplier 1) zeroed that row of W, so y - C holds no x W0^T there (was 0/0 = NaN
            # into dora_scale, A, B). Its true term sum g x W0_i^T is 0 when the base row is zero - the only way LyCORIS
            # starts a magnitude at 0 (dora_scale = ||W0_i||). A zero magnitude on a NONZERO base row (a file trained on
            # another base) gets 0 here instead: that row's dora_scale gradient lacks the base term (A, B and every
            # other row stay exact); recovering it would need the dense g^T x this function exists to avoid.
            k = e + 1
            nz = k != 0
            ge = torch.where(nz, ((gf * y).sum(0) - (gf * C).sum(0)) / torch.where(nz, k, 1.0), 0.0).to(e.dtype)
        return gx, None, None, ga, gb, ge, None, None, None


def _lf_dora_e(self, W0: torch.Tensor, A: torch.Tensor, B: torch.Tensor, n: int, device_type: str) -> torch.Tensor:
    """k - 1 of apply_weight_decompose (locon.py:873-895), per output row, for W' = W0 + scale * blockdiag(B_i A_i)
    without building W': |W'_i|^2 = |W0_i|^2 + 2s B_i.(W0 A^T)_i + s^2 B_i (A A^T) B_i^T. In fp32 at least: with a bf16
    dora_scale (--full_bf16) each term would be rounded on its own (the stock path rounds the one vector_norm)."""
    nd = torch.promote_types(self.dora_scale.dtype, torch.float32)
    s = float(self.scale)
    out_f, in_f = W0.shape
    if n == 1:
        Wb, A3, B3 = W0.unsqueeze(0), A.unsqueeze(0), B.unsqueeze(0)
    else:
        Wb = torch.diagonal(W0.view(n, out_f // n, n, in_f // n), dim1=0, dim2=2).permute(2, 0, 1)  # diagonal blocks
        A3, B3 = A, B
    W0A = torch.matmul(Wb, A3.transpose(1, 2))  # the one thin GEMM over W0: compute dtype (bf16 under autocast)
    with torch.autocast(device_type, enabled=False):  # the rest in nd
        Af, Bf = A3.to(nd), B3.to(nd)
        cross = (Bf * W0A.to(nd)).sum(-1).reshape(-1)
        quad = (torch.matmul(Bf, torch.matmul(Af, Af.transpose(1, 2))) * Bf).sum(-1).reshape(-1)
        n2 = torch.linalg.vector_norm(W0, dim=1, dtype=nd).square() + (2.0 * s) * cross + (s * s) * quad
        pos = n2 > 0  # a zero row: norm 0 and gradient 0, as vector_norm's backward gives there
        norm = torch.where(pos, torch.sqrt(torch.where(pos, n2, torch.ones_like(n2))), torch.zeros_like(n2))
        m = self.dora_scale.to(norm.device).reshape(-1)
        return self.multiplier_buf * (m / (norm + self._dora_eps) - 1)


class _LowRankDoRA(torch.autograd.Function):
    """_lf_dora_e + _LowRankLinear (n == 1, DoRA per output row) as one function with a hand-written backward: the same
    math and the same roundings (W0 A^T and the base / input-gradient GEMMs in cdt, the norm, k and the up-factor and
    k gradients in nd = fp32 at least), with no autograd node per small op. Measured on the RTX 5090 (Anima shapes,
    bf16 autocast, checkpointing): the split form was host-bound - ~145 launches and ~3.3 ms of host time per module
    call (forward, recompute, backward) against ~1.1 ms of GPU work and the stock path's 1.8 ms - and 640 modules made
    the GoRA training step 43% slower than the stock dense rebuild; this one ~120 launches, ~2.2 ms. y is the output
    (kept by the next op anyway): only it is saved, C is recomputed (one thin GEMM) instead of kept, which drops one
    activation-sized tensor per module (peak memory). Still more host time than _DenseDoRAOut: used only where the
    dense form's extra GPU work is larger (_lf_dora_dense).
    Inputs: x (t, in); W0 (out, in) and bias frozen; A (r, in), B (out, r) the factors without self.scale; d dora_scale;
    mult multiplier_buf; eps _dora_eps; s self.scale; cdt the compute dtype; up32 as in _LowRankLinear (LoRA²)."""

    @staticmethod
    def forward(ctx, x, W0, bias, A, B, d, mult, eps, s, cdt, up32):
        nd = torch.promote_types(d.dtype, torch.float32)
        with torch.autocast(x.device.type, enabled=False):
            xc, W0c, Ac = x.to(cdt), W0.to(cdt), A.to(cdt)
            bc = None if bias is None else bias.to(cdt)
            Af, Bf = A.to(nd), B.to(nd)
            W0A = torch.mm(W0c, Ac.t()).to(nd)  # the one thin GEMM over W0, in cdt (bf16 under autocast)
            AAt = torch.mm(Af, Af.t())
            # |W0_i|^2 + 2s B_i.(W0 A^T)_i + s^2 B_i (A A^T) B_i^T
            n2 = torch.linalg.vector_norm(W0, dim=1, dtype=nd).square()
            n2 = n2 + (torch.addmm(W0A, Bf, AAt, beta=2.0 * s, alpha=s * s) * Bf).sum(1)
            norm = n2.clamp_min(0).sqrt()  # a zero row: norm 0 (its gradient is set to 0 in backward)
            den = norm + eps
            q = d.reshape(-1).to(nd) / den
            e = mult * (q - 1)  # k - 1
            el = e.to(cdt)
            W = torch.addcmul(W0c, W0c, el.view(-1, 1))  # k * W0 rows, one rounding (as W')
            Bkc = ((B * s) * (e + 1).view(-1, 1)).to(cdt)  # s * k * B, rounded once to cdt
            mid = torch.mm(xc, Ac.t())
            C = F.linear(mid, Bkc, bc)
            y = torch.addmm(C, xc, W.t())
        ctx.save_for_backward(xc, W0c, Ac, A, B, Bkc, mid, bc, el, e, q, den, norm, W0A, AAt, mult, y)
        ctx.s, ctx.up32, ctx.xdt, ctx.ddt, ctx.dshape = s, up32, x.dtype, d.dtype, d.shape
        return y

    @staticmethod
    def backward(ctx, g):
        with torch.autocast(g.device.type, enabled=False):  # explicit dtypes (in case backward runs under autocast)
            return _LowRankDoRA._backward(ctx, g)

    @staticmethod
    def _backward(ctx, g):
        xc, W0c, Ac, A, B, Bkc, mid, bc, el, e, q, den, norm, W0A, AAt, mult, y = ctx.saved_tensors
        s = ctx.s
        nd = e.dtype
        need_x, need_a, need_b, need_d = (ctx.needs_input_grad[i] for i in (0, 3, 4, 5))
        g = g.to(xc.dtype)
        gx = ga = gb = gd = None
        gmid = torch.mm(g, Bkc)  # (t, r), cdt
        if need_x:
            W = torch.addcmul(W0c, W0c, el.view(-1, 1))  # recomputed (one weight-sized pass) instead of saved
            gx = torch.addmm(torch.mm(gmid, Ac), g, W).to(ctx.xdt)
        if not (need_a or need_b or need_d):
            return gx, None, None, None, None, None, None, None, None, None, None
        Bf = B.to(nd)
        gf = g.to(nd)
        if ctx.up32:
            # LoRA²: nu's gradient sums the up-factor product over the out rows (through Lambda), where the rounding of
            # mid, shared by every row, does not average out: x A^T again in fp32 from x and the unrounded A
            f = torch.promote_types(A.dtype, torch.float32)
            midf = torch.mm(xc.to(f), A.to(f).t()).to(nd)
        else:
            midf = mid.to(nd)
        gBk = torch.mm(gf.t(), midf)  # (out, r): gradient of s * k * B, in nd (the norm path cancels most of it)
        # sum over tokens of g * (x W0'^T) = g * (y - C), C recomputed; in nd
        P = gf * y
        P.addcmul_(gf, F.linear(mid, Bkc, bc), value=-1)
        zW = P.sum(0)
        k = e + 1
        # k == 0 zeroed that row of W0' (was 0/0): exact for a zero base row (the only way LyCORIS starts dora_scale at
        # 0); a zero magnitude on a NONZERO base row loses the base term of its own dora_scale gradient here, as in
        # _LowRankLinear (see there). _DenseDoRAOut has no such case.
        nz = k != 0
        gk = torch.where(nz, zW / torch.where(nz, k, 1.0), 0.0) + s * (Bf * gBk).sum(1)
        dq = gk * mult
        gdv = dq / den  # gradient of d
        pos = norm > 0  # a zero row: gradient 0, as vector_norm's backward gives there
        dn2 = torch.where(pos, (-gdv * q) / (2 * torch.where(pos, norm, 1.0)), 0.0)  # d|W'_i|^2
        if need_d:
            gd = gdv.to(ctx.ddt).view(ctx.dshape)
        if need_b:
            gb = gBk * (s * k).view(-1, 1) + 2 * dn2.view(-1, 1) * torch.addmm(W0A, Bf, AAt, beta=s, alpha=s * s)
            gb = gb.to(B.dtype)
        if need_a:
            Bn = dn2.view(-1, 1) * Bf
            ga = (torch.mm(gmid.t(), xc).to(nd) + (2 * s) * torch.mm(Bn.to(xc.dtype).t(), W0c).to(nd)
                  + (2 * s * s) * torch.mm(torch.mm(Bf.t(), Bn), A.to(nd)))
            ga = ga.to(A.dtype)
        return gx, None, None, ga, gb, gd, None, None, None, None, None


class _DenseDoRAOut(torch.autograd.Function):
    """The stock DoRA rebuild F.linear(x, W' , b), W' = apply_weight_decompose(W0 + s B A, mult) per output row
    (locon.py:873-895, 1269-1314), as one function with a hand-written backward: W0 + s B A, its row norms, k and the
    weight gradient's chain back to A, B and dora_scale in nd (fp32 at least, as the stock rebuild), W' rounded once to
    the compute dtype cdt, the two weight-sized GEMMs (forward / input gradient) and the merged-weight gradient
    g^T x in cdt as under bf16 autocast. Differences: B A in nd (the stock make_weight ran it in bf16 under autocast),
    no fp32 copies of the activation (autocast casts them straight back) and ~50 kernel launches without autograd
    nodes per call (forward, recompute, backward) instead of ~65 with them (the stock) or ~120 (_LowRankDoRA): measured
    1.6 ms per call at 2048->2048 x 4096 tokens (stock 1.9, the split factored form 3.0), 1.35 on adaLN (1.5 / 3.0).
    The cheaper form unless the weight is wide next to the activation work (see _lf_dora_dense). W0 comes as the stock
    merge sees it (not pre-rounded to cdt: W' is rounded once, here)."""

    @staticmethod
    def forward(ctx, x, W0, bias, A, B, d, mult, eps, s, cdt):
        nd = torch.promote_types(d.dtype, torch.float32)
        with torch.autocast(x.device.type, enabled=False):
            V = torch.addmm(W0.to(nd), B.to(nd), A.to(nd), alpha=s)  # W0 + s B A
            norm = torch.linalg.vector_norm(V, dim=1)
            den = norm + eps
            q = d.reshape(-1).to(nd) / den
            k = mult * (q - 1) + 1  # apply_weight_decompose's scale
            Wc = _mul_into(V, k.view(-1, 1), cdt)  # W' rounded once to the compute dtype
            xc = x.to(cdt)
            y = F.linear(xc, Wc, None if bias is None else bias.to(cdt))
        ctx.save_for_backward(xc, A, B, V, Wc, k, q, den, norm, mult)
        ctx.s, ctx.xdt, ctx.ddt, ctx.dshape = s, x.dtype, d.dtype, d.shape
        return y

    @staticmethod
    def backward(ctx, g):
        with torch.autocast(g.device.type, enabled=False):
            xc, A, B, V, Wc, k, q, den, norm, mult = ctx.saved_tensors
            s = ctx.s
            need_x, need_a, need_b, need_d = (ctx.needs_input_grad[i] for i in (0, 3, 4, 5))
            g = g.to(xc.dtype)
            gx = ga = gb = gd = None
            if need_x:
                gx = torch.mm(g, Wc).to(ctx.xdt)
            if not (need_a or need_b or need_d):
                return gx, None, None, None, None, None, None, None, None, None
            G = torch.mm(g.t(), xc)  # gradient of W' (cdt, as the stock merged-weight gradient)
            gq = (G * V).sum(1) * mult  # d/dq of sum G * V * k
            gdv = gq / den  # gradient of dora_scale
            if need_d:
                gd = gdv.to(ctx.ddt).view(ctx.dshape)
            pos = norm > 0  # a zero row: gradient 0, as vector_norm's backward gives there
            cn = torch.where(pos, (-gdv * q) / torch.where(pos, norm, 1.0), 0.0)  # d|V_i| / |V_i|
            dV = torch.mul(G, (k * s).view(-1, 1))  # s * gradient of V, in nd
            dV.addcmul_(V, (cn * s).view(-1, 1))
            if need_b:
                gb = torch.mm(dV, A.to(dV.dtype).t()).to(B.dtype)
            if need_a:
                ga = torch.mm(B.to(dV.dtype).t(), dV).to(A.dtype)
        return gx, None, None, ga, gb, gd, None, None, None, None


_LF_DORA_PATH: Optional[str] = None  # tests / benchmarks: "dense" or "factored" instead of the cost rule
# Path choice for DoRA per output row (_lf_dora_dense): both functions give the same result, this only picks the
# faster one by what each adds to one training step (gradient checkpointing: forward twice) on an RTX 5090. Dense: the
# merged-weight gradient GEMM g^T x, ~40 bytes per weight element (W0 + s B A, its norms, W', the fp32 chain back) and
# the four B A sized GEMMs in fp32. Factored: ~20 bytes per output element (the fp32 g, g * y, g * C), ~12 per weight
# element (W0 A^T, k * W0 three times), the rank-sized GEMMs and ~70 more kernel launches - a fixed ~0.8 ms of host
# time per call, which the step pays: the A/B was host-bound. Fitted to the per-call times (ms, dense / factored, bf16
# autocast, 4096 tokens): 2048->2048 1.59 / 2.19, 2048->8192 5.05 / 5.11, 8192->2048 4.15 / 3.74; adaLN (1 token)
# 1.35 / 1.87; block level (16 modules + attention) all-dense 35.3 ms vs factored on 8192->2048 34.9 ms.
_LF_MM_LOWP, _LF_MM_THIN, _LF_MM_FP32, _LF_BW, _LF_FACTORED_HOST = 150e12, 100e12, 60e12, 1.4e12, 0.78e-3


def _lf_dora_dense(t: int, out_f: int, in_f: int, rank: int, cdt: torch.dtype, device: torch.device) -> bool:
    """True: _DenseDoRAOut, False: _LowRankDoRA for a call with t tokens (both exact; this only picks the faster one).
    On the CPU (no launch-bound host; the tests) the factored form, as before."""
    if _LF_DORA_PATH is not None:
        return _LF_DORA_PATH == "dense"
    if device.type == "cpu":
        return False
    w = float(out_f) * in_f
    lowp = cdt in (torch.bfloat16, torch.float16)
    mm = _LF_MM_LOWP if lowp else _LF_MM_FP32
    dense = 2.0 * t * w / mm + 40.0 * w / _LF_BW + 8.0 * w * rank / _LF_MM_FP32
    factored = ((20.0 * t * out_f + 12.0 * w) / _LF_BW + 6.0 * t * rank * (in_f + out_f) / (_LF_MM_THIN if lowp else mm)
                + _LF_FACTORED_HOST)
    return dense <= factored


def _lf_forward(self, x: torch.Tensor) -> torch.Tensor:
    """GoRAModule / RaLoRAModule / LoRA2Module.forward (LoConModule.forward, locon.py:1362-1409, and the bypass
    diffs locon.py:1175-1205 / :1916-1990 / lora2.py:273-330) in factored form; see the section comment."""
    if not _lf_eligible(self, x):
        return _locon_cls().forward(self, x)
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x)
    A, B, n = _lf_factors(self)
    device_type = x.device.type
    lowp = _lowp_autocast(device_type)
    if self.bypass_mode:  # org_forward(x) + drop(delta): bypass ignores DoRA (locon.py:1266-1267)
        Bs = B * (float(self.scale) * self.multiplier)
        if lowp is None:
            A, Bs = A.to(x.dtype), Bs.to(x.dtype)
        return self.org_forward(x) + self.drop(_lf_delta(x, A, Bs, n))
    W0 = self.get_org_weight_for_compute(x.device)
    bias = self.get_org_bias_for_compute(x.device)
    dtype = self._cached_dtype
    if self.wd and not self.wd_on_output:
        # DoRA per input column: the factored form needs k * x rounded to bf16 on top of the two output roundings, one
        # full-size rounding more than the stock W' (measured: ~20% larger bf16 error). Keep the stock dense rebuild
        # (locon.py:1269-1314) with these factors, so LoRA² still makes no host sync.
        x, W0 = x.to(dtype), W0.to(dtype)
        if n == 1:
            dw = B @ A
        else:
            dw = torch.block_diag(*torch.matmul(B, A).unbind(0))
        weight = self.apply_weight_decompose(W0 + dw.to(dtype) * float(self.scale), self.multiplier_buf)
        return self._call_op(self.drop(x), weight, None if bias is None else bias.to(dtype))
    if lowp is None:  # the stock casts (locon.py:1388-1396): compute in the module dtype
        x, W0 = x.to(dtype), W0.to(dtype)
        bias = None if bias is None else bias.to(dtype)
        A, B = A.to(dtype), B.to(dtype)
    if self.wd and not isinstance(self.drop, nn.Identity):
        x = self.drop(x.to(dtype))  # DoRA input dropout, on the tensor the stock drops
    cdt = dtype if lowp is None else lowp
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    up32 = type(self).__name__ == "LoRA2Module"
    if not self.wd:
        y = _LowRankLinear.apply(x2, W0, bias, A, B * (float(self.scale) * self.multiplier), None, cdt, n, up32)
    elif n == 1:  # y = k * (x W0^T + delta) + b; the multiplier only enters through k
        d = self.dora_scale.to(x2.device)
        if _lf_dora_dense(x2.shape[0], W0.shape[0], W0.shape[1], A.shape[0], cdt, x2.device):
            y = _DenseDoRAOut.apply(x2, W0, bias, A, B, d, self.multiplier_buf, self._dora_eps, float(self.scale), cdt)
        else:
            y = _LowRankDoRA.apply(x2, W0, bias, A, B, d, self.multiplier_buf, self._dora_eps, float(self.scale), cdt,
                                   up32)
    else:
        e = _lf_dora_e(self, W0, A, B, n, device_type)
        y = _LowRankLinear.apply(x2, W0, bias, A, B * float(self.scale) * (e + 1).view(B.shape[:-1] + (1,)), e, cdt, n,
                                 up32)
    return y.view(*shape[:-1], y.shape[-1])


def _lora2_total_rank_reg_loss(cls):
    """lora2.py:177-190: sum |nu - nu_target| over the modules, as one stacked op (the stock loop made a CUDA scalar
    from Python per call: a blocking copy, plus ~3 tiny kernels per module). Same value up to summation order."""
    from lycoris.modules.lora2 import LoRA2Module

    mods = LoRA2Module._lora2_modules
    if not mods:
        return torch.tensor(0.0)
    nus = [m.lora2_nu for m in mods]
    dev, dt = nus[0].device, nus[0].dtype
    if any(v.device != dev or v.dtype != dt or v.dim() != 0 for v in nus):
        return _original(LoRA2Module, "get_total_rank_reg_loss").__func__()
    key = (dev, dt, tuple(float(m.nu_target) for m in mods))
    tgt = _LORA2_TARGETS.get(key)
    if tgt is None:
        if len(_LORA2_TARGETS) > 8:
            _LORA2_TARGETS.clear()
        tgt = _LORA2_TARGETS[key] = torch.tensor(key[2], dtype=dt).to(dev)
    return (torch.stack(nus) - tgt).abs().sum()


def _tlora_mask(self, device) -> torch.Tensor:
    """_get_mask (tlora.py:396-414) without the per-call CPU ones tensor + copy when no mask is set, and without a
    blocking host-to-device copy per call for a CPU mask: train_network.apply_tlora_mask stores compute_timestep_mask's
    CPU tensor over set_timestep's device copy, so every module call (and its checkpoint recompute) synced with the
    host. The first call of the step moves it and stores the device copy back (same values) for the others."""
    from lycoris.modules.tlora import get_timestep_mask, set_timestep_mask

    r = int(self.lora_dim)
    mask = get_timestep_mask(self.mask_group_id)
    if mask is None:
        key = (r, device)
        mask = _ONES_MASK.get(key)
        if mask is None:
            mask = _ONES_MASK[key] = torch.ones(1, r).to(device)
        return mask
    if mask.device != device and mask.device.type == "cpu":
        mask = mask.to(device)
        set_timestep_mask(mask, self.mask_group_id)
    if mask.shape[1] > r:
        mask = mask[:, :r]
    elif mask.shape[1] < r:
        mask = F.pad(mask, (0, r - mask.shape[1]), value=1.0)
    return mask.to(device)


def _tlora_fast_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """TLoraModule.forward (tlora.py:672-694 rebuild, :627-665 bypass / per-sample masks) with one down and one up
    matmul: delta = scalar * scale * mult * [P lam - Pb lamb, Pb lamb] diag(m, m) [Q; Q - Qb] x."""
    if self.isconv or not x.is_floating_point():
        return _tlora_forward(self, x, *args, **kwargs)
    mask = _tlora_mask(self, x.device)
    batched = mask.shape[0] > 1
    if batched and (x.dim() < 2 or x.shape[0] != mask.shape[0]):
        return _tlora_forward(self, x, *args, **kwargs)
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)
    bypass = self.bypass_mode or batched
    ft = torch.promote_types(self.p_layer.weight.dtype, torch.float32)  # --full_bf16: no bf16 product before the "-"
    P, Q = self.p_layer.weight.to(ft), self.q_layer.weight.to(ft)
    lam, Pb_lamb = self.lambda_layer.to(P.device, ft), self.base_p.to(ft) * self.base_lambda.to(P.device, ft)
    down = torch.cat([Q, Q - self.base_q.to(ft)], 0)  # (2r, in)
    up = torch.cat([P * lam - Pb_lamb, Pb_lamb], 1) * (self.scalar.to(P.device) * (float(self.scale) * self.multiplier))
    m2 = torch.cat([mask, mask], 1).to(up.dtype)  # (1 or B, 2r)
    if not batched:
        up = up * m2
    if _lowp_autocast(x.device.type) is None:
        down, up = down.to(x.dtype), up.to(x.dtype)
    x2 = x.reshape(-1, x.shape[-1])
    mid = F.linear(x2, down)
    if batched:
        mid = (mid.view(x.shape[0], -1, mid.shape[-1]) * m2.to(mid.dtype).unsqueeze(1)).view(mid.shape)
    if bypass:
        base = self.org_forward(x)
        return base + self.drop(F.linear(mid, up).view(*x.shape[:-1], up.shape[0]))
    base = self.org_forward(x, *args, **kwargs)
    if base.dtype != mid.dtype and _lowp_autocast(x.device.type) is None:
        return base + F.linear(mid, up).view(base.shape)
    return torch.addmm(base.reshape(-1, base.shape[-1]), mid, up.t()).view(base.shape)


# ======================================================================================================================
# LoRA / LoCon, OrthoLoRA, DyLoRA: factored training forward
# ======================================================================================================================
#
# With bypass_mode=False the stock forwards build the merged weight W0 + up @ down (out x in, fp32) per call and per
# checkpoint recompute, and backward computes the full (out x in) weight gradient: three big matmuls per step (forward,
# weight gradient, input gradient). y = x W0^T + (x down^T) up^T is the same math with two big matmuls (forward, input
# gradient) and thin ones; the base matmul is the last op that saves tensors, so a checkpoint recompute stops before it.
# Rounding under bf16 autocast (fp32 LoRA parameters): the down projection runs in fp32 (autocast off) on an fp32 copy of
# the input, the up projection in bf16, and the base matmul adds it in its epilogue (addmm: fp32 accumulation, one
# rounding). So the output is rounded twice (delta, base matmul) where the stock path rounds W0 + delta and the output,
# but the delta is no longer lost in bf16(W0 + delta). Not bit-identical: the same math with other roundings (fp64:
# identical; bf16 errors vs fp64 in tests/test_lycoris_algo_fixes.py). Without autocast (Anima's adaLN layers)
# everything runs in the module dtype, as the stock forward does.
# DoRA keeps the stock forward: its magnitude gradient needs x (W0 + delta)^T in backward, so a factored path has to
# recompute the base matmul under checkpointing and saves nothing (counted at 2048->8192, x [1, 4096, 2048]: 426 vs 417
# GFLOP, 3.6 vs 3.9 GB of other traffic, twice the kernels) while its input gradient is rounded once more.

_AUTOCAST_DTYPES = (torch.bfloat16, torch.float16)
_FAST_DTYPES = (torch.float32, torch.float64)  # module (LoRA parameter) dtypes; full_bf16/fp16 networks keep the stock path


def _autocast_dtype(x: torch.Tensor) -> Optional[torch.dtype]:
    dev = x.device.type
    if torch.is_autocast_enabled(dev):
        dt = torch.get_autocast_dtype(dev)
        if dt in _AUTOCAST_DTYPES:
            return dt
    return None


def _factored_linear(x, W0, bias, down, up, cd):
    """F.linear(x, W0 + up @ down, bias) (up carries every scale) without building the merged weight. cd: the module
    dtype (the stock forward's compute dtype)."""
    if _autocast_dtype(x) is None:
        x, W0, down, up = x.to(cd), W0.to(cd), down.to(cd), up.to(cd)
        bias = bias.to(cd) if bias is not None else None
    with torch.autocast(x.device.type, enabled=False):
        mid = F.linear(x.to(cd), down.to(cd))  # (..., r): fp32 down projection, fp32 gradients for down and the input
    zl = F.linear(mid, up, bias)
    out_f = W0.shape[0]
    y = torch.addmm(zl.reshape(-1, out_f), x.reshape(-1, x.shape[-1]), W0.t())  # delta + x W0^T, one rounding
    return y.view(*x.shape[:-1], out_f)


_LOCON_STOCK: Dict[str, Callable] = {}


def _locon_fast_ok(self) -> bool:
    """Plain LoRA/LoCon on a Linear in rebuild mode (static, cached per module): no Tucker, O-LoRA, rank dropout, GGPO,
    row/column scalars, DoRA or torch_compile. GoRA/RaLoRA (subclasses) keep their own forward."""
    ok = self.__dict__.get("_laf_fast")
    if ok is None:
        from lycoris.modules.locon import LoConModule

        ok = bool(
            type(self) is LoConModule
            and self.module_type == "linear"
            and not self.bypass_mode
            and not self.olora
            and not self.tucker
            and not self.rank_dropout
            and not getattr(self, "_ggpo_enabled", False)
            and self.scalar_type in ("none", "scalar")
            and not self.wd
            and "_forward_rebuild_core" not in self.__dict__
        )
        self.__dict__["_laf_fast"] = ok
    return ok


def _locon_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # Same math as LoConModule.forward -> _forward_rebuild_core (locon.py:1269-1314, 1362-1408) without the merged weight.
    if not (_locon_fast_ok(self) and self._cached_dtype in _FAST_DTYPES):
        return _LOCON_STOCK["forward"](self, x)
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x)
    down = self._maybe_nora_down(self.lora_down.weight)
    if self.use_timestep_mask and self.training:
        down = down * self._timestep_mask.to(down).view(-1, 1)
    up = self._maybe_orthogonalize(self.lora_up.weight)
    coef = self.scale
    if self.scalar_type == "scalar":
        coef = self.scalar.to(up) * coef
    W0 = self.get_org_weight_for_compute(x.device)
    bias = self.get_org_bias_for_compute(x.device)
    return _factored_linear(x, W0, bias, down, up * (coef * self.multiplier_buf), self._cached_dtype)


def _ortho_fast_ok(self) -> bool:
    ok = self.__dict__.get("_laf_fast")
    if ok is None:
        ok = bool(self.module_type == "linear" and not self.bypass_mode and not self.wd
                  and "_forward_rebuild_core" not in self.__dict__)
        self.__dict__["_laf_fast"] = ok
    return ok


def _ortho_effective_bases(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """OrthoLoRAModule._compute_effective_bases (ortholora.py:184-200) with linalg.solve_ex: torch.linalg.solve is
    solve_ex plus an error check that copies the LU info to the host, a blocking sync on every call (forward and the
    checkpoint recompute of every module, stock and flag path alike): measured on the RTX 5090, the host could then
    never run ahead of the GPU (OrthoLoRA step +7% with the factored forward, whose GPU work is smaller). I + (S - S^T)
    is never singular (eigenvalues 1 + i mu), so the check never fired; the same values bit for bit."""
    work = self.P_basis.dtype  # bf16
    skew = torch.stack([self.S_q, self.S_p])
    A = skew - skew.transpose(-2, -1)
    R = torch.linalg.solve_ex(self._eye_r + A, self._eye_r - A)[0]
    R_q = R[0].to(work)
    R_p = R[1].to(work)
    return R_q @ self.Q_basis, self.P_basis @ R_p


def _ortho_up(self, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """(down, up) of the OrthoLoRA delta up @ down in the module dtype: down = Q_eff, up = P_eff * lam * scalar * scale,
    with the effective bases and lam rounded to the bases' dtype as the stock make_weight does (ortholora.py:202-219)."""
    Q_eff, P_eff = self._compute_effective_bases()
    lam = self.lambda_layer.to(Q_eff.dtype)
    if self.use_timestep_mask and self.training:
        lam = lam * self._timestep_mask.to(lam)
    up = (P_eff * lam).to(self._cached_dtype) * (self.scalar.to(device) * self.scale)
    return Q_eff, up


def _ortho_get_diff_weight(self, multiplier=1.0, shape=None, device=None):
    # The stock product runs in bf16 (the bases' dtype, ortholora.py:217), and get_merged_weight then adds it to the
    # base in bf16 (:229-231): merges and checks were rounded to bf16. Same factors, product in the module dtype.
    down, up = _ortho_up(self, self.lambda_layer.device)
    diff = (up * multiplier) @ down.to(up.dtype)
    diff = diff.view(self.shape if shape is None else shape)
    return (diff if device is None else diff.to(device)), None


def _ortho_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # OrthoLoRAModule.forward -> _forward_rebuild_core (ortholora.py:273-299) without the dense delta weight: the same
    # bf16 effective bases (_compute_effective_bases), delta = (P_eff * lam) @ Q_eff * scalar * scale.
    from lycoris.modules.ortholora import OrthoLoRAModule

    if not (_ortho_fast_ok(self) and self._cached_dtype in _FAST_DTYPES):
        return _original(OrthoLoRAModule, "forward")(self, x, *args, **kwargs)
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x, *args, **kwargs)
    Q_eff, up = _ortho_up(self, x.device)
    W0 = self.get_org_weight_for_compute(x.device)
    bias = self.get_org_bias_for_compute(x.device)
    return _factored_linear(x, W0, bias, Q_eff, up * self.multiplier, self._cached_dtype)


# ---------------------------------------------------------------------------------------------------------------- DyLoRA


def _dylora_block(self) -> int:
    """Index b of the highest trained block (blocks 0..b, rank (b+1)*block_size).

    Training: one random block per forward as LyCORIS (dylora.py:114-116), but a checkpoint recompute inside backward
    reuses the forward's block: the stock code drew a new one there (Python's random is not part of the checkpoint RNG
    state), so the recomputed graph had other shapes (CheckpointError) or trained another block than the loss used.
    Not training (sampling, merges): every block (dylora.py:155-162 sampled a random rank for each image too)."""
    if not self.training:
        return self.block_count - 1
    if torch._C._current_graph_task_id() != -1:  # inside backward: gradient checkpointing re-runs the forward
        b = self.__dict__.get("_laf_dylora_b")
        if b is not None:
            return b
    import random

    b = random.randint(0, self.block_count - 1)
    self.__dict__["_laf_dylora_b"] = b
    return b


def _dylora_factors(self, b: int):
    """(down (k, in*), up (out, k), alpha / (b + 1)) for blocks 0..b: DyLoraModule.get_weight(b * block_size)
    (dylora.py:103-112). Blocks below b are frozen for this step (.data there, detach here)."""
    down = torch.cat([t.detach() for t in self.down_list[:b]] + [self.down_list[b]], 0)
    up = torch.cat([t.detach() for t in self.up_list[:b]] + [self.up_list[b]], 1)
    return down, up, self.alpha / (b + 1)


def _dylora_get_diff_weight(self, multiplier=1, shape=None, device=None, rank=None):
    # rank None: all blocks (dylora.py:118-121 drew a random rank, so merges and saved-weight checks were random). A
    # given rank r now means ceil(r / block_size) blocks (get_weight(r) used one block more, and for r = lora_dim
    # alpha / (block_count + 1) instead of alpha / block_count).
    if rank is None:
        b = self.block_count - 1
    else:
        b = min(max(int(math.ceil(int(rank) / self.block_size)) - 1, 0), self.block_count - 1)
    down, up, coef = _dylora_factors(self, b)
    w = up @ (down * (coef * multiplier))
    if device is not None:
        w = w.to(device)
    w = w.reshape(up.shape[0], *self.shape[1:]) if shape is None else w.reshape(shape)
    return w, None


def _dylora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    # dylora.py:155-168 crashed under Anima's autocast-off adaLN layers (bf16 input, fp32 merged weight) and its bypass
    # path ignored the scale and the multiplier (dylora.py:142-150, crashes below full rank). Here: the stock rebuild
    # math (W0 + up @ (down * alpha / (b + 1) * multiplier)) as two thin matmuls for Linear layers.
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x)
    b = _dylora_block(self)
    down, up, coef = _dylora_factors(self, b)
    down = down * (coef * self.multiplier)
    k = down.shape[0]
    cd = self._cached_dtype
    if self.module_type == "linear":
        if self.bypass_mode:
            ac = _autocast_dtype(x)
            base = self.org_forward(x)
            if ac is None:
                down, up = down.to(x.dtype), up.to(x.dtype)
            return base + F.linear(F.linear(x, down), up).to(base.dtype)
        W0 = self.get_org_weight_for_compute(x.device)
        return _factored_linear(x, W0, self.get_org_bias_for_compute(x.device), down, up, cd)
    # conv: dense delta as the stock rebuild, in the module dtype (no dtype mismatch without autocast)
    if self.bypass_mode:
        base = self.org_forward(x)
        dt = x.dtype if _autocast_dtype(x) is None else down.dtype
        mid = self.op(x, down.reshape(k, *self.shape[1:]).to(dt), **self.kw_dict)
        return base + self.op(mid, up.reshape(-1, k, *([1] * (len(self.shape) - 2))).to(dt)).to(base.dtype)
    w = (up @ down).reshape(up.shape[0], *self.shape[1:])
    W0 = self.get_org_weight_for_compute(x.device).to(cd)
    bias = self.get_org_bias_for_compute(x.device)
    bias = bias.to(cd) if bias is not None else None
    return self.op(x.to(cd), W0 + w.to(cd), bias, **self.kw_dict)


def _dylora_load_weight_prehook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                error_msgs):
    # The saved state (custom_state_dict, dylora.py:90-101) is lora_up / lora_down / alpha but the parameters are
    # up_list.N / down_list.N: --resume (strict) failed and --network_weights loaded nothing. Split them into blocks.
    up = state_dict.pop(prefix + "lora_up.weight", None)
    down = state_dict.pop(prefix + "lora_down.weight", None)
    if up is None and down is None:
        return
    if up is None or down is None:
        error_msgs.append(f"{prefix}: lora_up.weight and lora_down.weight must both be present")
        return
    up2 = up.reshape(up.shape[0], -1)
    down2 = down.reshape(down.shape[0], -1)
    if up2.shape[1] != self.lora_dim or down2.shape[0] != self.lora_dim:
        error_msgs.append(f"{prefix}: DyLoRA rank {down2.shape[0]} in the file, {self.lora_dim} in the network")
        return
    bs = self.block_size
    for i in range(self.block_count):
        state_dict[f"{prefix}up_list.{i}"] = up2[:, i * bs:(i + 1) * bs]
        state_dict[f"{prefix}down_list.{i}"] = down2[i * bs:(i + 1) * bs]


# ----------------------------------------------------------------------------------------------------------------- PiSSA


def _locon_init_svd_segment(self, segment):
    """PiSSA (init_weights=pissa): the stock init subtracts up0 @ down0 from the base unscaled (locon.py:404-407) while
    the forward adds scale * scalar * up @ down, so with alpha != dim (or use_scalar) the model changed at step 0. Here
    the initial factors are divided by sqrt(scale * scalar), so the adapter starts exactly at the removed component, and
    the initial factors are kept in fp32 (the stock copy was rounded to the base dtype). Applies before NoRA's init
    (locon.py:334-335), which keeps the same invariant."""
    from lycoris.modules.locon import LoConModule

    _original(LoConModule, "_init_svd_segment")(self, segment)
    if not (type(self) is LoConModule and segment == "top" and self.is_pissa and self.pissa_A_init is not None):
        return
    eff = float(self.scale)
    scalar0 = 1.0
    if self.scalar_type == "scalar":
        scalar0 = float(self.scalar.detach())
        eff *= scalar0
    elif self.scalar_type != "none":
        logger.warning(f"{self.lora_name}: PiSSA with scalar_type={self.scalar_type}: initial factors not rescaled")
        eff = 1.0
    with torch.no_grad():
        if eff > 0 and eff != 1.0:
            f = 1.0 / math.sqrt(eff)
            self.lora_up.weight.mul_(f)
            self.lora_down.weight.mul_(f)
        elif eff <= 0:
            logger.warning(f"{self.lora_name}: PiSSA with scale * scalar = {eff}: initial factors not rescaled")
        # pissa_A_init @ pissa_B_init * scale == the removed component (the scalar's initial value is in pissa_A_init)
        self.pissa_A_init = self.lora_up.weight.detach().reshape(self.lora_up.weight.shape[0], -1) * scalar0
        self.pissa_B_init = self.lora_down.weight.detach().reshape(self.lora_down.weight.shape[0], -1).clone()


def _locon_load_weight_prehook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                               error_msgs):
    # PiSSA keys are plain attributes, not buffers: a state with them (pissa_convert=False, which the flag uses for the
    # accelerate state) failed strict loading with "unexpected keys". Take them here.
    from lycoris.modules.locon import LoConModule

    if type(self) is not LoConModule:
        return
    state_dict.pop(prefix + "pissa_converted", None)
    for name in ("pissa_A_init", "pissa_B_init"):
        t = state_dict.pop(prefix + name, None)
        if t is not None and getattr(self, "is_pissa", False):
            cur = getattr(self, name, None)
            dev = cur.device if isinstance(cur, torch.Tensor) else self.lora_up.weight.device
            setattr(self, name, t.detach().to(device=dev, dtype=torch.float32).clone())


# ======================================================================================================================
# LoKr / GLoRA / IA3
# ======================================================================================================================


def _lokr_cls():
    from lycoris.modules.lokr import LokrModule

    return LokrModule


def _lokr_dora_at(self, weight: torch.Tensor, diff: torch.Tensor, multiplier: float) -> torch.Tensor:
    """DoRA weight at multiplier m: W0 + m (W1 - W0), W1 = the m=1 weight training uses (apply_weight_decompose(W0 + dW,
    1), lokr.py:656-674). Stock keeps the whole dW for every m ((W0 + dW) (m (s - 1) + 1), lokr.py:651/:851): at m=0
    (differential output preservation, iLECO, ADDifT targets) it is not the base model. Forge and sam-extra's "lycoris"
    DoRA mode interpolate like this."""
    w1 = self.apply_weight_decompose(weight + diff, 1.0)
    weight = weight.to(w1.dtype)
    return weight + (w1 - weight) * float(multiplier)


_LOKR_STOCK: Dict[str, Callable] = {}  # LokrModule.forward as installed before the patch, set by apply_module_patches


def _lokr_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    m = self._multiplier  # python float, kept equal to multiplier_buf by the property setter (base.py:747-751)
    if not self.wd or self.bypass_mode or m == 1.0:
        # multiplier 1 (every training step): the forward that was installed (the fork's fast LoKr+DoRA path or stock),
        # looked up once when patching (an _original() scan here cost ~1 us on every module call of the production runs)
        return _LOKR_STOCK["forward"](self, x, *args, **kwargs)
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x)
    if m == 0.0:
        return self.org_forward(x)  # exactly the base layer
    # lokr.py:858-874 with the merge of _lokr_dora_at
    x = x.to(self._cached_dtype)
    org_weight = self.get_org_weight_for_compute(x.device).data.to(self._cached_dtype)
    bias = self.get_org_bias_for_compute(x.device)
    if bias is not None:
        bias = bias.to(x.dtype)
    diff = self.get_weight(self.shape).to(self._cached_dtype) * self.scalar
    return self._call_op(x, _lokr_dora_at(self, org_weight, diff, m), bias)


def _lokr_get_merged_weight(self, multiplier=1, shape=None, device=None):
    if not self.wd or float(multiplier) == 1.0:
        return _original(_lokr_cls(), "get_merged_weight")(self, multiplier, shape, device)
    diff = self.get_diff_weight(multiplier=1, shape=shape, device=device)[0]
    weight = self.get_org_weight_for_compute(diff.device)
    if weight.dtype != diff.dtype:
        weight = weight.to(diff.dtype)
    return _lokr_dora_at(self, weight, diff, float(multiplier)), None


def _glora_cls():
    from lycoris.modules.glora import GLoRAModule

    return GLoRAModule


def _glora_autocast_dtype(x: torch.Tensor) -> Optional[torch.dtype]:
    """The dtype F.linear casts x to here (autocast on), None where it runs in x.dtype."""
    dev = x.device.type
    return torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else None


def _linear_as_autocast(x, weight, bias, cdtype):
    """F.linear(x, weight, bias) as it runs under autocast (fp32 inputs cast to cdtype; same kernel), returning the cast
    x and weight for the backward."""
    if cdtype is None:
        return F.linear(x, weight, bias), x, weight
    xc, wc = x.to(cdtype), weight.to(cdtype)
    bc = None if bias is None else bias.to(cdtype)
    with torch.autocast(x.device.type, enabled=False):
        return F.linear(xc, wc, bc), xc, wc


def _input_grad(g: torch.Tensor, wc: torch.Tensor, x_dtype: torch.dtype) -> torch.Tensor:
    """d/dx of F.linear(x, wc): grad_output @ weight, the product autograd's linear backward runs, then the backward of
    the autocast cast (back to x.dtype)."""
    with torch.autocast(g.device.type, enabled=False):
        return g.reshape(-1, g.shape[-1]).mm(wc).view(*g.shape[:-1], wc.shape[1]).to(x_dtype)


class _GLoRADenseLinear(torch.autograd.Function):
    """y = F.linear(x, W + k (W A1 A2 + B1 B2), b), k = scale * scalar * multiplier: the stock rebuild forward
    (glora.py:356-395, same float ops and autocast casts: same output) with a backward that never forms the full weight
    gradient G = g^T x (a tokens x out x in matmul in the stock backward, and again a dense chain through W A1 A2).
    The factor gradients are G's contractions with the factors, as thin fp32 matmuls over the tokens:
        dB1 = k G B2^T = k g^T (x B2^T)          dB2 = k B1^T G = k (g B1)^T x
        dA1 = k W^T G A2^T = k W^T (g^T (x A2^T)) dA2 = k (W A1)^T G = k (g W A1)^T x
        dscalar = scale * multiplier * <G, W A1 A2 + B1 B2>
    and the input gradient is the stock g W' (same product)."""

    @staticmethod
    def forward(ctx, x, wa1, wa2, wb1, wb2, scalar, weight, bias, scale, multiplier, cdtype):
        orig = weight.to(x.dtype)  # glora.py:367 (x is already in the module dtype)
        wb = (wb1 @ wb2).view(*orig.shape)
        w_wa2 = (orig @ wa1) @ wa2
        diff_w = (wb + w_wa2) * scale * scalar * multiplier
        wp = orig.add(diff_w)
        y, xc, wpc = _linear_as_autocast(x, wp, bias, cdtype)
        ctx.save_for_backward(xc, wpc, weight, wa1, wa2, wb1, wb2, scalar, multiplier)
        ctx.scale = scale
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, g):
        xc, wpc, weight, wa1, wa2, wb1, wb2, scalar, multiplier = ctx.saved_tensors
        need = ctx.needs_input_grad
        grad_x = _input_grad(g, wpc, ctx.x_dtype) if need[0] else None
        d_a1 = d_a2 = d_b1 = d_b2 = d_scalar = None
        if any(need[1:6]):
            pdt = wa1.dtype
            r = wa2.shape[0]
            with torch.autocast(g.device.type, enabled=False):
                g2 = g.reshape(-1, g.shape[-1]).to(pdt)
                x2 = xc.reshape(-1, xc.shape[-1]).to(pdt)
                w = weight.to(pdt)
                w_a1 = w @ wa1  # (out, r)
                p = x2 @ torch.cat([wa2, wb2], 0).t()  # (T, 2r): x A2^T | x B2^T
                q = g2 @ torch.cat([wb1, w_a1], 1)  # (T, 2r): g B1 | g W A1
                gp = g2.t() @ p  # (out, 2r): G A2^T | G B2^T
                qx = q.t() @ x2  # (2r, in): B1^T G | (W A1)^T G
                sm = multiplier.to(pdt) * float(ctx.scale)
                k = scalar.to(pdt) * sm
                if need[1]:
                    d_a1 = (w.t() @ gp[:, :r]) * k
                if need[2]:
                    d_a2 = qx[r:] * k
                if need[3]:
                    d_b1 = gp[:, r:] * k
                if need[4]:
                    d_b2 = qx[:r] * k
                if need[5]:
                    # <G, B1 B2> = sum(g B1 * x B2^T), <G, W A1 A2> = sum(g W A1 * x A2^T)
                    d_scalar = ((q[:, :r] * p[:, r:]).sum() + (q[:, r:] * p[:, :r]).sum()) * sm
                    d_scalar = d_scalar.reshape(scalar.shape).to(scalar.dtype)
        return grad_x, d_a1, d_a2, d_b1, d_b2, d_scalar, None, None, None, None, None


def _glora_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Stock rebuild-mode GLoRA output for linears (glora.py:356-417) through _GLoRADenseLinear: same forward, one
    tokens x out x in matmul less in the backward."""
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.org_forward(x)
    cdtype = _glora_autocast_dtype(x)
    if self.bypass_mode or self.module_type != "linear" or self.tucker or (
        cdtype is not None and self._cached_dtype == torch.float64
    ):
        return _original(_glora_cls(), "forward")(self, x, *args, **kwargs)
    x = x.to(self._cached_dtype)
    weight = self.get_org_weight_for_compute(x.device)
    bias = self.get_org_bias_for_compute(x.device)
    if bias is not None:
        bias = bias.to(x.dtype)
    if self.dropout:
        x = self.drop(x)  # glora.py:388-389: dropout on the layer input in rebuild mode
    wa1 = self._orthogonalize(self.a1.weight).view(self.a1.weight.size(0), -1)
    wa2 = self._orthogonalize(self.a2.weight).view(-1, self.a2.weight.size(1))
    wb1 = self._orthogonalize(self.b1.weight).view(self.b1.weight.size(0), -1)
    wb2 = self._orthogonalize(self.b2.weight).view(self.b2.weight.size(0), -1)
    return _GLoRADenseLinear.apply(x, wa1, wa2, wb1, wb2, self.scalar.to(device=x.device), weight, bias, self.scale,
                                   self.multiplier_buf, cdtype)


def _glora_bypass_forward(self, x, scale=1, diff=False):
    """glora.py:292-354 with alpha/rank and the scalar applied once: stock multiplies both branches by scale once in the
    down step and by self.scale again at the end (glora.py:298/:306/:350-354), drops the learnable scalar (use_scalar:
    never trained) and adds the base bias to the diff (bypass_forward_diff = W(A x) + b + B x)."""
    c = self.scale * scale * self.scalar.to(x.device, x.dtype)
    wa1 = self._orthogonalize(self.a1.weight).to(x.device, dtype=x.dtype)
    wa2 = self._orthogonalize(self.a2.weight).to(x.device, dtype=x.dtype)
    wb1 = self._orthogonalize(self.b1.weight).to(x.device, dtype=x.dtype)
    wb2 = self._orthogonalize(self.b2.weight).to(x.device, dtype=x.dtype)
    ax_mid = self.down_op(x, wa2) * c
    if self.isconv:
        bx_mid = self.op(x, wb2, bias=None, stride=self.b2.stride, padding=self.b2.padding, dilation=self.b2.dilation,
                         groups=self.b2.groups) * c
    else:
        bx_mid = self.down_op(x, wb2) * c
    if self.rank_dropout and self.training:
        drop_a = (torch.rand(self.lora_dim, device=ax_mid.device) > self.rank_dropout).to(ax_mid.dtype)
        drop_b = (torch.rand(self.lora_dim, device=bx_mid.device) > self.rank_dropout).to(bx_mid.dtype)
        if self.rank_dropout_scale:
            drop_a /= drop_a.mean()
            drop_b /= drop_b.mean()
        dims = len(x.shape)
        if dims >= 4:
            drop_a = drop_a.view(1, -1, *([1] * (dims - 2)))
            drop_b = drop_b.view(1, -1, *([1] * (dims - 2)))
        else:
            drop_a = drop_a.view(*[1] * (dims - 1), -1)
            drop_b = drop_b.view(*[1] * (dims - 1), -1)
        ax_mid = ax_mid * drop_a
        bx_mid = bx_mid * drop_b
    ax = self.up_op(ax_mid, wa1)
    if self.tucker:
        wbm = self._orthogonalize(self.bm.weight).to(x.device, dtype=x.dtype)
        bx_mid = self.op(bx_mid, wbm, bias=None, stride=self.bm.stride, padding=self.bm.padding,
                         dilation=self.bm.dilation, groups=self.bm.groups)
    bx = self.up_op(bx_mid, wb1)
    ax = self.drop(ax)
    if not diff:
        return self.org_forward(x + ax) + self.drop(bx)
    out = self.org_forward(ax)
    bias = self.get_org_bias_for_compute(out.device)
    if bias is not None:
        out = out - _bias_view(self, bias.to(out.dtype), out)
    return out + self.drop(bx)


def _bias_view(self, bias: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """The base bias broadcast against out (channels on axis 1 for convs, last axis for linears)."""
    return bias.view(-1, *([1] * (out.dim() - 2))) if self.module_type.startswith("conv") else bias


def _ia3_cls():
    from lycoris.modules.ia3 import IA3Module

    return IA3Module


def _ia3_bypass_forward(self, x, scale=1, diff=False):
    """ia3.py:115-122 without scaling the base bias: stock multiplies org_forward(x) = W x + b by (1 + w), so the bias
    is scaled too (and bypass_forward_diff = w (W x + b)); with train_on_input the diff kept the whole bias."""
    bias = self.get_org_bias_for_compute(x.device)
    if bias is None:
        return _original(_ia3_cls(), "_bypass_forward")(self, x, scale, diff)
    w = self.weight * scale
    if self.train_input:
        out = self.org_forward(x * (w + int(not diff)))
        return out - _bias_view(self, bias.to(out.dtype), out) if diff else out
    out = self.org_forward(x)
    d = (out - _bias_view(self, bias.to(out.dtype), out)) * w
    return d if diff else out + d


def _ia3_make_module_from_state_dict(cls, lora_name, orig_module, weight, on_input=None):
    # ia3.py:84-92 takes (weight) but create_network_from_weights passes every weight_list entry (weight, on_input):
    # TypeError. on_input picks the scaled axis.
    train_on_input = bool(int(on_input.reshape(-1)[0])) if isinstance(on_input, torch.Tensor) else bool(on_input)
    module = cls(lora_name, orig_module, 1, train_on_input=train_on_input)
    module.weight.data.copy_(weight.reshape(module.weight.shape))
    return module


# ======================================================================================================================
# full (FullModule: trains W = W0 + dW in place of the base weight; W0 kept on the CPU in _org_weight)
# ======================================================================================================================


def _full_cls():
    from lycoris.modules.full import FullModule

    return FullModule


def _full_for_input(t: Optional[torch.Tensor], x: torch.Tensor) -> Optional[torch.Tensor]:
    """t in the dtype the op computes in: the autocast dtype (the cast linear/conv would do, same values; done here
    because autocast keeps its casts of fp32 leaf parameters until the outermost autocast region exits, i.e. a 16-bit
    copy of every full weight for the whole forward), or the input dtype where autocast is off (Anima adaLN)."""
    if t is None or not x.is_floating_point():
        return t
    dev = x.device.type
    dt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else x.dtype
    return t.to(dt) if t.dtype != dt else t


def _full_org(self, device, dtype) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """W0 (and the base bias) on `device` in `dtype`: one host-to-device copy each (full.py:184+190 made two)."""
    from lycoris.modules.base import transfer_ramtensor_to_device

    w = transfer_ramtensor_to_device(self._org_weight[0], device)
    b = None
    if self.org_bias is not None:
        b = transfer_ramtensor_to_device(self.org_bias[0], device)
    return w.to(dtype), (None if b is None else b.to(dtype))


def _full_apply_to(self, **kwargs):
    _original(_full_cls(), "apply_to")(self, **kwargs)
    # The trained tensor holds W0 + dW and was created as zeros_like(W0) (full.py:83): bf16 on a bf16 base. An AdamW step
    # smaller than half a bf16 ulp of |W| is rounded away (lr 2e-5 on Anima 2.9B: 38% of the weights at |step| = lr,
    # 72% at lr/5, 91% at 1e-6). fp32 master weights, as every other algo's factors are; the forward casts them to the
    # compute dtype (autocast, or _full_for_input).
    with torch.no_grad():
        for p in (self.weight, self.bias):
            if p is not None and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()


def _full_make_module_from_state_dict(cls, lora_name, orig_module, diff, diff_b):
    # full.py:89-101 copies dW into zeros_like(W0) (bf16 on a bf16 base) before apply_to adds W0: dW rounded twice
    module = cls(lora_name, orig_module, 1)
    with torch.no_grad():
        for p in (module.weight, module.bias):
            if p is not None and p.dtype in (torch.float16, torch.bfloat16):
                p.data = p.data.float()
        module.weight.copy_(diff)
        if diff_b is not None:
            if orig_module.bias is not None:
                module.bias.copy_(diff_b)
            else:
                module.bias = nn.Parameter(diff_b.float() if diff_b.dtype in (torch.float16, torch.bfloat16) else diff_b)
    module.is_diff = True
    return module


def _full_diff(self, device) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """(dW, db, W0, b0) on device in the trained tensor's dtype."""
    dev = device if device is not None else self.weight.device
    w = self.weight.to(dev)
    b = None if self.bias is None else self.bias.to(dev)
    org_w, org_b = _full_org(self, dev, w.dtype)
    if self.is_diff:  # before apply_to / make_module_from_state_dict: the tensors are the differences
        return w, b, org_w, org_b
    db = None
    if b is not None:
        db = b - org_b if org_b is not None else b  # no base bias: the bias tensor is the difference (full.py:96-100)
    return w - org_w, db, org_w, org_b


def _full_get_diff_weight(self, multiplier=1, shape=None, device=None):
    # full.py:216-250 multiplies by `multiplier` twice (and crashes on layers with a bias: diff_b is None at :232).
    dw, db, _, _ = _full_diff(self, device)
    dw = dw * multiplier
    if db is not None:
        db = db * multiplier
    if shape is not None:
        dw = dw.view(shape)
    return dw, db


def _full_drop(self, dw: torch.Tensor, db: Optional[torch.Tensor]):
    """rank_dropout / dropout on dW as LyCORIS defines them for the rebuild path (base.py: (W + rank_drop(dW)) x,
    lokr.py:616-623). full.py:194-198 crashes on any rank_dropout (bool of a tensor) and ignores dropout."""
    if self.rank_dropout:
        drop = (torch.rand(dw.size(0), device=dw.device) > self.rank_dropout).to(dw.dtype)
        if self.rank_dropout_scale:
            drop /= drop.mean().clamp_min(torch.finfo(dw.dtype).eps)
        dw = dw * drop.view(-1, *[1] * (dw.dim() - 1))
        if db is not None:
            db = db * drop
    if self.dropout:
        dw = self.drop(dw)
    return dw, db


def _full_make_weight(self, scale=1, device=None):
    drop = self.training and bool(self.rank_dropout or self.dropout)
    if not (drop or scale != 1 or self.is_diff):  # the trained tensors themselves (full.py:208-214)
        w, b = self.weight, self.bias
        if device is not None:
            w = w.to(device)
            b = None if b is None else b.to(device)
        return w, b
    if scale == 0 and not self.is_diff:  # DOP / iLECO / ADDifT teacher pass: exactly the base layer
        org_w, org_b = _full_org(self, device if device is not None else self.weight.device, self.weight.dtype)
        return org_w, org_b
    dw, db, org_w, org_b = _full_diff(self, device)
    if scale != 1:
        dw = dw * scale
        db = None if db is None else db * scale
    if drop:
        dw, db = _full_drop(self, dw, db)
    w = org_w + dw
    if db is None:
        b = org_b
    else:
        b = db if org_b is None else org_b + db
    return w, b


def _full_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    if self.module_dropout and self.training and torch.rand(1) < self.module_dropout:
        # the base layer: org_forward cannot run, apply_to removed its weight (full.py:139)
        w, b = _full_org(self, x.device, self._org_weight[0].dtype)
    else:
        w, b = self.make_weight(self.multiplier, x.device)
    kw = self.kw_dict | {"weight": _full_for_input(w, x), "bias": _full_for_input(b, x)}
    return self.op(x, **kw)


def _full_load_weight_prehook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
    # full.py:178-184 adds the file's dW to the current weight: --network_weights followed by --resume (or any second
    # load) counts the first file twice, and the sum runs in the file's dtype (fp16 files: dW rounded to W's ulp).
    # Here: W = W0 + dW in fp32 (W0 is _org_weight, never trained).
    if self.is_diff:
        return _original(_full_cls(), "load_weight_prehook")(self, state_dict, prefix, local_metadata, strict,
                                                             missing_keys, unexpected_keys, error_msgs)
    key = f"{prefix}diff"
    if key not in state_dict:
        return  # partial file (strict=False): "weight" is reported missing instead of a KeyError
    dw = state_dict.pop(key)
    base = self._org_weight[0]
    dt = torch.promote_types(torch.promote_types(dw.dtype, base.dtype), torch.float32)
    state_dict[f"{prefix}weight"] = base.to(device=dw.device, dtype=dt) + dw.to(dt)
    bkey = f"{prefix}diff_b"
    if bkey in state_dict and self.bias is not None:
        db = state_dict.pop(bkey)
        bdt = torch.promote_types(db.dtype, torch.float32)
        b0 = self.org_bias[0].to(device=db.device, dtype=bdt) if self.org_bias is not None else 0.0
        state_dict[f"{prefix}bias"] = b0 + db.to(bdt)


def _full_param_count(network) -> int:
    return sum(p.numel() for m in _all_loras(network) if type(m).__name__ == "FullModule" for p in m.parameters())


# ======================================================================================================================
# apply_module_patches
# ======================================================================================================================


def apply_module_patches() -> None:
    """Monkeypatch LyCORIS classes/functions in this process (idempotent). Call before the network is created."""
    from lycoris.modules import gora_utils, ralora_utils
    from lycoris.modules.abba import AbbaModule
    from lycoris.modules.boft import ButterflyOFTModule
    from lycoris.modules.diag_oft import DiagOFTModule
    from lycoris.modules.dylora import DyLoraModule
    from lycoris.modules.full import FullModule
    from lycoris.modules.locon import GoRAModule, LoConModule, RaLoRAModule
    from lycoris.modules.ortholora import OrthoLoRAModule
    from lycoris.modules.loha import LohaModule
    from lycoris.modules.lora2 import LoRA2Module
    from lycoris.modules.tlora import TLoraModule
    from lycoris.modules.tsm import TSMModule

    new = []

    def add(owner, name, func, reason, kind="function"):
        if _install(owner, name, func, reason, kind):
            new.append(_PATCH_LOG[-1])

    add(LohaModule, "get_diff_weight", _loha_get_diff_weight, "alpha/dim applied once (loha.py:349-350 applied it twice)")
    add(LohaModule, "make_module_from_state_dict", _loha_make_module_from_state_dict,
        "wd_on_output from the dora_scale shape (loha.py:256-264)", "classmethod")
    add(LohaModule, "bypass_forward_diff", _loha_bypass_forward_diff,
        "bypass delta in the input dtype (bf16, autocast off), as (x KA^T) KB^T when cheaper")
    add(LohaModule, "forward", _loha_forward,
        "rebuild forward: x W0^T + (x KA^T) KB^T where cheaper, else the stock ops without fp32 copies / host sync")

    add(AbbaModule, "initialize_weights", _abba_initialize_weights, "SVD init rank clamped to min(out, in) (abba.py:216)")
    add(AbbaModule, "get_merged_weight", _abba_get_merged_weight, "multiplier applied once (abba.py:371)")
    add(AbbaModule, "bypass_forward_diff", _abba_bypass_forward_diff,
        "Khatri-Rao factors rebuilt per call in the input dtype (abba.py:386-388 cached them), scales on B1")
    add(AbbaModule, "make_module_from_state_dict", _abba_make_module_from_state_dict,
        "wd_on_output from the dora_scale shape (abba.py:230-238)", "classmethod")
    add(AbbaModule, "forward", _abba_forward,
        "rebuild forward: x W0^T + (x KA^T) KB^T where cheaper, else the stock ops without fp32 copies")

    add(LoConModule, "make_module_from_state_dict", _locon_make_module_from_state_dict,
        "accepts all 8 weight_list entries + wd_on_output (locon.py:491-493)", "classmethod")
    add(RaLoRAModule, "make_module_from_state_dict", _ralora_make_module_from_state_dict,
        "accepts all weight_list entries + wd_on_output (locon.py:2162)", "classmethod")
    add(RaLoRAModule, "dynamic_init", _ralora_dynamic_init, "block weights on the module's device/dtype (locon.py:1773-1796)")
    add(RaLoRAModule, "load_weight_prehook", _ralora_load_weight_prehook,
        "same-shape block weights load in place (locon.py:2045: new Parameters, stale optimizer)")
    add(RaLoRAModule, "load_weight_hook", _ralora_load_weight_hook, "block layers: unused dense keys not reported missing")

    add(LoRA2Module, "custom_state_dict", _lora2_custom_state_dict, "full-rank tensors saved (lora2.py:372-373 cut them)")
    add(LoRA2Module, "make_module_from_state_dict", _lora2_make_module_from_state_dict,
        "wd_on_output from the dora_scale shape (lora2.py:381-398)", "classmethod")
    add(LoRA2Module, "get_total_rank_reg_loss", _lora2_total_rank_reg_loss,
        "one stacked op, no CUDA scalar from Python per step (lora2.py:177-190)", "classmethod")
    for cls in (GoRAModule, RaLoRAModule, LoRA2Module):
        add(cls, "forward", _lf_forward, "factored forward: no dense delta / merged weight, DoRA norm from the factors"
            + (", no nu.item() per call" if cls is LoRA2Module else ""))

    add(gora_utils, "gora_precompute_gradients", _gora_precompute_gradients,
        "grad enabled, base .grad freed (gora_utils.py:58/:785)")
    add(gora_utils, "allocate_ranks", _gora_allocate_ranks, "no second '/ iters' after the averaging (gora_utils.py:277)")
    add(gora_utils, "gora_dynamic_init", _gora_dynamic_init, "init not divided by N twice (gora_utils.py:549 + :791)")
    add(ralora_utils, "ralora_precompute_gradients", _ralora_precompute_gradients,
        "forward_fn(batch) or (model, batch), grad enabled, base requires_grad/grad reset (ralora_utils.py:516/:542/:551)")
    add(ralora_utils, "get_allocated_rank", _ralora_get_allocated_rank, "no second '/ iters' (ralora_utils.py:324)")
    add(ralora_utils, "compute_n_split_allocations", _ralora_compute_n_split_allocations,
        "no second '/ iters' (ralora_utils.py:428)")

    add(DiagOFTModule, "_forward_rebuild_core", _make_oft_rebuild_core(_diag_oft_cls),
        "rebuilt weight in the input dtype (diag_oft.py:302)")
    add(ButterflyOFTModule, "_forward_rebuild_core", _make_oft_rebuild_core(_boft_cls),
        "rebuilt weight in the input dtype (boft.py:358)")
    for cls, rotate in ((DiagOFTModule, _diag_oft_rotate), (ButterflyOFTModule, _boft_rotate)):
        add(cls, "get_r", _oft_get_r, "Cayley transform without host syncs (inv_ex, constraint by where)")
        add(cls, "forward", _make_oft_forward(_diag_oft_cls if cls is DiagOFTModule else _boft_cls, rotate),
            "Linear: base GEMM + blockwise rotation of the output instead of a rebuilt W' (no dW' GEMM)")

    add(TLoraModule, "forward", _tlora_fast_forward,
        "one down/up matmul pair, no dense delta (tlora.py:672-694); delta in the input dtype (tlora.py:691-693)")
    add(TLoraModule, "bypass_forward_diff", _tlora_bypass_forward_diff, "lambda in the input dtype (tlora.py:636-637)")

    add(TSMModule, "forward", _tsm_forward,
        "per-sample experts; stage 2 routes through the real input (tsm.py:538-545); base GEMM with the concatenated"
        " active experts' delta as its C input and a hand-written backward on the concatenated factors (_LowRankLinear,"
        " or _TSMLinear for the fp32 low-rank part; tsm_expert_fn=True: the experimental _TSMExpertLinear;"
        " tsm_fused=False: org_forward + separate delta)")
    add(TSMModule, "_compute_gates", _tsm_compute_gates, "gate pooling over all leading axes (tsm.py:362-371), no sync")
    add(TSMModule, "_get_timestep", _tsm_get_timestep,
        "block compile: no integer timestep inside a compiled graph (hoisted expert choice / embedding row)")

    add(LoConModule, "forward", _locon_forward,
        "plain LoRA/LoCon Linear rebuild as two thin matmuls (DoRA keeps the stock forward; GoRA/RaLoRA/LoRA2: _lf_forward)")
    _LOCON_STOCK["forward"] = _original(LoConModule, "forward")
    add(LoConModule, "_init_svd_segment", _locon_init_svd_segment,
        "PiSSA init factors / sqrt(scale * scalar), kept in fp32 (locon.py:404-407)")
    add(LoConModule, "load_weight_prehook", _locon_load_weight_prehook, "PiSSA initial factors load from a state")
    add(OrthoLoRAModule, "forward", _ortho_forward, "Linear rebuild as two thin matmuls (ortholora.py:273-299)")
    add(OrthoLoRAModule, "get_diff_weight", _ortho_get_diff_weight, "delta in the module dtype (ortholora.py:217: bf16)")
    add(OrthoLoRAModule, "_compute_effective_bases", _ortho_effective_bases,
        "Cayley solve without a host sync per call (linalg.solve's error check, ortholora.py:194)")
    add(DyLoraModule, "forward", _dylora_forward,
        "dtype fix, one rank per forward and its checkpoint recompute, all ranks when not training, thin matmuls")
    add(DyLoraModule, "get_diff_weight", _dylora_get_diff_weight, "all blocks unless a rank is given (dylora.py:118-121)")
    add(DyLoraModule, "load_weight_prehook", _dylora_load_weight_prehook,
        "lora_up/lora_down split into up_list/down_list (dylora.py:87-101)")

    from lycoris.modules.glora import GLoRAModule
    from lycoris.modules.ia3 import IA3Module
    from lycoris.modules.lokr import LokrModule

    add(LokrModule, "forward", _lokr_forward,
        "DoRA at multiplier != 1 interpolates W0 -> W1 (lokr.py:851: multiplier 0 kept the whole dW); m=1 untouched")
    _LOKR_STOCK["forward"] = _original(LokrModule, "forward")  # lycoris_fast's forward when installed first, else stock
    add(LokrModule, "get_merged_weight", _lokr_get_merged_weight, "DoRA merge at multiplier != 1 as the forward (lokr.py:651)")
    add(GLoRAModule, "forward", _glora_forward,
        "rebuild linear: same forward, backward without the full weight gradient (one tokens x out x in matmul less)")
    add(GLoRAModule, "_bypass_forward", _glora_bypass_forward,
        "alpha/rank and scalar applied once, no bias in the diff (glora.py:298-354)")
    add(IA3Module, "_bypass_forward", _ia3_bypass_forward, "base bias not scaled (ia3.py:115-122)")
    add(IA3Module, "make_module_from_state_dict", _ia3_make_module_from_state_dict,
        "accepts (weight, on_input) (ia3.py:84: TypeError in create_network_from_weights)", "classmethod")

    add(FullModule, "apply_to", _full_apply_to, "fp32 master weights (full.py:83 made them bf16 on a bf16 base)")
    add(FullModule, "get_diff_weight", _full_get_diff_weight, "multiplier applied once, bias handled (full.py:232/:244+250)")
    add(FullModule, "make_weight", _full_make_weight, "multiplier once, one W0 copy, rank_dropout/dropout on dW (full.py:189-214)")
    add(FullModule, "forward", _full_forward, "module_dropout uses W0; weight cast to the compute dtype here (adaLN: autocast off)")
    add(FullModule, "load_weight_prehook", _full_load_weight_prehook, "W = W0 + dW, not current W + dW (full.py:178-184)")
    add(FullModule, "make_module_from_state_dict", _full_make_module_from_state_dict,
        "dW copied into an fp32 tensor (full.py:96: bf16 on a bf16 base)", "classmethod")

    from lycoris.wrapper import network_module_dict

    from library import tsm_lokr

    if _install_mapping(network_module_dict, tsm_lokr.ALGO, tsm_lokr.get_module_class(),
                        "TSM routing with LoKr(+DoRA) experts (library/tsm_lokr.py)"):
        new.append(_PATCH_LOG[-1])

    if new:
        logger.info(f"LyCORIS fixed paths: patched {len(new)} LyCORIS targets")
        for line in new:
            logger.info(f"  {line}")


# ======================================================================================================================
# network kwargs
# ======================================================================================================================

_INT_KEYS = {
    "gora": ("gora_min_rank", "gora_max_rank", "gora_ref_rank"),
    "ralora": ("ralora_min_rank", "ralora_max_rank", "ralora_ref_rank", "ralora_n_max"),
    "lora2": ("lora2_nu_target",),
}


def _to_int(key: str, value) -> Optional[int]:
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none")):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"network_args {key}={value!r}: expected an integer / 정수가 필요합니다") from None


def prepare_network_kwargs(algo: str, net_kwargs: dict, logger=logger) -> dict:
    """Typed, validated copy of the LyCORIS network kwargs. Numbers LyCORIS passes on uncast become int; booleans and
    everything else stay as given (kohya.py parses them; train_network.py reads some as strings). Raises ValueError for
    combinations LyCORIS would silently ignore or crash on."""
    algo = (algo or "lora").lower()
    kw = dict(net_kwargs)
    if algo not in FIX_ALGOS:
        return kw

    for key in _INT_KEYS.get(algo, ()):
        if key in kw:
            v = _to_int(key, kw[key])
            if v is None:
                kw.pop(key)
            else:
                if v < 1:
                    raise ValueError(f"network_args {key}={v}: must be >= 1 / 1 이상이어야 합니다")
                kw[key] = v

    for prefix in ("gora", "ralora"):
        lo, hi = kw.get(f"{prefix}_min_rank"), kw.get(f"{prefix}_max_rank")
        if algo == prefix and isinstance(lo, int) and isinstance(hi, int) and lo > hi:
            raise ValueError(f"network_args {prefix}_min_rank={lo} > {prefix}_max_rank={hi}")

    if _str_bool(kw.get("dora_wd", False)) and algo in DORA_UNSUPPORTED:
        raise ValueError(
            f"algo={algo} does not support DoRA (dora_wd=True): LyCORIS would log 'DoRA is enabled' and train without it."
            f" Remove dora_wd or use one of {sorted(DORA_SUPPORTED)}."
            f" / algo={algo} 는 DoRA(dora_wd=True)를 지원하지 않습니다(LyCORIS 는 켜졌다고 로그만 남기고 무시합니다)."
            f" dora_wd 를 지우거나 {sorted(DORA_SUPPORTED)} 중 하나를 쓰세요."
        )

    if algo == "diag-oft" and _str_bool(kw.get("rescaled", False)):
        logger.warning(
            "diag-oft rescaled=True: Forge's OFT adapter ignores the rescale vector (oft.py:70-71); the file will not"
            " reproduce training in Forge. boft applies it."
        )
    if algo in ("diag-oft", "boft"):
        logger.info(
            f"{algo}: use strength 1.0 in Forge (its OFT/BOFT adapters apply the strength twice, oft.py:95+101,"
            " boft.py:96+109)"
        )
    if algo == "tlora":
        ignored = [k for k in ("sig_type", "use_data_init") if k in kw]
        if ignored:
            logger.warning(f"tlora: {ignored} never reach TLoraModule (kohya.py does not pass them); defaults are used")
        logger.warning(
            "tlora: Forge has no T-LoRA adapter; the saved file is a plain LoRA at the full mask (all ranks, no timestep"
            " masking)"
        )
    if algo == "tsm":
        logger.warning(
            "tsm: Forge has no TSM adapter; files are saved in LyCORIS' native TSM format"
        )
        global _TSM_FUSED, _TSM_LOWRANK_FP32
        fused = kw.pop("tsm_fused", None)  # not a TSMModule argument
        _TSM_FUSED = True if fused in (None, "") else _str_bool(fused)
        if not _TSM_FUSED:
            logger.warning("tsm: tsm_fused=False - org_forward + a separate expert delta per call (the slower split path)")
        global _TSM_EXPERT_FN
        global _TSM_MIXED
        mixed = kw.pop("tsm_mixed", None)  # not a TSMModule argument
        _TSM_MIXED = True if mixed in (None, "") else _str_bool(mixed)
        if not _TSM_MIXED:
            logger.warning("tsm: tsm_mixed=False - a mixed-timestep chunk runs one fused call per expert group (the slower"
                           " group loop)")
        efn = kw.pop("tsm_expert_fn", None)  # not a TSMModule argument
        _TSM_EXPERT_FN = False if efn in (None, "") else _str_bool(efn)
        if _TSM_EXPERT_FN:
            logger.warning("tsm: tsm_expert_fn=True - experimental _TSMExpertLinear (GPU A/B 10-25: +1.3 GiB at the peak,"
                           " not faster); the default is the factor path")
        lr32 = kw.pop("tsm_lowrank_fp32", None)  # not a TSMModule argument
        _TSM_LOWRANK_FP32 = None if lr32 in (None, "") else _str_bool(lr32)
        if _TSM_FUSED:
            logger.info("tsm: expert projections / gradients in fp32 (autocast off): "
                        + {None: "stage 2 only (default)", True: "on (tsm_lowrank_fp32=True, ~+11% step time)",
                           False: "off (tsm_lowrank_fp32=False)"}[_TSM_LOWRANK_FP32])
        if int(float(kw.get("tsm_stage", 1) or 1)) == 2:
            logger.info("tsm stage 2: start from the stage-1 file with --network_weights (not --resume)")
            gate = float(kw.get("tsm_gate_init", 0.0) or 0.0)
            if gate:
                logger.info(f"tsm stage 2: tsm_gate_init={gate:g} - the router bias starts at {gate:g} when the loaded"
                            " router is all zero (1.0 = the stage-1 model)")
        elif kw.get("tsm_gate_init") not in (None, "", "0", "0.0"):
            logger.warning("tsm: tsm_gate_init only applies to tsm_stage=2; ignored")
    if algo == "tsm-lokr":
        from library import tsm_lokr

        cfg = tsm_lokr.parse_config(kw)
        tsm_lokr.set_active_config(cfg)
        kw["tsm_n_scales"] = list(cfg.n_scales)  # kohya.py parses tsm_n_scales only for algo=tsm (kohya.py:147-154)
        logger.info(
            f"tsm-lokr: n_scales={list(cfg.n_scales)} stage={cfg.stage} factors core={cfg.core_factor}"
            f" context={cfg.context_factor} full_matrix={cfg.full_matrix} dora={cfg.dora}"
            f" offload_states={cfg.offload_states} resident_cores={cfg.resident_cores}/{cfg.n_core}"
            f" grouped_accumulation={cfg.grouped_accumulation}"
        )
        if cfg.stage == 2:
            logger.info("tsm-lokr stage 2: start from the stage-1 file with --network_weights (not --resume)")
    if algo == "ortholora" and _str_bool(kw.get("dora_wd", False)) and not _str_bool(kw.get("wd_on_output", True)):
        raise ValueError(
            "ortholora normalises DoRA on the output axis only (ortholora.py:137): wd_on_output=False would be ignored."
            " Set wd_on_output=True / ortholora 의 DoRA 는 출력 축만 지원합니다(wd_on_output=False 는 무시됨)."
            " wd_on_output=True 로 설정하세요."
        )
    if algo == "dylora":
        logger.info(
            "dylora: rank k trains with scale alpha * block_size / k (LyCORIS divides alpha by the block count); saved"
            " files store alpha * block_size so Forge applies the trained strength"
        )
    if algo == "lora2" and "lora2_lambda_e" in kw:
        logger.warning("lora2: lora2_lambda_e is stored but LyCORIS implements no entropy loss for it (lora2.py:143)")
    if algo == "abba" and _str_bool(kw.get("bypass_mode", False)):
        logger.info("abba bypass_mode: fixed (factors rebuilt every call; stock LyCORIS with --lycoris_stock caches them)")
    if algo == "ia3":
        logger.warning(
            "ia3: Forge has no IA3 adapter; files are saved in LyCORIS' native format (a Forge-loadable copy is a full"
            " '.diff' per layer: W * scale, the size of the adapted base weights)"
        )
    return kw


# ======================================================================================================================
# after_network_created: save/load/requires_grad
# ======================================================================================================================


def _parse_network_args(args) -> dict:
    out = {}
    for item in getattr(args, "network_args", None) or []:
        if isinstance(item, str) and "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _network_cfg(network) -> dict:
    return network.__dict__.setdefault("_lycoris_algo_fixes_cfg", {})


def _tsm_apply_gate_init(network, gate: float, logger=logger) -> int:
    """algo=tsm stage 2 with network_args tsm_gate_init (user 2026-10-24; same option as algo=tsm-lokr). A stage-1 file
    has an all-zero router (tsm.py:227-230), so stage 2 starts with gate 0: the context expert is off at the first step
    (LyCORIS' default, kept when tsm_gate_init is 0). A non-zero value is written to router_fc.bias, so gate = value at
    every t and input (1.0 = exactly the stage-1 model). Only over an all-zero router - never over a trained one, so a
    --resume (load_state after this) or a stage-2 file passed to --network_weights keeps its router."""
    if not gate:
        return 0
    done = 0
    with torch.no_grad():
        for m in _all_loras(network):
            if type(m).__name__ != "TSMModule" or int(getattr(m, "tsm_stage", 1)) != 2:
                continue
            if bool(torch.any(m.router_fc.weight != 0)) or bool(torch.any(m.router_fc.bias != 0)) \
                    or bool(torch.any(m.timestep_embed.weight != 0)):
                continue
            m.router_fc.bias.fill_(gate)
            done += 1
    if done:
        logger.info(f"tsm stage 2: router bias set to tsm_gate_init={gate:g} in {done} modules (stage-1 router was zero)")
    return done


def after_network_created(network, args, logger=logger) -> None:
    """Patch this network instance: Forge-loadable save_weights, shape-aware load_weights, TSM stage kept after
    requires_grad_(True). Idempotent."""
    cfg = _network_cfg(network)
    if cfg.get("installed"):
        return
    net_args = _parse_network_args(args)
    algo = (net_args.get("algo", "lora") or "lora").lower()
    cfg.update({"installed": True, "algo": algo})
    if BLOCK_COMPILE:
        n = rebind_forwards(network)
        logger.info(f"block compile: {n} adapted module forwards re-bound through closures (traceable by dynamo)")
    if algo == "tsm" and int(float(net_args.get("tsm_stage", 1) or 1)) == 2:
        cfg["tsm_gate_init"] = float(net_args.get("tsm_gate_init", 0.0) or 0.0)
    dim = getattr(args, "network_dim", None)
    if algo == "tlora" or any(type(m).__name__ == "TLoraModule" for m in _all_loras(network)):
        max_rank = int(dim) if dim is not None else max((int(m.lora_dim) for m in _all_loras(network)), default=4)
        min_rank = net_args.get("tlora_min_rank")
        min_rank = int(math.ceil(max_rank * 0.5)) if min_rank in (None, "", "None") else int(float(min_rank))
        cfg["tlora"] = {
            "max_rank": max_rank,
            "min_rank": max(0, min(min_rank, max_rank)),
            "alpha": float(net_args.get("tlora_mask_alpha", 1.0)),
        }

    orig_save = network.save_weights
    orig_requires_grad = network.requires_grad_

    def save_weights(file, dtype, metadata):
        return _save_weights_for_forge(network, orig_save, file, dtype, metadata, logger)

    def load_weights(file):
        info = _load_weights_adapted(network, file, logger)
        _tsm_apply_gate_init(network, cfg.get("tsm_gate_init", 0.0), logger)
        return info

    def requires_grad_(requires_grad: bool = True):
        result = orig_requires_grad(requires_grad)
        if requires_grad:
            loras_ = _all_loras(network)
            for lora in loras_:
                if hasattr(lora, "tsm_stage") and hasattr(lora, "set_stage"):
                    lora.set_stage(lora.tsm_stage)  # kohya.py:1475/:1650 unfroze the stage-2 experts
            _freeze_unused_ralora_dense(loras_)
        return result

    network.save_weights = save_weights
    network.load_weights = load_weights
    network.requires_grad_ = requires_grad_
    _use_native_states(network, logger)

    if hasattr(network, "prepare_optimizer_params"):
        orig_prepare = network.prepare_optimizer_params

        def prepare_optimizer_params(*a, **k):
            # Leave frozen tensors (TSM stage-2 experts, RaLoRA block layers' unused dense factors) out of the
            # optimizer: kohya.py:1520 takes every parameter. A no-op when nothing is frozen.
            res = orig_prepare(*a, **k)
            groups, descs = (res if isinstance(res, tuple) else (res, None))
            aligned = descs is not None and len(descs) == len(groups)
            kept_groups, kept_descs = [], []
            for i, g in enumerate(groups):
                params = [p for p in g["params"] if p.requires_grad]
                if params or not aligned:
                    kept_groups.append({**g, "params": params})
                    if aligned:
                        kept_descs.append(descs[i])
            if not isinstance(res, tuple):
                return kept_groups
            return kept_groups, (kept_descs if aligned else descs)

        network.prepare_optimizer_params = prepare_optimizer_params

    kinds = Counter(type(m).__name__ for m in _all_loras(network))
    if algo == "ia3" and not kinds:
        # preset=ia3 (config.py:284-293) targets SD/CLIP names (to_k, to_v, ff.net.2): nothing on a DiT, and the
        # optimizer later fails with an empty parameter list
        raise ValueError(
            f"algo=ia3 adapted no module (preset={net_args.get('preset', 'full')}). preset=ia3 only names SD/SDXL layers"
            " (to_k, to_v, ff.net.2); on Anima use preset=anima or anima-exp"
            " / algo=ia3 가 적용된 레이어가 없습니다. preset=ia3 은 SD/SDXL 레이어 이름만 지정합니다(Anima 는 anima/anima-exp)"
        )
    logger.info(f"LyCORIS fixed paths: network {dict(kinds)}; saves are written in a Forge-loadable format")
    if kinds.get("TSMModule"):
        logger.warning("TSM: no Forge adapter exists; the saved file keeps LyCORIS' TSM format")
    if kinds.get("TLoraModule"):
        logger.warning("T-LoRA: the saved file is a plain LoRA at the full mask (Forge applies every rank at every step)")
    if kinds.get("IA3Module"):
        logger.warning("IA3: no Forge adapter exists; the saved file keeps LyCORIS' IA3 format")

    n_full = _full_param_count(network)
    if n_full:
        gib = n_full * 4 / 2**30
        logger.info(
            f"full: {n_full / 1e9:.3f}B trained parameters in fp32: weights {gib:.1f} GiB + grads {gib:.1f} GiB +"
            f" AdamW states {2 * gib:.1f} GiB on the GPU (8-bit states: {n_full * 2 / 2**30:.1f} GiB); W0 stays on the CPU"
            f" ({sum(m._org_weight[0].numel() * m._org_weight[0].element_size() for m in _all_loras(network) if type(m).__name__ == 'FullModule') / 2**30:.1f} GiB)"
        )
        if getattr(args, "full_bf16", False) or getattr(args, "full_fp16", False):
            logger.warning(
                "full with full_bf16/full_fp16: the trained weights W0 + dW are cast to 16 bits, AdamW steps under half"
                " an ulp of |W| are rounded away (lr 2e-5 on Anima 2.9B: 38-91% of the weights never move)"
                " / full_bf16 에서는 W0+dW 가 16비트라 작은 갱신이 반올림으로 사라집니다"
            )

    if kinds.get("TSMLokrModule"):
        _after_tsm_lokr_created(network, args, cfg, logger)


def _tsm_lokr_modules(network) -> List[nn.Module]:
    return [m for m in _all_loras(network) if type(m).__name__ == "TSMLokrModule"]


def _after_tsm_lokr_created(network, args, cfg: dict, logger=logger) -> None:
    from library import lycoris_fast, tsm_lokr

    te = [m.lora_name for m in getattr(network, "text_encoder_loras", None) or [] if type(m).__name__ == "TSMLokrModule"]
    if te:
        raise ValueError(
            f"tsm-lokr: {len(te)} text-encoder modules (e.g. {te[0]}) get no timestep; set network_train_unet_only"
            " / tsm-lokr: 텍스트 인코더 모듈에는 타임스텝이 없습니다. network_train_unet_only 를 켜세요"
        )
    mods = _tsm_lokr_modules(network)
    # one compile switch for the LoKr rebuild and this one (--anima_no_lokr_compile / --moss_no_lokr_compile)
    tsm_lokr.enable_compiled_rebuild(bool(lycoris_fast._REBUILD.enabled))
    mp = str(getattr(args, "mixed_precision", "") or "").lower()
    cfg.setdefault("tsm_timestep_dtype_fallback", {"bf16": "bf16", "fp16": "fp16"}.get(mp, "fp32"))
    cfg["tsm_lokr"] = True  # set_timestep records the trainer's timestep dtype for the saved file
    summ = tsm_lokr.size_summary(mods)
    c = mods[0].cfg
    logger.info(
        f"tsm-lokr: {summ['modules']} modules, {c.n_core} core + {sum(c.n_scales[1:])} context experts;"
        f" core expert {summ['core_expert_params'] / 1e6:.2f} M params ({summ['core_state_gib_each']:.2f} GiB AdamW state"
        f" each), all weights {summ['total_params'] / 1e6:.1f} M ({summ['weights_fp32_gib']:.2f} GiB fp32);"
        f" compiled rebuild {'on' if tsm_lokr._TSM_REBUILD.enabled else 'off'}"
    )
    logger.warning(
        f"tsm-lokr: each saved LoRA file is ~{summ['file_fp16_gib']:.2f} GiB (fp16) and each --save_state"
        f" ~{summ['save_state_gib']:.1f} GiB (weights + AdamW); check save_every_n_epochs / save_last_n_epochs(_state)"
        f" and the free space of output_dir / tsm-lokr 저장 파일 하나 약 {summ['file_fp16_gib']:.2f} GiB, 상태 저장 하나"
        f" 약 {summ['save_state_gib']:.1f} GiB 입니다"
    )


def _use_native_states(network, log) -> None:
    """The accelerate state (network.state_dict) keeps what training needs: OrthoLoRA's S_p/S_q/bases/lambda
    (native_save; the stock state was the distilled LoRA, which --resume could not load) and PiSSA's trained and initial
    factors (pissa_convert=False; the stock state was the converted rank-2r LoRA). save_weights converts both for Forge."""
    n_ortho = n_pissa = 0
    for m in _all_loras(network):
        kind = type(m).__name__
        if kind == "OrthoLoRAModule" and not m.native_save:
            m.native_save = True
            n_ortho += 1
        elif kind == "LoConModule" and getattr(m, "is_pissa", False) and getattr(m, "pissa_convert", False):
            m.pissa_convert = False
            n_pissa += 1
    if n_ortho or n_pissa:
        log.info(f"LyCORIS fixed paths: native training state for {n_ortho} OrthoLoRA / {n_pissa} PiSSA modules"
                 " (saved weights are converted to LoRA)")


# ---------------------------------------------------------------------------------------------------------------- save


def _lora_shapes(module, up2d: torch.Tensor, down2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(out, r) / (r, in*k) -> the LoCon layout of this module (conv: up (out, r, 1, ...), down (r, in, *k))."""
    r = up2d.shape[1]
    if len(module.shape) > 2:
        up = up2d.reshape(module.shape[0], r, *([1] * (len(module.shape) - 2)))
        down = down2d.reshape(r, *module.shape[1:])
        return up, down
    return up2d, down2d


def _convert_abba(m) -> Tuple[dict, str]:
    out = _out_dim(m)
    r1, r2 = m.r1, m.r2
    c = float(m.scale) * _scalar_value(m)
    f = math.sqrt(abs(c))
    up1 = m.lora_up1.weight.detach().double().cpu().reshape(out, r1) * f
    up2 = m.lora_up2.weight.detach().double().cpu().reshape(out, r2) * (f if c >= 0 else -f)
    down1 = m.lora_down1.weight.detach().double().cpu().reshape(r1, -1)
    down2 = m.lora_down2.weight.detach().double().cpu().reshape(r2, -1)
    r = max(r1, r2)
    if r1 < r:  # odd dim: pad pair 1 with zero ranks so LyCORIS' LoHa loader (one lora_dim) reads it back
        up1 = torch.cat([up1, up1.new_zeros(out, r - r1)], 1)
        down1 = torch.cat([down1, down1.new_zeros(r - r1, down1.shape[1])], 0)
    sd = {
        "hada_w1_a": up1, "hada_w1_b": down1, "hada_w2_a": up2, "hada_w2_b": down2,
        "alpha": torch.tensor(float(r)),  # Forge: alpha / w1b.shape[0] = 1 (loha.py:68-71); the scale is in w*_a
    }
    if m.wd:
        sd["dora_scale"] = m.dora_scale.detach().double().cpu()
    return sd, "loha"


def _convert_lora2(m) -> Tuple[dict, str]:
    from lycoris.modules.lora2_utils import compute_effective_rank, compute_lambda_diag

    d = compute_effective_rank(m.lora2_nu, m.lora2_quantile, m.lora_dim)
    lam = compute_lambda_diag(m.lora2_nu.detach().double().cpu(), d)
    out = _out_dim(m)
    up = m.lora_up.weight.detach().double().cpu().reshape(out, -1)[:, :d]
    down = m.lora_down.weight.detach().double().cpu().reshape(m.lora_down.weight.shape[0], -1)[:d]
    up = up * (float(m.scale) * _scalar_value(m))  # lora2.py:258-271 multiplies by self.scalar
    down = down * lam.unsqueeze(1)  # Lambda into down: down is large (std ~5), Lambda small -> fp16-safe
    up, down = _lora_shapes(m, up, down)
    sd = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(d))}
    if m.wd:
        sd["dora_scale"] = m.dora_scale.detach().double().cpu()
    return sd, "lora"


def _convert_ralora(m, native: dict) -> Tuple[dict, str]:
    out = _out_dim(m)
    sc = float(m.scale)
    if m.n_split > 1:
        n, r = m.n_split, m.mini_lora_rank
        mo, mi = m.mini_out_features, m.mini_in_features
        with torch.no_grad():
            As = [m._maybe_nora_down(a).detach().double().cpu() for a in m._mini_lora_A]
            Bs = [m._orthogonalize(b).detach().double().cpu() for b in m._mini_lora_B]
        k = int(math.prod(m.shape[2:])) if len(m.shape) > 2 else 1
        up = torch.zeros(out, n * r, dtype=torch.float64)
        down = torch.zeros(n * r, m.shape[1], k, dtype=torch.float64)
        for i in range(n):
            up[i * mo:(i + 1) * mo, i * r:(i + 1) * r] = Bs[i].reshape(mo, r)
            down[i * r:(i + 1) * r, i * mi:(i + 1) * mi, :] = As[i].reshape(r, mi, k)
        up = up * (sc * _scalar_value(m))
        up, down = _lora_shapes(m, up, down.reshape(n * r, -1))
        sd = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(n * r))}
    else:
        # LoCon export (scalars, NoRA, PiSSA conversion already applied); fold the trained scale (RaLoRA-Pro saved
        # alpha = scaling_alpha while it trains with alpha / avg_rank, locon.py:1811-1823).
        up = native["lora_up.weight"].detach().double().cpu() * sc
        down = native["lora_down.weight"].detach().double().cpu()
        sd = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(down.shape[0]))}
        if "lora_mid.weight" in native:
            sd["lora_mid.weight"] = native["lora_mid.weight"].detach().double().cpu()
    if m.wd:
        sd["dora_scale"] = m.dora_scale.detach().double().cpu()
    return sd, "lora"


def _balanced_factors(U: torch.Tensor, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """U @ D (out x k, k x in) refactored as up @ down with orthogonal columns/rows scaled by sqrt(singular values), so
    each factor is about as large as the product (fp16-safe: the difference of two large products is not stored)."""
    Qu, Ru = torch.linalg.qr(U)
    Qd, Rd = torch.linalg.qr(D.T)
    Uc, S, Vh = torch.linalg.svd(Ru @ Rd.T)
    keep = int((S > S.max() * 1e-12).sum()) if S.numel() and float(S.max()) > 0 else 0
    keep = max(keep, 1)
    s = S[:keep].sqrt()
    up = Qu @ Uc[:, :keep] * s.unsqueeze(0)
    down = (s.unsqueeze(1) * Vh[:keep]) @ Qd.T
    return up, down


def _convert_tlora(m) -> Tuple[dict, str]:
    out = _out_dim(m)
    r = m.lora_dim
    sv = _scalar_value(m)
    lam = m.lambda_layer.detach().double().cpu().reshape(1, r) * sv
    lamb = m.base_lambda.detach().double().cpu().reshape(1, r) * sv
    P = m.p_layer.weight.detach().double().cpu().reshape(out, r)
    Q = m.q_layer.weight.detach().double().cpu().reshape(r, -1)
    Pb = m.base_p.detach().double().cpu().reshape(out, r)
    Qb = m.base_q.detach().double().cpu().reshape(r, -1)
    # delta = scale * (P diag(lam) Q - Pb diag(lamb) Qb) at the full mask (tlora.py:482-532)
    U = torch.cat([P * lam, -Pb * lamb], 1) * float(m.scale)
    D = torch.cat([Q, Qb], 0)
    up, down = _balanced_factors(U, D)
    k = up.shape[1]
    if len(m.shape) > 2:
        kernel = tuple(m.shape[2:])
        up = up.reshape(out, k, *([1] * len(kernel)))
        if all(s == 1 for s in kernel):
            down = down.reshape(k, m.shape[1], *kernel)
        else:  # 1x1 delta placed at the padding offset, as get_merged_weight does (tlora.py:546-556)
            full = torch.zeros(k, m.shape[1], *kernel, dtype=down.dtype)
            pad = m.kw_dict.get("padding", tuple(s // 2 for s in kernel))
            full[(slice(None), slice(None)) + tuple(slice(p, p + 1) for p in pad)] = down.reshape(
                k, m.shape[1], *([1] * len(kernel))
            )
            down = full
    return {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(k))}, "lora"


def _convert_dylora(m, native: dict) -> Tuple[dict, str]:
    # Training applies alpha / (b + 1) for b + 1 blocks (dylora.py:112), i.e. alpha * block_size / rank, but the file
    # kept alpha, which Forge divides by the rank (weight_adapter/lora.py:108-111): block_size times too weak. Storing
    # alpha * block_size gives the trained strength, also for a file cut to fewer blocks. Conv lora_up gets LoCon's
    # (out, r, 1, ...) layout (dylora.py:93-95 saved it 2-D, which create_network_from_weights could not copy).
    up, down = native["lora_up.weight"].detach(), native["lora_down.weight"].detach()
    up, down = _lora_shapes(m, up.reshape(up.shape[0], -1), down.reshape(down.shape[0], -1))
    sd = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(m.alpha) * m.block_size)}
    return sd, "lora"


def _convert_ortholora(m) -> Tuple[dict, str]:
    # The stock distilled save (ortholora.py:323-353) drops dora_scale (Forge applied a plain LoRA to a DoRA-trained
    # module) and rounds the factors to bf16. Here: the training forward's factors (_ortho_up), balanced per rank by
    # sqrt(|lam * scalar * scale|) in fp64 (fp16-safe), scale folded (alpha = rank), dora_scale kept.
    with torch.no_grad():
        q, u = _ortho_up(m, m.lambda_layer.device)
        q, u = q.detach().double().cpu(), u.detach().double().cpu()
        lam = m.lambda_layer.detach().double().cpu().reshape(-1) * (_scalar_value(m) * float(m.scale))
        d = lam.abs().sqrt()
        d = torch.where(d > 0, d, torch.ones_like(d))
        up = u / d.unsqueeze(0)
        down = q * d.unsqueeze(1)
    sd = {"lora_up.weight": up, "lora_down.weight": down, "alpha": torch.tensor(float(up.shape[1]))}
    if m.wd:
        sd["dora_scale"] = m.dora_scale.detach().double().cpu()
    return sd, "lora"


def _convert_pissa(m) -> Tuple[dict, str]:
    # PiSSA trains W_res + c * up @ down with W_res = W - c * up0 @ down0 (the flag's init, see _locon_init_svd_segment),
    # so the change to the original W is c * up @ down - c * up0 @ down0 = [c*up | c*up0] @ [down | -down0]. The stock
    # conversion (locon.py:922-941) kept alpha for the doubled rank (Forge: half strength), never applied the scale to
    # the initial factors, and left a pissa_converted key Forge does not read.
    out = _out_dim(m)
    c = float(m.scale)
    up = m._apply_row_scalar(m.lora_up.weight).detach().double().cpu().reshape(out, -1) * c
    down = m._apply_col_scalar(m._nora_export_weight(m.lora_down.weight)).detach().double().cpu()
    down = down.reshape(down.shape[0], -1)
    up0 = m.pissa_A_init.detach().double().cpu().reshape(out, -1) * c
    down0 = m.pissa_B_init.detach().double().cpu().reshape(down.shape[0], -1)
    up2, down2 = _lora_shapes(m, torch.cat([up, up0], 1), torch.cat([down, -down0], 0))
    sd = {"lora_up.weight": up2, "lora_down.weight": down2, "alpha": torch.tensor(float(up2.shape[1]))}
    if m.wd:
        sd["dora_scale"] = m.dora_scale.detach().double().cpu()
    return sd, "lora"


def _convert_glora(m, native: dict) -> Tuple[dict, str]:
    # Forge's GLoRA adapter scales by alpha / a2 rows (weight_adapter/glora.py:62-77). rs_lora trains with alpha /
    # sqrt(rank) but saves alpha (glora.py:126-134): save scale * rank instead (also what make_module_from_state_dict
    # reads back without rs_lora).
    out = dict(native)
    if getattr(m, "rs_lora", False):
        out["alpha"] = torch.tensor(float(m.scale) * int(m.lora_dim))
    return out, "glora"


def convert_state_dict_for_forge(network, state_dict: dict) -> Tuple["OrderedDict[str, torch.Tensor]", Counter]:
    """LyCORIS state_dict -> the Forge-loadable state_dict (see the module docstring). Returns (new dict, formats)."""
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    formats: Counter = Counter()
    claimed = set()
    for m in _all_loras(network):
        prefix = f"{m.lora_name}."
        keys = [k for k in state_dict if k.startswith(prefix)]
        if not keys:
            continue
        claimed.update(keys)
        native = {k[len(prefix):]: state_dict[k] for k in keys}
        kind = type(m).__name__
        conv, fmt = None, None
        with torch.no_grad():
            if kind == "AbbaModule" and m.r1 > 0 and m.r2 > 0:
                conv, fmt = _convert_abba(m)
            elif kind == "LoRA2Module":
                conv, fmt = _convert_lora2(m)
            elif kind == "RaLoRAModule":
                conv, fmt = _convert_ralora(m, native)
            elif kind == "TLoraModule":
                conv, fmt = _convert_tlora(m)
            elif kind == "DyLoraModule":
                conv, fmt = _convert_dylora(m, native)
            elif kind == "OrthoLoRAModule":
                conv, fmt = _convert_ortholora(m)
            elif kind == "LoConModule" and getattr(m, "is_pissa", False) and getattr(m, "pissa_A_init", None) is not None:
                conv, fmt = _convert_pissa(m)
            elif kind == "GLoRAModule":
                conv, fmt = _convert_glora(m, native)
        if conv is None:
            conv = native
            fmt = {"GoRAModule": "lora", "LohaModule": "loha", "DiagOFTModule": "oft", "ButterflyOFTModule": "boft",
                   "TSMModule": "tsm-native", "LoConModule": "lora", "LokrModule": "lokr",
                   "IA3Module": "ia3-native", "FullModule": "diff",
                   "TSMLokrModule": "tsm-lokr-native"}.get(kind, "native")
            if kind in ("DiagOFTModule", "ButterflyOFTModule"):
                # the identity buffer (diag_oft.py:82, boft.py:102) is not in their weight_list and Forge's OFT/BOFT
                # adapters never read it: left in, Forge reports one unmatched key per module
                conv = {k: v for k, v in native.items() if k != "I"}
        formats[fmt] += 1
        for suffix, value in conv.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype == torch.float64:
                value = value.float()
            out[prefix + suffix] = value.detach().contiguous() if isinstance(value, torch.Tensor) else value
    for k, v in state_dict.items():
        if k not in claimed:
            out[k] = v
    return out, formats


def _tsm_timestep_dtype(network) -> str:
    cfg = _network_cfg(network)
    return cfg.get("tsm_timestep_dtype") or cfg.get("tsm_timestep_dtype_fallback") or "fp32"


def _save_weights_for_forge(network, orig_save, file, dtype, metadata, log) -> Any:
    mgr = network.__dict__.get("_tsm_offload")
    if mgr is not None:
        mgr.flush()  # pending write-backs into the host copies of offloaded experts
    sd_native = network.state_dict()
    converted, formats = convert_state_dict_for_forge(network, sd_native)
    md = dict(metadata) if metadata else {}
    md[METADATA_FLAG_KEY] = "1"
    md[METADATA_FORMAT_KEY] = "+".join(sorted(formats)) if formats else "native"
    md[METADATA_ALGO_KEY] = _network_cfg(network).get("algo", "")
    if METADATA_ALLOC_KEY not in md:
        alloc = gora_ralora_allocation_state(network)
        if alloc is not None:
            md[METADATA_ALLOC_KEY] = json.dumps(alloc)
    tl = _tsm_lokr_modules(network)
    if tl:
        from library import tsm_lokr

        md[tsm_lokr.METADATA_EXPERT_KEY] = "lokr"
        md[tsm_lokr.METADATA_CONFIG_KEY] = tsm_lokr.metadata_json(tl[0], _tsm_timestep_dtype(network))
    network.__dict__["state_dict"] = lambda *a, **k: converted  # read once by LycorisNetworkKohya.save_weights
    try:
        return orig_save(file, dtype, md)
    finally:
        network.__dict__.pop("state_dict", None)


# ---------------------------------------------------------------------------------------------------------------- load


def _resize_dense_lora(m, rank: int) -> None:
    """lora_up/lora_down of a LoCon-style module to `rank` (fresh init: down Kaiming, up 0)."""
    if getattr(m, "tucker", False):
        raise ValueError(f"{m.lora_name}: resizing a Tucker (lora_mid) module is not supported")
    ref = m.lora_down.weight
    dev, dt = ref.device, ref.dtype
    out_dim = _out_dim(m)
    in_dim = int(m.shape[1])
    if len(m.shape) > 2:
        down = type(m.lora_down)(in_dim, rank, m.lora_down.kernel_size, m.lora_down.stride, m.lora_down.padding,
                                 bias=False)
        up = type(m.lora_up)(rank, out_dim, 1, bias=False)
    else:
        down = nn.Linear(in_dim, rank, bias=False)
        up = nn.Linear(rank, out_dim, bias=False)
    nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
    nn.init.zeros_(up.weight)
    m.lora_down = down.to(device=dev, dtype=dt)
    m.lora_up = up.to(device=dev, dtype=dt)
    m.lora_dim = rank


def _set_alpha(m, value: float) -> None:
    alpha = getattr(m, "alpha", None)
    if isinstance(alpha, torch.Tensor):
        m.alpha = torch.tensor(float(value), dtype=alpha.dtype, device=alpha.device)
    else:
        m.register_buffer("alpha", torch.tensor(float(value)))


def adapt_state_dict_to_network(network, sd: dict, metadata: Optional[dict] = None, log=logger) -> dict:
    """Make a weights file trained/saved for this network loadable into it: Forge-converted ABBA (LoHa keys) back to
    ABBA, GoRA/RaLoRA per-layer ranks (modules resized, scale = alpha/rank as Forge reads it), old truncated LoRA² files
    padded. Raises for conversions that cannot be undone (LoRA² Lambda, T-LoRA)."""
    sd = dict(sd)
    metadata = metadata or {}
    touched = Counter()
    for m in _all_loras(network):
        p = f"{m.lora_name}."
        kind = type(m).__name__
        if kind == "AbbaModule" and f"{p}hada_w1_a" in sd and f"{p}lora_up1.weight" not in sd:
            w1a, w1b = sd.pop(f"{p}hada_w1_a").double(), sd.pop(f"{p}hada_w1_b").double()
            w2a, w2b = sd.pop(f"{p}hada_w2_a").double(), sd.pop(f"{p}hada_w2_b").double()
            alpha_f = float(sd.pop(f"{p}alpha")) if f"{p}alpha" in sd else float(w1b.shape[0])
            ratio = (alpha_f / w1b.shape[0]) / float(m.scale)
            f = math.sqrt(abs(ratio))
            g = f if ratio >= 0 else -f
            r1, r2 = m.r1, m.r2
            sd[f"{p}lora_up1.weight"] = (w1a[:, :r1] * f).reshape(m.lora_up1.weight.shape)
            sd[f"{p}lora_down1.weight"] = w1b[:r1].reshape(m.lora_down1.weight.shape)
            sd[f"{p}lora_up2.weight"] = (w2a[:, :r2] * g).reshape(m.lora_up2.weight.shape)
            sd[f"{p}lora_down2.weight"] = w2b[:r2].reshape(m.lora_down2.weight.shape)
            sd[f"{p}alpha"] = m.alpha.detach().clone()
            touched["abba<-loha"] += 1
        elif kind == "LoRA2Module" and f"{p}lora_up.weight" in sd:
            if f"{p}lora2_nu" not in sd:
                raise ValueError(
                    f"{m.lora_name}: this file is a Forge export of LoRA² (Lambda baked into lora_down) and cannot"
                    " continue LoRA² training; use --resume with the saved state instead"
                    " / Forge 용으로 변환된 LoRA² 파일은 이어서 학습할 수 없습니다(--resume 상태를 쓰세요)"
                )
            d = sd[f"{p}lora_down.weight"].shape[0]
            if d < m.lora_dim:  # file from stock LyCORIS (cut to rank D)
                up = m.lora_up.weight.detach().cpu().float().clone()
                down = m.lora_down.weight.detach().cpu().float().clone()
                up.zero_()
                up[:, :d] = sd[f"{p}lora_up.weight"].float()
                down[:d] = sd[f"{p}lora_down.weight"].float()
                sd[f"{p}lora_up.weight"], sd[f"{p}lora_down.weight"] = up, down
                touched["lora2 padded"] += 1
        elif kind == "OrthoLoRAModule" and f"{p}S_p" not in sd and f"{p}lora_up.weight" in sd:
            raise ValueError(
                f"{m.lora_name}: this file is a distilled OrthoLoRA (a plain LoRA) and cannot continue OrthoLoRA"
                " training; use --resume with the saved state instead / 일반 LoRA 로 변환된 OrthoLoRA 파일은 이어서"
                " 학습할 수 없습니다(--resume 상태를 쓰세요)"
            )
        elif (kind == "LoConModule" and getattr(m, "is_pissa", False) and f"{p}lora_down.weight" in sd
              and f"{p}pissa_A_init" not in sd and sd[f"{p}lora_down.weight"].shape[0] == 2 * m.lora_dim):
            raise ValueError(
                f"{m.lora_name}: this file is a converted PiSSA LoRA (rank 2r against the original base) and cannot"
                " continue PiSSA training; use --resume with the saved state instead / 변환된 PiSSA 파일은 이어서"
                " 학습할 수 없습니다(--resume 상태를 쓰세요)"
            )
        elif kind == "DyLoraModule" and f"{p}alpha" in sd and metadata.get(METADATA_FLAG_KEY) == "1":
            sd[f"{p}alpha"] = sd[f"{p}alpha"].float() / m.block_size  # saved as alpha * block_size (_convert_dylora)
            touched["dylora alpha"] += 1
        elif kind == "TLoraModule" and f"{p}q_layer.weight" not in sd and f"{p}lora_up.weight" in sd:
            raise ValueError(
                f"{m.lora_name}: this file is a Forge export of T-LoRA (a plain LoRA) and cannot continue T-LoRA"
                " training; use --resume with the saved state instead / Forge 용 T-LoRA 파일은 이어서 학습할 수 없습니다"
            )
        elif kind in ("GoRAModule", "RaLoRAModule") and f"{p}lora_down.weight" in sd and f"{p}lora_down_block0.weight" not in sd:
            if kind == "RaLoRAModule" and m.n_split > 1:
                # the file is the block-diagonal export (plain LoRA): training continues as a dense LoRA of rank n*r
                m.dynamic_init(avg_rank=m.lora_dim, rank=int(sd[f"{p}lora_down.weight"].shape[0]), n_split=1)
                touched["RaLoRA blocks -> dense"] += 1
            rows = int(sd[f"{p}lora_down.weight"].shape[0])
            if rows != int(m.lora_down.weight.shape[0]):
                _resize_dense_lora(m, rows)
                touched[f"{kind} resized"] += 1
            alpha_f = float(sd[f"{p}alpha"]) if f"{p}alpha" in sd else float(rows)
            m.scale = alpha_f / rows  # what Forge applies (weight_adapter/lora.py:107-110)
            _set_alpha(m, alpha_f)
            touched[f"{kind} loaded"] += 1
    if any(k.startswith(("GoRA", "RaLoRA")) for k in touched):
        network._gora_needs_init = False
        network._ralora_needs_init = False
    if touched:
        log.info(f"LyCORIS fixed paths: adapted the weights file to the network: {dict(touched)}")
    if touched.get("RaLoRA blocks -> dense"):
        log.warning(
            f"RaLoRA: {touched['RaLoRA blocks -> dense']} block-diagonal layers (n_split > 1) were loaded from a Forge export"
            " and continue as dense LoRA layers (same output, no block constraint); use --resume to keep the blocks"
            " / RaLoRA 블록 레이어가 Forge 용 파일에서 일반 LoRA 로 읽혀 블록 제약 없이 이어서 학습됩니다(블록 유지는 --resume)"
        )
    return sd


def _load_weights_adapted(network, file, log):
    metadata: dict = {}
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file, safe_open

        with safe_open(file, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")
        metadata = getattr(weights_sd, "_metadata", {}).get("", {}) if hasattr(weights_sd, "_metadata") else {}
    network._apply_nora_metadata(metadata)
    weights_sd = adapt_state_dict_to_network(network, weights_sd, metadata, log)
    network.weights_sd = weights_sd
    missing, unexpected = network.load_state_dict(weights_sd, strict=False)
    oft_eye = {f"{m.lora_name}.I" for m in _all_loras(network) if type(m).__name__ in ("DiagOFTModule", "ButterflyOFTModule")}
    missing = [k for k in missing if k not in oft_eye]  # identity buffer, not saved (convert_state_dict_for_forge)
    stage2 = [f"{m.lora_name}.experts." for m in _tsm_lokr_modules(network) if m.tsm_stage == 2]
    if stage2:
        lost = [k for k in missing if k.startswith(tuple(stage2))]
        if lost:
            raise ValueError(
                f"tsm-lokr stage 2: {len(lost)} expert tensors are missing from {file} (e.g. {lost[0]}); stage 2 trains"
                " only the router on top of the stage-1 experts / tsm-lokr 2단계: 1단계 전문가 가중치가 파일에 없습니다"
            )
    state = {}
    if missing:
        state["missing keys"] = missing
    if unexpected:
        state["unexpected keys"] = unexpected
    return state


# ======================================================================================================================
# timesteps (T-LoRA mask, TSM routing)
# ======================================================================================================================


_TSM_KINDS = ("TSMModule", "TSMLokrModule")


def set_timestep(network, timesteps, max_timestep: int = 1000) -> None:
    """Per-step timestep for T-LoRA's rank mask and TSM's expert routing.

    timesteps: as the trainer has them before Anima's "/ 1000" (anima_train_network.py:1164-1166): sigma * 1000, a
    float tensor in [0, max_timestep] (bf16 is fine), shape (B,) or scalar. T-LoRA gets one mask per sample (B > 1:
    tlora.py takes its bypass path) with max_rank = network_dim, min_rank = tlora_min_rank or ceil(dim/2) and
    tlora_mask_alpha, as train_network.py:1198-1220. TSM gets round(t) rescaled to tsm_num_timesteps, per sample.
    Keep it set until after accelerator.backward (gradient checkpointing re-runs the forward there)."""
    loras = _all_loras(network)
    if not loras:
        return
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(timesteps)
    t = timesteps.detach().float().reshape(-1)
    tl = [m for m in loras if type(m).__name__ == "TLoraModule"]
    tsm = [m for m in loras if type(m).__name__ in _TSM_KINDS]
    if tl:
        from lycoris.modules.tlora import compute_timestep_mask, compute_timestep_mask_batch, set_timestep_mask

        cfg = _network_cfg(network).get("tlora")
        if cfg is None:
            max_rank = max(int(m.lora_dim) for m in tl)
            cfg = {"max_rank": max_rank, "min_rank": int(math.ceil(max_rank * 0.5)), "alpha": 1.0}
        if t.numel() == 1:
            mask = compute_timestep_mask(int(t.item()), max_timestep, cfg["max_rank"], cfg["min_rank"], cfg["alpha"])
        else:
            mask = compute_timestep_mask_batch(t, max_timestep, cfg["max_rank"], cfg["min_rank"], cfg["alpha"])
        groups = {int(getattr(m, "mask_group_id", 0)) for m in tl}
        for g in groups:
            set_timestep_mask(mask.to(t.device), g)
    if tsm:
        from lycoris.modules.tsm import set_tsm_timestep

        T = int(tsm[0].tsm_num_timesteps)
        ints = [int(round(float(v) * T / float(max_timestep))) for v in t.cpu().tolist()]
        ints = [max(0, min(T, v)) for v in ints]
        groups = {int(getattr(m, "mask_group_id", 0)) for m in tsm}
        _TSM_PLANS.clear()
        for g in groups:
            set_tsm_timestep(ints[0], g)
            if len(ints) > 1:
                _TSM_SAMPLE_T[g] = ints
            else:
                _TSM_SAMPLE_T.pop(g, None)
        if BLOCK_COMPILE:
            _hoist_tsm_step(tsm, ints)
        cfg = _network_cfg(network)
        if cfg.get("tsm_lokr") and torch.is_grad_enabled() and "tsm_timestep_dtype" not in cfg:
            from library import tsm_lokr

            # the dtype the trainer's timesteps had (Anima: bf16, MOSS: fp32); Forge quantises sigma*1000 the same way
            cfg["tsm_timestep_dtype"] = tsm_lokr.dtype_name(timesteps.dtype)
        mgr = network.__dict__.get("_tsm_offload")
        if mgr is not None:
            mgr.prepare_micro_step(ints)


def clear_timestep(network) -> None:
    """Undo set_timestep (call after accelerator.backward, and after sampling)."""
    loras = _all_loras(network)
    tl_groups = {int(getattr(m, "mask_group_id", 0)) for m in loras if type(m).__name__ == "TLoraModule"}
    tsm_groups = {int(getattr(m, "mask_group_id", 0)) for m in loras if type(m).__name__ in _TSM_KINDS}
    if tl_groups:
        from lycoris.modules.tlora import clear_timestep_mask

        for g in tl_groups:
            clear_timestep_mask(g)
    if tsm_groups:
        from lycoris.modules.tsm import clear_tsm_timestep

        _TSM_PLANS.clear()
        for g in tsm_groups:
            clear_tsm_timestep(g)
            _TSM_SAMPLE_T.pop(g, None)


# ======================================================================================================================
# GoRA / RaLoRA allocation (resume)
# ======================================================================================================================


def gora_ralora_allocation_state(network) -> Optional[dict]:
    """Per-module rank / n_split / scale / alpha of GoRA and RaLoRA modules (JSON-able), or None if there are none."""
    modules = {}
    algo = None
    for m in _all_loras(network):
        kind = type(m).__name__
        if kind not in ("GoRAModule", "RaLoRAModule"):
            continue
        algo = "gora" if kind == "GoRAModule" else "ralora"
        n_split = int(getattr(m, "n_split", 1) or 1)
        rank = int(m.mini_lora_rank) if n_split > 1 else int(m.lora_down.weight.shape[0])
        alpha = getattr(m, "alpha", None)
        modules[m.lora_name] = {
            "rank": rank,
            "n_split": n_split,
            "scale": float(m.scale),
            "alpha": float(alpha) if isinstance(alpha, torch.Tensor) else None,
        }
    if not modules:
        return None
    return {"version": ALLOC_STATE_VERSION, "algo": algo, "modules": modules}


def restore_gora_ralora_allocation(network, state: dict) -> None:
    """Rebuild the shapes gora_ralora_allocation_state() recorded (before the optimizer is built and the state is
    loaded) and mark the pre-pass done. Weights are re-initialized: load the saved state/weights afterwards."""
    if not state or not state.get("modules"):
        return
    entries = state["modules"]
    done, missing = 0, []
    for m in _all_loras(network):
        kind = type(m).__name__
        if kind not in ("GoRAModule", "RaLoRAModule"):
            continue
        e = entries.get(m.lora_name)
        if e is None:
            missing.append(m.lora_name)
            continue
        rank, n_split = int(e["rank"]), int(e.get("n_split", 1) or 1)
        if kind == "RaLoRAModule":
            m.dynamic_init(avg_rank=rank, rank=rank, n_split=n_split)
            if n_split == 1 and int(m.lora_down.weight.shape[0]) != rank:
                _resize_dense_lora(m, rank)
        elif int(m.lora_down.weight.shape[0]) != rank:
            _resize_dense_lora(m, rank)
        m.scale = float(e["scale"])
        if e.get("alpha") is not None:
            _set_alpha(m, float(e["alpha"]))
        done += 1
    if missing:
        logger.warning(f"GoRA/RaLoRA allocation: {len(missing)} modules not in the saved state, e.g. {missing[:3]}")
    network._gora_needs_init = False
    network._ralora_needs_init = False
    logger.info(f"GoRA/RaLoRA allocation restored on {done} modules")


# ======================================================================================================================
# TSM-LoKr: trainer checks and the core-expert offload manager
# ======================================================================================================================

_GROUPED_REFUSED = (
    ("antithetic_timestep_sampling", "--antithetic_timestep_sampling"),
    ("stratified_timestep_sampling", "--stratified_timestep_sampling"),
    ("qmc_timestep_sampling", "--qmc_timestep_sampling"),
    ("adaptive_timestep_sampling", "--adaptive_timestep_sampling"),
    ("ileco", "--ileco"),
    ("addift", "--addift"),
)


def check_tsm_lokr_args(args) -> None:
    """Start-up checks that need the trainer args (raise, never adjust). Call after prepare_network_kwargs."""
    from library import tsm_lokr

    cfg = tsm_lokr.ACTIVE_CONFIG
    if cfg is None:
        return

    def refuse(en: str, ko: str):
        raise ValueError(f"tsm-lokr: {en} / tsm-lokr: {ko}")

    if getattr(args, "deepspeed", False):
        refuse("--deepspeed is not supported", "--deepspeed 는 지원하지 않습니다")
    if getattr(args, "use_ramtorch_network", False):
        refuse("--use_ramtorch_network would move the router into RamTorch; use --use_ramtorch (base weights) instead",
               "--use_ramtorch_network 은 쓸 수 없습니다(--use_ramtorch 는 됩니다)")
    if getattr(args, "scale_weight_norms", None):
        refuse("--scale_weight_norms has no single weight to rescale", "--scale_weight_norms 는 지원하지 않습니다")
    if getattr(args, "dim_from_weights", False):
        refuse("--dim_from_weights is not supported (use --network_weights)", "--dim_from_weights 대신 --network_weights 를 쓰세요")
    if getattr(args, "network_train_text_encoder_only", False):
        refuse("text-encoder training gets no timestep", "텍스트 인코더 학습에는 타임스텝이 없습니다")
    if cfg.stage == 2 and not getattr(args, "network_weights", None) and not getattr(args, "resume", None):
        refuse("stage 2 needs the stage-1 file (--network_weights)", "2단계는 1단계 파일(--network_weights)이 필요합니다")
    if cfg.grouped_accumulation:
        for attr, flag in _GROUPED_REFUSED:
            if getattr(args, attr, None):
                refuse(f"tsm_grouped_accumulation=True cannot be combined with {flag}",
                       f"tsm_grouped_accumulation=True 는 {flag} 와 함께 쓸 수 없습니다")


def _is_torch_adam(optimizer) -> bool:
    return isinstance(optimizer, (torch.optim.AdamW, torch.optim.Adam))


def attach_tsm_offload(network, optimizer, args, device, batch_size: int = 1, grouped_supported: bool = False,
                       logger=logger):
    """ExpertOffloadManager for a TSM-LoKr network (tsm_offload_states / tsm_resident_cores / tsm_grouped_accumulation),
    or None when none is in use. Call after accelerator.prepare and before the accelerate state hooks / --resume.
    `optimizer` is the torch optimizer (AcceleratedOptimizer.optimizer). Inconsistent settings raise."""
    from library import tsm_lokr
    from library.tsm_offload import ExpertOffloadManager

    mods = _tsm_lokr_modules(network)
    if not mods:
        return None
    cfg: tsm_lokr.TsmLokrConfig = mods[0].cfg
    n_core = cfg.n_core
    stage1 = cfg.stage == 1
    grouped = bool(cfg.grouped_accumulation)
    accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
    window = n_core if grouped else min(n_core, max(1, int(batch_size)) * accum)

    def refuse(en: str, ko: str):
        raise ValueError(f"tsm-lokr: {en} / tsm-lokr: {ko}")

    if grouped and not grouped_supported:
        refuse("tsm_grouped_accumulation=True is implemented for anima_train_network.py only",
               "tsm_grouped_accumulation=True 는 아직 Anima 학습기에서만 됩니다")
    need = 1 if grouped else window
    if cfg.resident_cores < need and stage1:
        refuse(f"tsm_resident_cores={cfg.resident_cores} is below the core experts one optimizer step can touch"
               f" (batch {batch_size} x accumulation {accum} = {need}); raise it or set tsm_grouped_accumulation=True",
               f"tsm_resident_cores={cfg.resident_cores} 가 한 옵티마이저 스텝이 쓰는 코어 전문가 수({need})보다 작습니다."
               " 값을 올리거나 tsm_grouped_accumulation=True 를 쓰세요")
    if cfg.resident_cores < min(n_core, max(1, int(batch_size))):
        refuse(f"tsm_resident_cores={cfg.resident_cores} is below the batch size {batch_size}",
               f"tsm_resident_cores={cfg.resident_cores} 가 배치 크기 {batch_size} 보다 작습니다")
    offload_states = bool(cfg.offload_states) and stage1
    if offload_states and not _is_torch_adam(optimizer):
        refuse(f"tsm_offload_states needs torch AdamW/Adam; set tsm_offload_states=False for {type(optimizer).__name__}",
               f"tsm_offload_states 는 torch AdamW/Adam 에서만 됩니다. {type(optimizer).__name__} 에서는"
               " tsm_offload_states=False 로 하세요")
    if offload_states and any(pg.get("amsgrad") for pg in optimizer.param_groups):
        refuse("tsm_offload_states moves exp_avg / exp_avg_sq only, not AMSGrad's max_exp_avg_sq; set amsgrad=False or"
               " tsm_offload_states=False",
               "tsm_offload_states 는 AMSGrad 의 max_exp_avg_sq 를 옮기지 않습니다. amsgrad=False 또는"
               " tsm_offload_states=False 로 하세요")
    # Swapped-out cores (tsm_resident_cores < n_core) are written back only when the step post-hook marked them stepped,
    # and grouped accumulation starts a new window in that hook: both need every update to go through optimizer.step().
    lru = cfg.resident_cores < n_core and stage1
    name = type(optimizer).__name__
    if (lru or grouped) and callable(getattr(optimizer, "step_func", None)):
        refuse(f"{name} updates through step_func, not optimizer.step(); tsm_resident_cores < {n_core} and"
               " tsm_grouped_accumulation follow optimizer.step()",
               f"{name} 는 optimizer.step() 대신 step_func 로 갱신합니다. tsm_resident_cores < {n_core} 와"
               " tsm_grouped_accumulation 은 optimizer.step() 을 따라갑니다")
    if lru and callable(getattr(optimizer, "train", None)) and callable(getattr(optimizer, "eval", None)):
        refuse(f"{name} (schedule-free train()/eval()) rewrites every parameter outside optimizer.step(), swapped-out"
               f" cores included; keep tsm_resident_cores={n_core}",
               f"{name}(schedule-free 의 train()/eval())는 optimizer.step() 밖에서 모든 파라미터를 바꿉니다."
               f" tsm_resident_cores={n_core} 로 두세요")
    if not offload_states and cfg.resident_cores >= n_core and not grouped:
        return None
    groups = [[p for m in mods for p in m.experts[0][i].parameters()] for i in range(n_core)]
    mgr = ExpertOffloadManager(
        groups, optimizer, device, core_index=cfg.core_index, offload_states=offload_states,
        resident_cores=cfg.resident_cores, grouped=grouped, core_index_t=cfg.core_index_from_timestep,
    )
    mgr.attach_network_hooks(network)
    network.__dict__["_tsm_offload"] = mgr
    logger.info(
        f"tsm-lokr offload: core AdamW states in {'pinned RAM' if offload_states else 'VRAM'},"
        f" core weights resident {cfg.resident_cores}/{n_core}, grouped accumulation {'on' if grouped else 'off'};"
        f" host copies {mgr.host_bytes() / 2 ** 30:.2f} GiB"
    )
    return mgr
