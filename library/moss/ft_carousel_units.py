# MOSS-Image full fine-tune, Carousel mode (docs/moss_ft_carousel_plan.md 3.1, 3.5, 3.7, 3.8): the pure parts.
#
#   discover_units(named_shapes, stem)    the rotating units in forward order: 44 (2 text refiner blocks, 42 joint
#                                         blocks), 45 with the stem as a unit (--moss_carousel_always rotate_stem); a
#                                         unit holds every trainable tensor of its block, the block's norms included
#   tensor_role(name)                     "unit" / "stem" / "head" (every MOSS parameter is one of the three)
#   always_names(names, policy)           the always-trainable tensors of --moss_carousel_always small / head / none /
#                                         rotate_stem
#   Selector(units, k, kind, seed)        select(i): the active unit ids of switch index i (pure: no RNG state)
#   grad_floor(names, num_layers)         (lowest joint block that needs gradients, whether the text path needs them)
#   apply_grad_floor(model, f, text)      the no-grad prefix below the floor (instance forward wrappers, models.py as is)
#   tensor_device_bytes / state_bytes     the per-tensor byte counts of the budget (plan 3.8)
#
# Nothing here touches a device or allocates a tensor except apply_grad_floor's wrappers (which only switch grad mode).

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

STEM_PREFIXES = ("video_embedder.", "text_embedder.proj.", "time_embedding.", "shared_block_modulation.")
HEAD_PREFIXES = ("final_norm_video.", "final_linear_video.", "video_head_modulation.")
REFINER_PREFIX = "text_embedder.blocks."
JOINT_PREFIX = "layers."
TEXT_PREFIX = "text_embedder."
# what feeds every joint block besides the stream: a trained one needs the activation gradient at every block
STREAM_FEED_PREFIXES = ("video_embedder.", "time_embedding.", "shared_block_modulation.")

ALWAYS_POLICIES = ("small", "head", "none", "rotate_stem")
STEM_UNIT = "stem"  # the stem's unit name with --moss_carousel_always rotate_stem (the head stays always trainable)
SELECTIONS = ("random", "round_robin", "window")
STATE_POLICIES = ("keep", "reset")
STATE_DTYPES = ("fp32", "bf16")
HOST_PINNED = ("none", "masters")
KINDS = ("adamw", "adamw8bit", "came")
BNB_MIN_8BIT_SIZE = 4096  # bitsandbytes Optimizer2State: smaller tensors keep fp32 states
BNB_BLOCK = 256  # blockwise absmax: one fp32 per 256 values (absmax1 and absmax2)


@dataclass(frozen=True)
class Unit:
    """One rotating unit: a refiner block (`text_embedder.blocks.j`) or a joint block (`layers.i`). index: the unit's
    position in forward order (refiners first); joint: the joint block index (None for a refiner); names: its trainable
    tensors in registration order; numel: their parameter count."""
    index: int
    name: str
    joint: Optional[int]
    names: Tuple[str, ...]
    numel: int


def block_of(name: str) -> Optional[str]:
    """`layers.7` / `text_embedder.blocks.1` for a tensor of that block, None for the stem and head tensors."""
    for prefix in (REFINER_PREFIX, JOINT_PREFIX):
        if name.startswith(prefix):
            rest = name[len(prefix):]
            number = rest.split(".", 1)[0]
            if number.isdigit():
                return prefix + number
    return None


def tensor_role(name: str) -> str:
    """"unit" (inside a refiner or joint block), "stem" (feeds the blocks) or "head" (after the blocks)."""
    if block_of(name) is not None:
        return "unit"
    if name.startswith(STEM_PREFIXES):
        return "stem"
    if name.startswith(HEAD_PREFIXES):
        return "head"
    raise ValueError(f"MOSS full FT: {name} is neither a block, stem nor head tensor of the MOSS DiT (another model?)")


def stem_rotates(policy: str) -> bool:
    """--moss_carousel_always rotate_stem: the stem tensors form a rotating unit of their own."""
    return policy == "rotate_stem"


def discover_units(named_shapes: Iterable[Tuple[str, Sequence[int]]], stem_unit: bool = False) -> List[Unit]:
    """The units of the trainable tensors [(name, shape)] in forward order: (stem_unit: the stem tensors as unit
    "stem",) the refiner blocks, then the joint blocks by index. A block whose tensors are all frozen has no unit."""
    members: Dict[str, List[Tuple[str, int]]] = {}
    for name, shape in named_shapes:
        block = block_of(name)
        if block is None:
            if tensor_role(name) == "stem" and stem_unit:  # tensor_role: a known stem / head tensor, else ValueError
                members.setdefault(STEM_UNIT, []).append((name, math.prod(tuple(shape))))
            continue
        members.setdefault(block, []).append((name, math.prod(tuple(shape))))

    def order(block: str):
        if block == STEM_UNIT:
            return (-1, 0)
        refiner = block.startswith(REFINER_PREFIX)
        return (0 if refiner else 1, int(block.rsplit(".", 1)[1]))

    units = []
    for index, block in enumerate(sorted(members, key=order)):
        joint = None if block == STEM_UNIT or block.startswith(REFINER_PREFIX) else int(block.rsplit(".", 1)[1])
        names = tuple(n for n, _ in members[block])
        units.append(Unit(index=index, name=block, joint=joint, names=names, numel=sum(c for _, c in members[block])))
    return units


def always_names(names: Iterable[str], policy: str) -> List[str]:
    """The always-trainable tensors (in the given order) of --moss_carousel_always: small = stem + head, head = the
    head only (rotate_stem too: its stem is a unit), none = nothing."""
    if policy not in ALWAYS_POLICIES:
        raise ValueError(f"MOSS full FT: the always policy must be one of {ALWAYS_POLICIES}, got {policy!r}")
    roles = {"small": ("stem", "head"), "head": ("head",), "none": (), "rotate_stem": ("head",)}[policy]
    return [n for n in names if block_of(n) is None and tensor_role(n) in roles]


# ---- selection (plan 3.7) -------------------------------------------------------------------------------------------
class Selector:
    """select(i) -> the sorted unit ids of switch index i (i = global_step // period). A pure function of (units, k,
    kind, seed): no RNG state, no draws from torch / numpy / python's global RNGs.

    random: a stream of cycles, each unit exactly once per cycle; cycle c = the units shuffled by
      random.Random(f"{seed}:{c}"); set i = the next k units of the stream. At a cycle boundary the set takes the
      r < k remaining units of the old cycle (the carry, r may be 0) and k - r of the new one. The new cycle's order is
      the shuffle with a stable lead: first k - r units that are neither carried nor in the previous set, then k units
      that are not carried; the rest follow in shuffle order. So no unit is drawn twice into a set, and (U >= 2k) no
      unit is active in two consecutive sets across a boundary.
    round_robin: M = ceil(U / k) sets per cycle, set i = {u : u mod M == i mod M} (spread over the depth).
    window: k contiguous units starting at (i * k) mod U, wrapping."""

    def __init__(self, num_units: int, k: int, kind: str = "random", seed: int = 0):
        if kind not in SELECTIONS:
            raise ValueError(f"MOSS full FT: the selection must be one of {SELECTIONS}, got {kind!r}")
        self.num_units, self.k, self.kind, self.seed = int(num_units), int(k), kind, int(seed)
        if self.num_units < 1 or not 1 <= self.k <= self.num_units:
            raise ValueError(f"MOSS full FT: {self.k} active units of {self.num_units}: need 1 <= k <= units")
        self._sets: List[Tuple[int, ...]] = []
        self._cycle, self._pos, self._order = -1, 0, []

    def _cycle_order(self, cycle: int, carry: Sequence[int], previous: Sequence[int]) -> List[int]:
        order = list(range(self.num_units))
        random.Random(f"{self.seed}:{cycle}").shuffle(order)
        carried, before = set(carry), set(previous)
        need = self.k - len(carried)
        first = [u for u in order if u not in carried and u not in before][:need]
        if len(first) < need:  # U < 2k: the previous set cannot be avoided (the carry always is: U >= k)
            first = [u for u in order if u not in carried][:need]
        taken = set(first)
        second = [u for u in order if u not in carried and u not in taken][:self.k]
        taken.update(second)
        return first + second + [u for u in order if u not in taken]

    def __call__(self, i: int) -> Tuple[int, ...]:
        i = int(i)
        if i < 0:
            raise ValueError(f"switch index must be >= 0, got {i}")
        if self.k == self.num_units:
            return tuple(range(self.num_units))
        if self.kind == "round_robin":
            m = -(-self.num_units // self.k)
            return tuple(u for u in range(self.num_units) if u % m == i % m)
        if self.kind == "window":
            start = (i * self.k) % self.num_units
            return tuple(sorted((start + t) % self.num_units for t in range(self.k)))
        while len(self._sets) <= i:
            chosen: List[int] = []
            while len(chosen) < self.k:
                if self._pos >= len(self._order):
                    self._cycle += 1
                    self._order = self._cycle_order(self._cycle, chosen, self._sets[-1] if self._sets else ())
                    self._pos = 0
                chosen.append(self._order[self._pos])
                self._pos += 1
            self._sets.append(tuple(sorted(chosen)))
        return self._sets[i]


# ---- backward truncation (plan 3.5) ---------------------------------------------------------------------------------
def grad_floor(names: Iterable[str], num_layers: int) -> Tuple[int, bool]:
    """(f, text_grad) for the tensors trainable right now: f = the lowest joint block whose output needs a gradient
    (0 when a stream feed - video embedder, time MLP, shared modulation - or the text path is trained; num_layers when
    no joint block and nothing below them is trained), text_grad = the text path (projection or a refiner) is trained."""
    names = list(names)
    text_grad = any(n.startswith(TEXT_PREFIX) for n in names)
    if text_grad or any(n.startswith(STREAM_FEED_PREFIXES) for n in names):
        return 0, text_grad
    joint = [int(n[len(JOINT_PREFIX):].split(".", 1)[0]) for n in names if n.startswith(JOINT_PREFIX)]
    return (min(joint) if joint else int(num_layers)), False


def _no_grad_forward(module):
    forward = type(module).forward

    def no_grad_forward(*args, **kwargs):
        with torch.no_grad():
            return forward(module, *args, **kwargs)

    no_grad_forward._carousel = "no_grad"
    return no_grad_forward


def _grad_input_forward(module):
    forward = type(module).forward

    def grad_input_forward(x, *args, **kwargs):
        # unsloth's checkpoint is an autograd.Function: its output carries grad only if a tensor INPUT does, and block
        # parameters are not inputs - without this the floor block's weight gradients would be lost silently
        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.detach().requires_grad_(True)
        return forward(module, x, *args, **kwargs)

    grad_input_forward._carousel = "grad_input"
    return grad_input_forward


def clear_grad_floor(model) -> None:
    """Remove the wrappers of apply_grad_floor (the class forwards run again)."""
    for module in [*model.layers, model.text_embedder]:
        if getattr(module.__dict__.get("forward"), "_carousel", None):
            del module.__dict__["forward"]


def apply_grad_floor(model, floor: int, text_grad: bool, unsloth: Optional[bool] = None) -> None:
    """Blocks below `floor` run under torch.no_grad() (nothing saved, no checkpoint node, no unsloth copy, no
    recompute); the text embedder too when the text path is not trained. Under unsloth offload the floor block's input
    is made a grad-requiring leaf (one block's input dgrad). unsloth None: read from the blocks."""
    clear_grad_floor(model)
    if unsloth is None:
        unsloth = any(getattr(block, "unsloth_offload_checkpointing", False) for block in model.layers)
    for index, layer in enumerate(model.layers):
        if index < floor:
            layer.__dict__["forward"] = _no_grad_forward(layer)
        elif index == floor and unsloth:
            layer.__dict__["forward"] = _grad_input_forward(layer)
    if not text_grad:
        model.text_embedder.__dict__["forward"] = _no_grad_forward(model.text_embedder)


# ---- bytes (plan 3.8) -----------------------------------------------------------------------------------------------
def factored_elements(shape: Sequence[int]) -> int:
    """CAME's statistics besides exp_avg (ft_budget._factored_elements: rows + cols of exp_avg_sq and exp_avg_res for
    >= 2-D, a full exp_avg_sq (+ exp_avg_res, an upper bound) for 1-D)."""
    shape = tuple(int(d) for d in shape)
    if len(shape) >= 2:
        return 2 * (math.prod(shape[:-1]) + math.prod(shape[:-2]) * shape[-1])
    return 2 * math.prod(shape)


def state_bytes(kind: str, shape: Sequence[int], state_dtype: str = "fp32") -> int:
    """The optimizer state of one tensor (what `keep` stores on the host): adamw m + v fp32; adamw8bit uint8 m + v and
    the fp32 absmax per 256 values (fp32 m + v below 4096 values); came exp_avg (fp32, bf16 with state_dtype bf16) +
    the factored statistics fp32."""
    n = math.prod(tuple(int(d) for d in shape))
    if kind == "adamw":
        return 8 * n
    if kind == "adamw8bit":
        if n < BNB_MIN_8BIT_SIZE:
            return 8 * n
        return 2 * n + 8 * (-(-n // BNB_BLOCK))
    if kind == "came":
        return (2 if state_dtype == "bf16" else 4) * n + 4 * factored_elements(shape)
    raise ValueError(f"unknown rotating-block optimizer kind {kind!r}")


def tensor_device_bytes(kind: str, shape: Sequence[int]) -> int:
    """Device bytes of an active tensor on top of its compute copy: fp32 master + fp32 gradient accumulator + the
    optimizer state (always fp32 on the device)."""
    return 8 * math.prod(tuple(int(d) for d in shape)) + state_bytes(kind, shape)


def group_device_bytes(kind: str, shapes: Iterable[Sequence[int]]) -> int:
    return sum(tensor_device_bytes(kind, shape) for shape in shapes)


def optimizer_temp_bytes(kind: str, shapes: Iterable[Sequence[int]]) -> int:
    """Step temporaries: CAME's core holds two fp32 tensors of the parameter (update, residual); fused AdamW and
    bitsandbytes step in place."""
    if kind != "came":
        return 0
    return 2 * 4 * max([math.prod(tuple(int(d) for d in s)) for s in shapes] + [0])


def unit_shapes(unit: Unit, shapes: Mapping[str, Sequence[int]]) -> List[Tuple[int, ...]]:
    return [tuple(shapes[n]) for n in unit.names]
