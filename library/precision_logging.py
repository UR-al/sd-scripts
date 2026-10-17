"""One-time dtype observations without copying tensors or changing training precision."""

from collections import defaultdict
from collections.abc import Mapping

import torch


def dtype_name(dtype):
    return {
        torch.float64: "FP64", torch.float32: "FP32", torch.float16: "FP16",
        torch.bfloat16: "BF16",
    }.get(dtype, str(dtype).removeprefix("torch."))


def tensor_summary(tensors):
    """Inspect metadata only; shared tensor objects are counted once."""
    counts = defaultdict(lambda: [0, 0])
    seen = set()
    for tensor in tensors:
        if tensor is None or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        count = counts[(dtype_name(tensor.dtype), str(tensor.device))]
        count[0] += 1
        count[1] += tensor.numel()
    return "; ".join(
        f"{dtype}@{device}: {count:,} tensors / {numel:,} elements"
        for (dtype, device), (count, numel) in sorted(counts.items())
    ) or "none observed"


def _raw_optimizer(optimizer):
    seen = set()
    while id(optimizer) not in seen:
        seen.add(id(optimizer))
        inner = getattr(optimizer, "optimizer", None)
        if inner is None:
            break
        optimizer = inner
    return optimizer


def _optimizer_parameters(optimizer):
    seen = set()
    for group in _raw_optimizer(optimizer).param_groups:
        for parameter in group["params"]:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield parameter


def _state_tensors(value, path="", *, parameter_ids=frozenset(), seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        if id(value) not in parameter_ids:
            yield path, value
    elif isinstance(value, Mapping):
        if id(value) in seen:
            return
        seen.add(id(value))
        for key, child in value.items():
            # optimi gradient_release keeps a parameter-group reference in state.
            if key == "group" and isinstance(child, Mapping) and "params" in child:
                continue
            yield from _state_tensors(child, f"{path}.{key}" if path else str(key),
                                      parameter_ids=parameter_ids, seen=seen)
    elif isinstance(value, (list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        for child in value:
            yield from _state_tensors(child, f"{path}[]", parameter_ids=parameter_ids, seen=seen)


def _state_optimizers(optimizer):
    """Schedule-free wrappers can own state in addition to their base optimizer."""
    current = _raw_optimizer(optimizer)
    prefix = ""
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield prefix, current
        current = getattr(current, "base_optimizer", None)
        prefix += "base_optimizer."


class PrecisionReporter:
    """Report stored dtypes and observed tensor boundaries, not every kernel's math."""

    def __init__(self, logger, enabled=True):
        self.logger = logger
        self.enabled = enabled
        self._reported = set()
        self._attempts = defaultdict(int)

    def _pending(self, key):
        return self.enabled and key not in self._reported

    def _log(self, message):
        self.logger.info("[precision] %s", message)

    def log_startup(self, args, optimizer, network, base_model, text_encoders, vae, save_dtype):
        if not self._pending("startup"):
            return
        self._reported.add("startup")
        raw = _raw_optimizer(optimizer)
        self._log(
            f"settings: mixed_precision={args.mixed_precision}, "
            f"full_fp16={getattr(args, 'full_fp16', False)}, "
            f"full_bf16={getattr(args, 'full_bf16', False)}; "
            f"optimizer={type(raw).__module__}.{type(raw).__name__}"
        )
        modules = [("network trainable parameters", network, True),
                   ("base frozen parameters", base_model, False),
                   ("VAE frozen parameters", vae, False)]
        modules.extend((f"text encoder {i + 1} frozen parameters", enc, False)
                       for i, enc in enumerate(text_encoders))
        for label, module, trainable in modules:
            summary = (tensor_summary(p for p in module.parameters() if p.requires_grad == trainable)
                       if module is not None else "unloaded / not available")
            self._log(f"{label}: {summary}")
        self._log(f"optimizer parameter storage: {tensor_summary(_optimizer_parameters(optimizer))}")
        export = dtype_name(save_dtype) if save_dtype is not None else "unchanged from training tensors"
        self._log(f"checkpoint export dtype: {export} (save only; does not set training dtype)")
        self._log("Gradient buffers and optimizer state are observed during training; "
                  "tensor storage/output dtype does not imply every internal operation uses that dtype.")

    def log_tensors(self, stage, **tensors):
        if not self._pending(stage):
            return
        self._reported.add(stage)
        details = ", ".join(f"{name}={dtype_name(value.dtype)}@{value.device}"
                            for name, value in tensors.items() if isinstance(value, torch.Tensor))
        self._log(f"{stage}: {details or 'no tensors observed'}")

    def log_forward(self, device_type, **tensors):
        if not self._pending("training forward boundary"):
            return
        enabled = torch.is_autocast_enabled(device_type)
        target = dtype_name(torch.get_autocast_dtype(device_type)) if enabled else "disabled"
        self._log(f"observed autocast: enabled={enabled}, target={target} (operator dependent)")
        self.log_tensors("training forward boundary", **tensors)

    def log_gradients(self, optimizer):
        if not self._pending("gradients"):
            return
        parameters = list(_optimizer_parameters(optimizer))
        gradients = [p.grad for p in parameters if p.grad is not None]
        self._attempts["gradients"] += 1
        if gradients or self._attempts["gradients"] >= 3:
            self._log(f"gradient buffers before optimizer step: {tensor_summary(gradients)}; "
                      f"{len(parameters) - len(gradients):,} parameters have grad=None")
            if not gradients:
                self._log("Gradient dtype could not be observed after backward; hooks may have cleared buffers.")
            self._reported.add("gradients")

    def log_optimizer_state(self, optimizer, *, sync_gradients, step_was_skipped=False):
        if not self._pending("optimizer state") or not sync_gradients or step_was_skipped:
            return
        grouped = defaultdict(list)
        parameter_ids = {id(p) for p in _optimizer_parameters(optimizer)}
        for prefix, state_optimizer in _state_optimizers(optimizer):
            for state in getattr(state_optimizer, "state", {}).values():
                for key, tensor in _state_tensors(state, parameter_ids=parameter_ids):
                    grouped[prefix + key].append(tensor)
        self._attempts["optimizer state"] += 1
        if grouped:
            self._log("optimizer state observed after an update boundary (local stored tensors):")
            for key, tensors in sorted(grouped.items()):
                self._log(f"  state.{key}: {tensor_summary(tensors)}")
            self._reported.add("optimizer state")
        elif self._attempts["optimizer state"] >= 3:
            self._log("optimizer state: no tensor state exposed after 3 update boundaries; "
                      "dtype unknown or optimizer is stateless")
            self._reported.add("optimizer state")
