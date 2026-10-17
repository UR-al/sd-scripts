"""CPU-only checks that precision diagnostics observe, rather than alter, training."""

import copy
import logging
from types import SimpleNamespace

import unittest
import torch

from library.precision_logging import PrecisionReporter, tensor_summary


LOGGER = logging.getLogger("test.precision")


class CapturedLogs(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    @property
    def text(self):
        return "\n".join(record.getMessage() for record in self.records)


class PrecisionLoggingTests(unittest.TestCase):
    def setUp(self):
        self.logs = CapturedLogs()
        self.old_level = LOGGER.level
        LOGGER.setLevel(logging.INFO)
        LOGGER.addHandler(self.logs)

    def tearDown(self):
        LOGGER.removeHandler(self.logs)
        LOGGER.setLevel(self.old_level)

    def test_fp32_adamw_with_bf16_autocast(self):
        self._check_adamw_update(torch.float32, "FP32")

    def test_bf16_adamw_is_reported_as_bf16(self):
        self._check_adamw_update(torch.bfloat16, "BF16")


    def _check_adamw_update(self, parameter_dtype, label):
        model = torch.nn.Linear(4, 2, device="cpu", dtype=parameter_dtype)
        control = copy.deepcopy(model)
        base = torch.nn.Linear(4, 2, device="cpu", dtype=torch.bfloat16).requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        control_optimizer = torch.optim.AdamW(control.parameters(), lr=1e-3)
        # Exercise the wrapper used by the trainer without importing Accelerate's model stack.
        wrapped = SimpleNamespace(optimizer=optimizer)
        reporter = PrecisionReporter(LOGGER)
        args = SimpleNamespace(mixed_precision="bf16", full_fp16=False, full_bf16=parameter_dtype == torch.bfloat16)
        reporter.log_startup(args, wrapped, model, base, [None], None, torch.bfloat16)
        assert len(optimizer.state) == 0  # Logging must not lazily create optimizer state.
        assert f"network trainable parameters: {label}@cpu" in self.logs.text
        assert "base frozen parameters: BF16@cpu" in self.logs.text
        assert "checkpoint export dtype: BF16 (save only" in self.logs.text

        inputs = torch.arange(12, dtype=torch.float32, device="cpu").reshape(3, 4) / 12
        rng_before = torch.get_rng_state().clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = model(inputs)
            reporter.log_forward("cpu", prediction=prediction, inputs=inputs)
            control_prediction = control(inputs)
        loss = prediction.float().square().mean()
        control_loss = control_prediction.float().square().mean()
        reporter.log_tensors("loss passed to backward", loss=loss)
        loss.backward()
        control_loss.backward()
        gradients_before = [p.grad.clone() for p in model.parameters()]
        reporter.log_gradients(wrapped)
        for parameter, gradient in zip(model.parameters(), gradients_before):
            assert torch.equal(parameter.grad, gradient)

        # An accumulation micro-step and an overflow-skipped step must not consume the report.
        reporter.log_optimizer_state(wrapped, sync_gradients=False)
        reporter.log_optimizer_state(wrapped, sync_gradients=True, step_was_skipped=True)
        assert "state.exp_avg:" not in self.logs.text
        optimizer.step()
        control_optimizer.step()
        state_before = copy.deepcopy(optimizer.state_dict())
        reporter.log_optimizer_state(wrapped, sync_gradients=True)
        assert "observed autocast: enabled=True, target=BF16" in self.logs.text
        assert "prediction=BF16@cpu" in self.logs.text
        assert "loss passed to backward: loss=FP32@cpu" in self.logs.text
        assert f"gradient buffers before optimizer step: {label}@cpu" in self.logs.text
        assert f"state.exp_avg: {label}@cpu" in self.logs.text
        assert f"state.exp_avg_sq: {label}@cpu" in self.logs.text
        for actual, expected in zip(model.parameters(), control.parameters()):
            assert torch.equal(actual, expected)
        for key, state in optimizer.state_dict()["state"].items():
            for name, value in state.items():
                assert torch.equal(value, state_before["state"][key][name])
                assert torch.equal(value, control_optimizer.state_dict()["state"][key][name])
        assert torch.equal(torch.get_rng_state(), rng_before)
        optimizer.zero_grad(set_to_none=True)
        log_count = len(self.logs.records)
        reporter.log_gradients(wrapped)
        reporter.log_optimizer_state(wrapped, sync_gradients=True)
        reporter.log_forward("cpu", prediction=prediction)
        assert len(self.logs.records) == log_count  # No repeated hot-loop logging.


    def test_nested_mixed_optimizer_state_and_resume_are_not_mislabeled(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        raw = SimpleNamespace(
            param_groups=[{"params": [parameter]}],
            state={parameter: {"step": torch.tensor(8.), "moments": [torch.ones(2, dtype=torch.uint8)],
                               "nested": {"residual": torch.ones(2, dtype=torch.bfloat16)}}},
        )
        reporter = PrecisionReporter(LOGGER)
        reporter.log_optimizer_state(SimpleNamespace(optimizer=raw), sync_gradients=True)
        assert "state.moments[]: uint8@cpu" in self.logs.text
        assert "state.nested.residual: BF16@cpu" in self.logs.text
        assert "state.step: FP32@cpu" in self.logs.text


    def test_missing_gradients_and_state_are_explicitly_unknown_with_bounded_retries(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(2))], lr=0.1)
        reporter = PrecisionReporter(LOGGER)
        for _ in range(8):
            reporter.log_gradients(optimizer)
            reporter.log_optimizer_state(optimizer, sync_gradients=True)
        assert self.logs.text.count("dtype unknown or optimizer is stateless") == 1
        assert self.logs.text.count("Gradient dtype could not be observed") == 1
        assert "FP32" not in self.logs.text  # Do not infer precision from parameter defaults.

    def test_wrapper_base_state_excludes_parameters_and_handles_cycles(self):
        parameter = torch.nn.Parameter(torch.ones(2))
        group = {"params": [parameter], "lr": 0.1}
        state = {"group": group, "parameter_reference": parameter, "z": torch.ones(2)}
        state["cycle"] = state
        base = SimpleNamespace(state={parameter: {"exp_avg": torch.ones(2, dtype=torch.bfloat16)}})
        raw = SimpleNamespace(param_groups=[group], state={parameter: state}, base_optimizer=base)
        reporter = PrecisionReporter(LOGGER)
        reporter.log_optimizer_state(SimpleNamespace(optimizer=raw), sync_gradients=True)
        assert "state.z: FP32@cpu" in self.logs.text
        assert "state.base_optimizer.exp_avg: BF16@cpu" in self.logs.text
        assert "state.group" not in self.logs.text
        assert "state.parameter_reference" not in self.logs.text


    def test_disabled_reporter_never_inspects_models_or_tensors(self):
        reporter = PrecisionReporter(LOGGER, enabled=False)
        reporter.log_startup(None, None, None, None, None, None, None)
        reporter.log_forward("not-a-device")
        reporter.log_gradients(None)
        reporter.log_optimizer_state(None, sync_gradients=True)
        assert not self.logs.records


    def test_summary_deduplicates_shared_parameters_and_keeps_mixed_dtypes(self):
        first = torch.ones(3)
        second = torch.ones(5, dtype=torch.bfloat16)
        result = tensor_summary([first, second, first, None])
        assert "FP32@cpu: 1 tensors / 3 elements" in result
        assert "BF16@cpu: 1 tensors / 5 elements" in result


if __name__ == "__main__":
    unittest.main()
