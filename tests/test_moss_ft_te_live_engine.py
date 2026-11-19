"""CPU tests of library/moss/te_live.py (docs/moss_ft_te_live_fp16_plan.md 1.2, 1.6): the layer-major streamed pass
gives every caption's cached-path encoding bit for bit (lengths around the 64-token chunks of the gated delta rule,
2 and 4 ring slots, copy and mmap host forms, with and without the threaded stager), the shell is unbound after a pass,
nothing carries requires_grad and the outputs work in an autograd graph (no inference tensors), the strategy branch,
and a failing stager surfaces in the main thread without leaving the shell bound.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_te_live_engine.py -q -p no:cacheprovider
"""
import os
import pickle

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch

from _moss_te_tiny import LENGTHS, TINY_TE, live_encoder, random_ids, reference_model, tokenize_strategy, write_tiny_te
from library.moss import strategy, te_live


@pytest.fixture(scope="module")
def te_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("te_live_engine") / "te.safetensors"
    write_tiny_te(path)
    return path


@pytest.fixture(scope="module")
def reference(te_file):
    return reference_model(te_file)


def _rows():
    return [random_ids(length, seed=100 + length) for length in LENGTHS]


@pytest.mark.parametrize("host", ["copy", "mmap"])
@pytest.mark.parametrize("ring_slots", [2, 4])
@pytest.mark.parametrize("staging", [False, True])
def test_every_row_equals_the_cached_encoder(te_file, reference, host, ring_slots, staging):
    encoder = live_encoder(te_file, host, ring_slots, staging)
    try:
        rows = _rows()
        out = encoder.encode_rows(rows)
        assert len(out) == len(rows)
        for ids, hidden in zip(rows, out):
            expected = reference(ids[None])[0]
            assert hidden.dtype == torch.bfloat16 and hidden.shape == (ids.numel(), TINY_TE.hidden_size)
            assert torch.equal(hidden, expected), (host, ring_slots, staging, ids.numel())
        assert encoder.bound_layers() == []
        assert encoder.passes == 1 and encoder.stats["rows"] == len(rows)
        assert encoder.stats["tokens"] == sum(LENGTHS)
        again = encoder.encode_rows(rows[:2])  # a second pass on the same encoder
        assert all(torch.equal(a, b) for a, b in zip(again, out[:2]))
    finally:
        encoder.close()


def test_one_row_and_one_token(te_file, reference):
    encoder = live_encoder(te_file)
    try:
        for length in (1, 2):
            ids = random_ids(length, seed=length)
            assert torch.equal(encoder.encode_rows([ids])[0], reference(ids[None])[0])
        assert encoder.encode_rows([]) == []
        with pytest.raises(ValueError, match="chat-template"):
            encoder.encode_rows([torch.zeros(0, dtype=torch.long)])
    finally:
        encoder.close()


def test_no_gradients_and_usable_in_an_autograd_graph(te_file):
    encoder = live_encoder(te_file)
    try:
        assert not any(p.requires_grad for p in encoder.shell.parameters())
        hidden = encoder.encode_rows([random_ids(20, seed=1)])[0]
        assert not hidden.requires_grad and not hidden.is_inference()
        proj = torch.nn.Linear(TINY_TE.hidden_size, 3, dtype=torch.bfloat16)
        loss = proj(hidden).float().square().mean()  # the DiT saves the text tensor for its backward the same way
        loss.backward()
        assert proj.weight.grad is not None and torch.isfinite(proj.weight.grad.float()).all()
        with torch.enable_grad():  # even when the caller has grad mode on, the pass records no graph
            assert not encoder.encode_rows([random_ids(5, seed=2)])[0].requires_grad
    finally:
        encoder.close()


def test_shell_runs_qwen35_code_on_meta_outside_a_pass(te_file):
    encoder = live_encoder(te_file)
    try:
        for name, param in encoder.shell.layers.named_parameters():
            assert param.device.type == "meta", name
        assert encoder.shell.norm.weight.device.type == "cpu"
        assert encoder.slot_bytes == max(layout.nbytes for layout in encoder.layouts)
        with pytest.raises(TypeError, match="never pickled"):
            pickle.dumps(encoder)
        with pytest.raises(ValueError, match="ring_slots"):
            te_live.LiveTextEncoder(encoder.weights, "cpu", ring_slots=3)
    finally:
        encoder.close()


def test_strategy_branch_equals_the_cached_strategy(te_file, reference, monkeypatch):
    """MossTextEncodingStrategy.encode_tokens through the live encoder = through the loaded model (embeds and mask);
    resolve_text_encoder returns the live encoder as is."""
    tok = tokenize_strategy(monkeypatch)
    encoding = strategy.MossTextEncodingStrategy()
    texts = ["a red square", "two blue circles on a white table, sitting", ""]
    tokens = tok.tokenize(texts)
    encoder = live_encoder(te_file)
    try:
        assert strategy.resolve_text_encoder(encoder) is encoder
        live = encoding.encode_tokens(tok, [encoder], tokens)
        cached = encoding.encode_tokens(tok, [reference], tokens)
        assert torch.equal(live[0], cached[0]) and torch.equal(live[1], cached[1])
        assert live[1].dtype == torch.long and live[0].dtype == torch.bfloat16
        provider = strategy.MossTextEncoderProvider("unused", "cpu", loader=lambda *a, **k: reference)
        via_provider = encoding.encode_tokens(tok, [provider], tokens)
        assert torch.equal(via_provider[0], live[0])
    finally:
        encoder.close()


def test_a_failing_stager_raises_in_the_main_thread(te_file, reference, monkeypatch):
    encoder = live_encoder(te_file, staging=True)
    try:
        real = encoder._stage
        calls = []

        def stage(index, s, done):
            calls.append(index)
            if index == 5:
                raise OSError("disk read failed")
            return real(index, s, done)

        monkeypatch.setattr(encoder, "_stage", stage)
        with pytest.raises(OSError, match="disk read failed"):
            encoder.encode_rows(_rows())
        assert encoder.bound_layers() == []  # every layer unbound again
        monkeypatch.setattr(encoder, "_stage", real)
        ids = random_ids(65, seed=7)
        assert torch.equal(encoder.encode_rows([ids])[0], reference(ids[None])[0])  # usable after the failure
    finally:
        encoder.close()
    assert encoder._executor is None


def test_a_failing_layer_unbinds_and_raises(te_file, monkeypatch):
    encoder = live_encoder(te_file)
    try:
        layer = encoder.shell.layers[4]

        def broken(*args, **kwargs):
            raise RuntimeError("layer failed")

        monkeypatch.setattr(layer, "forward", broken)
        with pytest.raises(RuntimeError, match="layer failed"):
            encoder.encode_rows([random_ids(12, seed=1)])
        assert encoder.bound_layers() == []
    finally:
        encoder.close()


def test_a_pass_stages_its_first_layers_after_the_previous_uploads(te_file, reference):
    """Review fix: the first R layers of a pass are staged into the entries the previous pass's last H2D copies read
    from; the stager waits for the event recorded after those copies (on the CPU: a completed stand-in)."""
    encoder = live_encoder(te_file, "copy", 2, True)

    class Gate:
        calls = 0

        def synchronize(self):
            Gate.calls += 1

    try:
        rows = _rows()[:2]
        encoder._uploads_done = Gate()
        out = encoder.encode_rows(rows)
        assert Gate.calls == 2  # one wait per ring slot, before its first staging of the pass
        assert all(torch.equal(h, reference(ids[None])[0]) for ids, h in zip(rows, out))
    finally:
        encoder.close()
    assert encoder._uploads_done is te_live._DONE
