"""CPU tests of the training-state directory (library/moss/ft_state.py; docs/moss_full_finetune_design.md 4.7, section-5
rows test_save_resume_equivalence / test_incomplete_state_refused / test_keep_policy / test_master_file_is_a_loadable_dit).

The state is written atomically (<name>.tmp, fsync, COMPLETE.json last, rename; an existing <name> moved to <name>.old
and deleted after), verified (COMPLETE.json and the recorded sizes, the slot list hash, the optimizer mode) and loaded
in place. Resume equivalence runs the real loop - load_moss_for_full_ft, the host optimizer pipeline, the LR phase
scheduler and moss_ft_noise_and_loss with its RNG draws - 4 windows straight against 2 windows, a save, new objects, a
load and 2 windows: master, moments, steps and compute copies bit-identical.

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_state.py -q -p no:cacheprovider
"""
import contextlib
import json
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _moss_ft_model_common as M  # noqa: E402

from library import sd3_train_utils  # noqa: E402
from library.moss import flow, ft_args, ft_host, ft_loss, ft_model, ft_state, strategy  # noqa: E402
from library.moss.ft_optim import HostAdamW, HostCAME  # noqa: E402

GA = 2
PHASES = "1:1e-3,1:4e-4"  # 2 optimizer steps per epoch: phase 1 = steps 0-1, phase 2 = steps 2-3


class _Accel:
    device = torch.device("cpu")

    def __init__(self, dtype=None):
        self.dtype = dtype

    def autocast(self):
        return torch.autocast("cpu", dtype=self.dtype) if self.dtype is not None else contextlib.nullcontext()


def _args(**kw):
    """moss_train.py's parser with the run's options (what the loader, the optimizer and the loss read)."""
    args = ft_args.setup_parser().parse_args([])
    values = dict(mixed_precision="no", moss_stream_fp32=False, optimizer_type="AdamW",
                  optimizer_args=["betas=(0.9,0.999)", "weight_decay=0.01"], gradient_accumulation_steps=GA,
                  max_grad_norm=0.05, moss_ft_clip_mode="stale", moss_ft_lr_phases=PHASES, learning_rate=1e-3,
                  gradient_checkpointing=True, output_name="tiny", timestep_sampling="moss", moss_ft_freeze="")
    values.update(kw)
    for key, value in values.items():
        setattr(args, key, value)
    return args


def _batch(w, j):
    g = torch.Generator().manual_seed(100 * w + j)
    return {"latents": torch.randn(1, M.TINY["latent_channels"], 4, 4, generator=g),
            "text_encoder_outputs_list": [torch.randn(1, 6, M.TINY["text_dim"], generator=g).to(torch.bfloat16),
                                          torch.ones(1, 6, dtype=torch.long), torch.zeros(1)],
            "loss_weights": torch.ones(1)}


class Run:
    """Every object of one (simulated) process: model, pool, optimizer, scheduler, pipeline, loss."""

    def __init__(self, base_path, args, inline=True):
        self.args = args
        self.model, self.pool, self.slots = ft_model.load_moss_for_full_ft(str(base_path), args, "cpu", config=M.TINY)
        self.model.enable_gradient_checkpointing()
        self.mode = ft_args.optimizer_mode(args)
        kwargs = ft_args.parse_optimizer_args(args, self.mode, quiet=True)
        named = ft_model.trainable_named(self.slots)
        groups = ft_model.param_groups(named, kwargs.get("weight_decay", 0.0))
        cls = HostAdamW if self.mode == ft_args.ADAMW_HOST else HostCAME
        self.optimizer = cls(groups, lr=args.learning_rate, **kwargs)
        ft_host.bind_slots(self.slots, named, self.optimizer)
        if args.moss_ft_lr_phases:
            self.scheduler = ft_args.make_lr_phase_scheduler(self.optimizer, args.moss_ft_lr_phases, 2, 1)
        else:  # moss_train.py's scheduler without phases: train_util.get_scheduler_fix (2 steps per epoch)
            import moss_train

            self.scheduler = moss_train.make_lr_scheduler(args, self.optimizer, 2, 1)
        self.lrs = []  # the LRs each window stepped with (what StepContext.snapshot read)
        self.pipe = ft_host.GradPipeline(self.slots, self.pool, self.optimizer, device="cpu",
                                         gradient_accumulation_steps=GA, max_grad_norm=args.max_grad_norm,
                                         clip_mode=args.moss_ft_clip_mode, piece_bytes=2048, ring_bytes=8 * 2048,
                                         upload_ring_bytes=4 * 2048, acc_chunk=128, step_chunk=128, inline=inline)
        self.pipe.attach()
        self.accel = _Accel(torch.bfloat16 if args.mixed_precision == "bf16" else None)
        self.sampler = flow.MossSigmaSampler(args)
        self.noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
                                                                               shift=args.discrete_flow_shift)
        self.stream = torch.float32 if (args.moss_stream_fp32 or args.mixed_precision == "no") else torch.bfloat16
        self.te = strategy.MossTextEncodingStrategy()
        self.global_step = 0

    def windows(self, count):
        for _ in range(count):
            w = self.global_step
            self.lrs.append([g["lr"] for g in self.optimizer.param_groups])
            self.pipe.begin_window(GA, ft_host.StepContext.snapshot(self.optimizer, w))
            stats = None
            for j in range(GA):
                self.pipe.begin_microstep(j)
                loss = ft_loss.moss_ft_noise_and_loss(self.args, self.model, _batch(w, j), self.accel, self.sampler,
                                                      self.noise_scheduler, self.stream, self.te)
                (loss / GA).backward()
                stats = self.pipe.end_microstep()
            assert stats is not None and stats.skipped_params == 0
            self.scheduler.step()
            self.global_step += 1

    def progress(self, epoch_step=0):
        return ft_state.TrainProgress(global_step=self.global_step, completed_epochs=self.global_step // 2,
                                      epoch_step=epoch_step, steps_per_epoch=2, gradient_accumulation_steps=GA,
                                      lr_schedule=self.scheduler.schedule() if hasattr(self.scheduler, "schedule")
                                      else None)

    def save(self, state_dir, name, **kw):
        return ft_state.write_state(state_dir, name, model=self.model, slots=self.slots, pool=self.pool,
                                    optimizer=self.optimizer, lr_scheduler=self.scheduler, mode=self.mode,
                                    progress=kw.pop("progress", None) or self.progress(), args=self.args,
                                    last_grad_norm=self.pipe.last_grad_norm, skip_streak=self.pipe.skip_streak, **kw)

    def load(self, path, **kw):
        info = ft_state.load_state(path, model=self.model, slots=self.slots, pool=self.pool, optimizer=self.optimizer,
                                   lr_scheduler=self.scheduler, mode=self.mode, args=self.args, **kw)
        self.pipe.restore_counters(info.last_grad_norm, info.skip_streak)
        self.global_step = info.progress.global_step
        return info

    def close(self):
        self.pipe.close()


def _perturb_rng():
    torch.manual_seed(987654)
    random.seed(987654)
    np.random.seed(987654)


def _assert_same(a: Run, b: Run):
    assert [s.name for s in a.slots] == [s.name for s in b.slots]
    for role in a.pool.roles:
        assert torch.equal(a.pool.roles[role], b.pool.roles[role]), role
    assert torch.equal(a.pool.steps, b.pool.steps)
    for (name, p), (_, q) in zip(a.model.named_parameters(), b.model.named_parameters()):
        assert p.dtype == q.dtype and torch.equal(p.detach(), q.detach()), name
    if hasattr(a.optimizer, "extra_state"):
        ea, eb = a.optimizer.extra_state(), b.optimizer.extra_state()
        assert ea.keys() == eb.keys() and all(torch.equal(ea[k], eb[k]) for k in ea)
    assert a.scheduler.last_epoch == b.scheduler.last_epoch
    assert [g["lr"] for g in a.optimizer.param_groups] == [g["lr"] for g in b.optimizer.param_groups]
    assert a.pipe.last_grad_norm == b.pipe.last_grad_norm


RESUME_CASES = {
    "adamw_fp32": dict(),
    "adamw_bf16_stream_fp32": dict(mixed_precision="bf16", moss_stream_fp32=True),
    "came_fp32_bias_correction": dict(optimizer_type="CAME",
                                      optimizer_args=["betas=(0.9,0.999,0.9999)", "bias_correction=True",
                                                      "weight_decay=0.01"]),
    "adamw_threaded_frozen_block": dict(moss_ft_freeze=r"layers\.1\..*"),
}


@pytest.mark.parametrize("case", sorted(RESUME_CASES))
def test_save_resume_equivalence(tmp_path, case):
    base_path, _ = M.write_base(tmp_path)
    args = _args(**RESUME_CASES[case])
    inline = case != "adamw_threaded_frozen_block"
    seed = 2024
    torch.manual_seed(seed)
    straight = Run(base_path, args, inline=inline)
    try:
        straight.windows(4)
    finally:
        straight.close()

    torch.manual_seed(seed)
    first = Run(base_path, args, inline=inline)
    try:
        first.windows(2)
        path = first.save(str(tmp_path / "states"), ft_state.epoch_state_name("tiny", 1))
    finally:
        first.close()
    assert os.path.basename(path) == "tiny-000001-state"
    _perturb_rng()
    resumed = Run(base_path, args, inline=inline)  # "a new process": every object made again
    try:
        info = resumed.load(path)
        assert info.progress.global_step == 2 and info.start_epoch == 1 and info.batches_to_skip == 0
        assert info.mode == straight.mode and info.progress.lr_schedule == straight.scheduler.schedule()
        assert resumed.scheduler.last_epoch == 2
        assert [g["lr"] for g in resumed.optimizer.param_groups] == [4e-4, 4e-4]  # phase 2 right after the load
        resumed.windows(2)
        _assert_same(straight, resumed)
    finally:
        resumed.close()


def test_resave_after_load_is_identical(tmp_path):
    """Design 7.4's check on the CPU: a state saved right after loading equals the loaded one file for file."""
    base_path, _ = M.write_base(tmp_path)
    args = _args()
    run = Run(base_path, args)
    try:
        run.windows(2)
        first = run.save(str(tmp_path / "a"), "s")
    finally:
        run.close()
    again = Run(base_path, args)
    try:
        again.load(first)
        second = again.save(str(tmp_path / "b"), "s")
    finally:
        again.close()
    for name in ("master.safetensors", "exp_avg.safetensors", "exp_avg_sq.safetensors", "optimizer_steps.safetensors",
                 "rng.pt", "scheduler.pt", "param_groups.pt"):
        with open(os.path.join(first, name), "rb") as f, open(os.path.join(second, name), "rb") as g:
            assert f.read() == g.read(), name


def test_state_files_and_marker(tmp_path):
    base_path, base = M.write_base(tmp_path)
    run = Run(base_path, _args(moss_ft_freeze=r"layers\.2\..*"))
    try:
        run.windows(1)
        path = run.save(str(tmp_path), "x-step00000001-state", progress=run.progress(epoch_step=2))
    finally:
        run.close()
    names = sorted(os.listdir(path))
    assert names == sorted(["train_state.json", "master.safetensors", "exp_avg.safetensors", "exp_avg_sq.safetensors",
                            "optimizer_steps.safetensors", "param_groups.pt", "scheduler.pt", "rng.pt",
                            "COMPLETE.json"])
    groups = torch.load(os.path.join(path, "param_groups.pt"), weights_only=True)
    assert [g["name"] for g in groups] == ["decay", "no_decay"] and all("params" not in g for g in groups)
    assert [g["lr"] for g in groups] == [g["lr"] for g in run.optimizer.param_groups]
    with open(os.path.join(path, "COMPLETE.json"), encoding="utf-8") as f:
        complete = json.load(f)
    assert complete["format"] == "moss-ft-state/1" and complete["global_step"] == 1
    assert complete["files"] == {n: os.path.getsize(os.path.join(path, n)) for n in names if n != "COMPLETE.json"}
    with open(os.path.join(path, "train_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    assert state["global_step"] == 1 and state["completed_epochs"] == 0 and state["epoch_step"] == 2
    assert state["steps_per_epoch"] == 2 and state["gradient_accumulation_steps"] == GA
    assert state["mode"] == "adamw_host" and state["slot_hash"] == ft_host.slot_list_hash(run.slots)
    assert state["lr_schedule"]["bounds"] == [2, 4] and state["torch_version"] == torch.__version__
    assert state["last_grad_norm"] == run.pipe.last_grad_norm and state["skip_streak"] == 0
    assert [g["name"] for g in state["hyperparameters"]] == ["decay", "no_decay"]
    assert len(state["args_digest"]) == 64 and "huggingface_token" not in state["args"]
    # moments and steps keyed by the trained parameter names; the master holds every Base key in fp32
    meta, infos = ft_state.ft_safetensors.read_header(os.path.join(path, "exp_avg.safetensors"))
    assert list(infos) == [s.name for s in run.slots]
    meta, infos = ft_state.ft_safetensors.read_header(os.path.join(path, "master.safetensors"))
    assert meta["format"] == "pt" and list(infos) == sorted(base)
    assert all(i.dtype == torch.float32 for i in infos.values())
    meta, infos = ft_state.ft_safetensors.read_header(os.path.join(path, "optimizer_steps.safetensors"))
    assert list(infos) == [s.name for s in run.slots] and all(i.dtype == torch.int64 for i in infos.values())


def test_master_file_is_a_loadable_dit(tmp_path, monkeypatch):
    import moss_train_network as mtn

    base_path, base = M.write_base(tmp_path)
    run = Run(base_path, _args(moss_ft_freeze=r"layers\.0\..*"))
    try:
        run.windows(2)
        path = run.save(str(tmp_path / "states"), "s")
    finally:
        run.close()
    monkeypatch.setattr(mtn, "MOSS_DIT_CONFIG", M.TINY)
    dit = mtn.load_moss_dit(os.path.join(path, "master.safetensors"), "torch", False, torch.device("cpu"),
                            torch.float32).state_dict()
    trained = {s.name: s for s in run.slots}
    for name, tensor in dit.items():
        if name in trained:
            s = trained[name]
            assert torch.equal(tensor, run.pool.view("master", s).view(s.shape)), name
        elif name != "rope.bands":
            assert name.startswith("layers.0.") and torch.equal(tensor, base[name].float()), name
    # and a full FT can start from it (an fp32 file: masters exact)
    model, pool, slots = ft_model.load_moss_for_full_ft(os.path.join(path, "master.safetensors"), _args(), "cpu",
                                                        config=M.TINY)
    for s in slots:
        want = run.pool.view("master", trained[s.name]) if s.name in trained else base[s.name].float().reshape(-1)
        assert torch.equal(pool.view("master", s), want), s.name


def _two_states(tmp_path):
    base_path, _ = M.write_base(tmp_path)
    args = _args()
    run = Run(base_path, args)
    state_dir = str(tmp_path / "states")
    try:
        run.windows(1)
        older = run.save(state_dir, ft_state.step_state_name("tiny", 1))
        run.windows(1)
        newer = run.save(state_dir, ft_state.step_state_name("tiny", 2))
    finally:
        run.close()
    return base_path, args, state_dir, older, newer


def test_incomplete_state_refused(tmp_path):
    base_path, args, state_dir, older, newer = _two_states(tmp_path)
    run = Run(base_path, args)
    try:
        marker = os.path.join(newer, "COMPLETE.json")
        saved = open(marker, "rb").read()
        os.remove(marker)
        with pytest.raises(ft_state.StateError, match="COMPLETE.json") as error:
            run.load(newer)
        assert "tiny-step00000001-state" in str(error.value) and "newest complete" in str(error.value)
        with open(marker, "wb") as f:
            f.write(saved)
        moments = os.path.join(newer, "exp_avg.safetensors")
        data = open(moments, "rb").read()
        with open(moments, "wb") as f:
            f.write(data[:-16])
        with pytest.raises(ft_state.StateError, match="size") as error:
            run.load(newer)
        assert "tiny-step00000001-state" in str(error.value)
        with open(moments, "wb") as f:
            f.write(data)
        assert run.load(newer).progress.global_step == 2  # complete again
    finally:
        run.close()
    # a changed trainable set (another --moss_ft_freeze) and another optimizer mode
    frozen = Run(base_path, _args(moss_ft_freeze=r"layers\.1\..*"))
    try:
        with pytest.raises(ft_state.StateError, match="trainable"):
            frozen.load(newer)
    finally:
        frozen.close()
    came = Run(base_path, _args(optimizer_type="CAME", optimizer_args=["bias_correction=True"]))
    try:
        with pytest.raises(ft_state.StateError, match="mode"):
            came.load(newer)
    finally:
        came.close()


def test_accelerate_and_missing_states_refused(tmp_path):
    base_path, args, state_dir, older, newer = _two_states(tmp_path)
    accelerate = os.path.join(state_dir, "tiny-step00000003-state")
    os.makedirs(accelerate)
    for name in ("optimizer.bin", "random_states_0.pkl", "model.safetensors"):
        open(os.path.join(accelerate, name), "wb").close()
    run = Run(base_path, args)
    try:
        with pytest.raises(ft_state.StateError, match="accelerate") as error:
            run.load(accelerate)
        assert "tiny-step00000002-state" in str(error.value)
        with pytest.raises(ft_state.StateError, match="not a directory"):
            run.load(os.path.join(state_dir, "nothing-here"))
    finally:
        run.close()
    assert [os.path.basename(p) for p in ft_state.complete_states(state_dir)] == ["tiny-step00000002-state",
                                                                                  "tiny-step00000001-state"]


def test_interrupted_write_keeps_the_previous_state(tmp_path, monkeypatch):
    base_path, args, state_dir, older, newer = _two_states(tmp_path)
    run = Run(base_path, args)
    try:
        run.load(newer)
        before = {n: open(os.path.join(newer, n), "rb").read() for n in os.listdir(newer)}

        def broken(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(ft_state, "capture_rng_state", broken)
        with pytest.raises(OSError, match="disk full"):
            run.save(state_dir, os.path.basename(newer))
        assert {n: open(os.path.join(newer, n), "rb").read() for n in os.listdir(newer)} == before
        assert sorted(os.listdir(state_dir)) == ["tiny-step00000001-state", "tiny-step00000002-state"]
    finally:
        run.close()


def test_keep_policy(tmp_path):
    base_path, _ = M.write_base(tmp_path)
    args = _args(save_last_n_epochs_state=1, save_last_n_steps_state=1, moss_ft_lr_phases="2:1e-3,2:4e-4")
    policy = ft_state.keep_policy_from_args(args)
    assert policy["pinned_epochs"] == [2] and policy["keep_epoch_states"] == 1 and policy["keep_step_states"] == 1
    state_dir = str(tmp_path / "states")
    run = Run(base_path, args)
    try:
        # a directory that only looks like one of ours (an accelerate state of a LoRA run) is never touched
        foreign = os.path.join(state_dir, "tiny-step00000000-state")
        os.makedirs(foreign)
        open(os.path.join(foreign, "optimizer.bin"), "wb").close()
        for step in (1, 2, 3):
            run.save(state_dir, ft_state.step_state_name("tiny", step))
            deleted = ft_state.apply_keep_policy(state_dir, **policy)
            assert [os.path.basename(p) for p in deleted] == ([] if step == 1 else
                                                              [ft_state.step_state_name("tiny", step - 1)])
        for epoch in (1, 2, 3, 4):
            run.save(state_dir, ft_state.epoch_state_name("tiny", epoch))
            ft_state.apply_keep_policy(state_dir, **policy)
        run.save(state_dir, ft_state.last_state_name("tiny"))
        ft_state.apply_keep_policy(state_dir, **policy)
        assert sorted(os.listdir(state_dir)) == ["tiny-000002-state", "tiny-000004-state", "tiny-state",
                                                 "tiny-step00000000-state", "tiny-step00000003-state"]
        # without pinning (--moss_ft_no_keep_phase_states) the phase-1 state goes too; None keeps every state
        ft_state.apply_keep_policy(state_dir, **ft_state.keep_policy_from_args(
            _args(save_last_n_epochs_state=1, save_last_n_steps_state=None, moss_ft_lr_phases="2:1e-3,2:4e-4",
                  moss_ft_no_keep_phase_states=True)))
        assert sorted(os.listdir(state_dir)) == ["tiny-000004-state", "tiny-state", "tiny-step00000000-state",
                                                 "tiny-step00000003-state"]
    finally:
        run.close()


def test_atomic_replace_of_an_existing_name(tmp_path):
    base_path, _ = M.write_base(tmp_path)
    run = Run(base_path, _args())
    state_dir = str(tmp_path / "states")
    name = ft_state.last_state_name("tiny")
    try:
        run.windows(1)
        path = run.save(state_dir, name)
        first = open(os.path.join(path, "master.safetensors"), "rb").read()
        # leftovers of an interrupted earlier write of the same name are cleared
        os.makedirs(os.path.join(state_dir, name + ".tmp"))
        open(os.path.join(state_dir, name + ".tmp", "junk"), "wb").close()
        os.makedirs(os.path.join(state_dir, name + ".old"))
        run.windows(1)
        again = run.save(state_dir, name)
    finally:
        run.close()
    assert again == path and sorted(os.listdir(state_dir)) == [name]
    assert open(os.path.join(path, "master.safetensors"), "rb").read() != first
    assert ft_state.verify_state(path)["global_step"] == 2


def test_rng_state_round_trip():
    torch.manual_seed(5)
    random.seed(5)
    np.random.seed(5)
    state = ft_state.capture_rng_state()
    want = (torch.rand(3), random.random(), np.random.rand(2))
    _perturb_rng()
    ft_state.restore_rng_state(state)
    got = (torch.rand(3), random.random(), np.random.rand(2))
    assert torch.equal(want[0], got[0]) and want[1] == got[1] and np.array_equal(want[2], got[2])


def test_mode_b_state_round_trip(tmp_path):
    """adafactor_fused keeps no host pool: the state holds the model values (master.safetensors, fp32) and the torch
    optimizer's state_dict. Plain Adafactor steps (the fused hooks are moss_train.py's) before and after a resume."""
    from transformers.optimization import Adafactor

    base_path, _ = M.write_base(tmp_path)
    args = _args(optimizer_type="Adafactor", fused_backward_pass=True, gradient_accumulation_steps=1, max_grad_norm=0.0,
                 optimizer_args=["scale_parameter=False", "relative_step=False", "warmup_init=False"])

    def make():
        model, pool, slots = ft_model.load_moss_for_full_ft(str(base_path), args, "cpu", config=M.TINY)
        assert pool is None
        named = ft_model.trainable_named(slots)
        opt = Adafactor(ft_model.param_groups(named), lr=1e-3, scale_parameter=False, relative_step=False,
                        warmup_init=False)
        sched = ft_args.make_lr_phase_scheduler(opt, PHASES, 2, 1)
        return model, slots, opt, sched

    def steps(model, opt, sched, start, count):
        accel, te = _Accel(), strategy.MossTextEncodingStrategy()
        sampler, noise = flow.MossSigmaSampler(args), sd3_train_utils.FlowMatchEulerDiscreteScheduler(1000, shift=1.0)
        for w in range(start, start + count):
            loss = ft_loss.moss_ft_noise_and_loss(args, model, _batch(w, 0), accel, sampler, noise, torch.float32, te)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()

    torch.manual_seed(1)
    a = make()
    steps(a[0], a[2], a[3], 0, 4)
    torch.manual_seed(1)
    b = make()
    steps(b[0], b[2], b[3], 0, 2)
    path = ft_state.write_state(str(tmp_path / "states"), "s", model=b[0], slots=b[1], pool=None, optimizer=b[2],
                                lr_scheduler=b[3], mode="adafactor_fused",
                                progress=ft_state.TrainProgress(global_step=2, completed_epochs=1), args=args)
    assert "optimizer.pt" in os.listdir(path) and "exp_avg.safetensors" not in os.listdir(path)
    _perturb_rng()
    c = make()
    info = ft_state.load_state(path, model=c[0], slots=c[1], pool=None, optimizer=c[2], lr_scheduler=c[3],
                               mode="adafactor_fused", args=args)
    assert info.progress.global_step == 2
    steps(c[0], c[2], c[3], 2, 2)
    for (name, p), (_, q) in zip(a[0].named_parameters(), c[0].named_parameters()):
        assert torch.equal(p.detach(), q.detach()), name


def test_corrupt_file_of_the_recorded_size_is_refused(tmp_path):
    base_path, args, state_dir, older, newer = _two_states(tmp_path)
    moments = os.path.join(newer, "exp_avg_sq.safetensors")
    data = bytearray(open(moments, "rb").read())
    data[8:16] = b"\x00" * 8  # the JSON header no longer parses; the size is unchanged
    with open(moments, "wb") as f:
        f.write(bytes(data))
    run = Run(base_path, args)
    try:
        with pytest.raises(ft_state.StateError, match="exp_avg_sq.safetensors"):
            run.load(newer)
    finally:
        run.close()


def test_changed_steps_per_epoch_and_ga_are_warned(tmp_path, caplog):
    import logging

    base_path, args, state_dir, older, newer = _two_states(tmp_path)
    run = Run(base_path, args)
    try:
        with caplog.at_level(logging.WARNING, logger="library.moss.ft_state"):
            run.load(newer, steps_per_epoch=3)
        lines = [r.getMessage() for r in caplog.records if r.name == "library.moss.ft_state"]
        assert any("steps per epoch" in line and "saved 2" in line and "now 3" in line for line in lines), lines
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="library.moss.ft_state"):
            run.load(newer, steps_per_epoch=2)
        assert not [r for r in caplog.records if r.name == "library.moss.ft_state"]
    finally:
        run.close()


# ---- review fixes ---------------------------------------------------------------------------------------------------
CUSTOM_RESTARTS = "LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts.CosineAnnealingWarmRestarts"
NON_PHASE_SCHEDULERS = {  # moss_train.py without --moss_ft_lr_phases: any get_scheduler_fix scheduler (design 4.5)
    "cosine_warmup": dict(lr_scheduler="cosine", lr_warmup_steps=2),
    "linear": dict(lr_scheduler="linear"),
    "constant_with_warmup": dict(lr_scheduler="constant_with_warmup", lr_warmup_steps=4),
    "exponential": dict(lr_scheduler_type="ExponentialLR", lr_scheduler_args=["gamma=0.5"]),  # recursive: reads lr
    "custom_restarts": dict(lr_scheduler_type=CUSTOM_RESTARTS,  # keeps its cycle state in the param groups
                            lr_scheduler_args=["first_cycle_max_steps=4", "warmup_steps=1", "gamma=0.9",
                                               "min_lr=1e-6"]),
}


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("case", sorted(NON_PHASE_SCHEDULERS))
def test_resume_lr_policy_is_explicit(tmp_path, monkeypatch, case, override):
    """Default restores the saved LR; opt-in starts the entire newly configured schedule at step zero."""
    if case == "custom_restarts":
        pytest.importorskip("LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts")
    base_path, _ = M.write_base(tmp_path)
    args = _args(moss_ft_lr_phases=None, max_train_steps=8, **NON_PHASE_SCHEDULERS[case])
    first = Run(base_path, args)
    try:
        first.windows(3)
        path = first.save(str(tmp_path / "states"), "saved")
        saved_lrs = [g["lr"] for g in first.optimizer.param_groups]
        saved_moments = first.pool.roles["exp_avg"].clone()
    finally:
        first.close()
    new_args = _args(moss_ft_lr_phases=None, max_train_steps=12, learning_rate=5e-4,
                     moss_ft_resume_new_lr_schedule=override, **NON_PHASE_SCHEDULERS[case])
    parsed = ft_args.setup_parser().parse_args(["--moss_ft_resume_new_lr_schedule"] if override else [])
    assert parsed.moss_ft_resume_new_lr_schedule is override
    resumed = Run(base_path, new_args)
    fresh = Run(base_path, new_args)
    lines = []
    monkeypatch.setattr(ft_state.logger, "warning", lambda msg, *a, **kw: lines.append(str(msg)))
    try:
        resumed.load(path)
        assert resumed.global_step == 3
        assert torch.equal(resumed.pool.roles["exp_avg"], saved_moments)
        if override:
            assert resumed.scheduler.state_dict() == fresh.scheduler.state_dict()
            for _ in range(4):
                assert [g["lr"] for g in resumed.optimizer.param_groups] == [g["lr"] for g in fresh.optimizer.param_groups]
                # Include recursive and group-state cycle schedulers, not just base_lrs.
                resumed.optimizer.step()
                fresh.optimizer.step()
                resumed.scheduler.step()
                fresh.scheduler.step()
        else:
            assert [g["lr"] for g in resumed.optimizer.param_groups] == saved_lrs
            assert resumed.scheduler.last_epoch == 3
            assert any("saved LR" in line and "--moss_ft_resume_new_lr_schedule" in line
                       and "learning_rate" in line for line in lines), lines
    finally:
        resumed.close()
        fresh.close()


@pytest.mark.parametrize("case", sorted(NON_PHASE_SCHEDULERS))
def test_resume_continues_a_non_phase_scheduler(tmp_path, caplog, case):
    """Review fix: the param groups (lr, initial_lr and whatever a scheduler keeps there) are part of the state. A
    LambdaLR's load_state_dict does not touch them, so a resumed run used the freshly built scheduler's step-0 LR (0
    with a warmup), and a recursive or param-group-keeping scheduler continued from the wrong values. 6 windows straight
    against 3 + save + load + 3: every window's LR and the pool bit-identical, no spurious hyperparameter warning."""
    import logging

    if case == "custom_restarts":
        pytest.importorskip("LoraEasyCustomOptimizer.CosineAnnealingWarmRestarts")
    base_path, _ = M.write_base(tmp_path)
    args = _args(moss_ft_lr_phases=None, max_train_steps=6, **NON_PHASE_SCHEDULERS[case])
    torch.manual_seed(7)
    straight = Run(base_path, args)
    try:
        straight.windows(6)
    finally:
        straight.close()
    assert len({tuple(lrs) for lrs in straight.lrs[3:]}) > 1 or case == "constant_with_warmup"

    torch.manual_seed(7)
    first = Run(base_path, args)
    try:
        first.windows(3)
        path = first.save(str(tmp_path / "states"), ft_state.step_state_name("tiny", 3))
    finally:
        first.close()
    _perturb_rng()
    resumed = Run(base_path, args)
    try:
        with caplog.at_level(logging.WARNING, logger="library.moss.ft_state"):
            resumed.load(path)
        assert not [r for r in caplog.records if r.name == "library.moss.ft_state"], caplog.text
        resumed.windows(3)
        assert resumed.lrs == straight.lrs[3:]
        _assert_same(straight, resumed)
    finally:
        resumed.close()


def test_new_lr_schedule_keeps_mode_b_optimizer_state(tmp_path):
    """Mode b loads whole optimizer groups: the saved LR must not overwrite the new scheduler's LR."""
    from transformers.optimization import Adafactor

    base_path, _ = M.write_base(tmp_path)
    args = _args(optimizer_type="Adafactor", fused_backward_pass=True, gradient_accumulation_steps=1,
                 moss_ft_lr_phases=None, max_grad_norm=0.0, moss_ft_resume_new_lr_schedule=True)

    def make(lr):
        model, pool, slots = ft_model.load_moss_for_full_ft(str(base_path), args, "cpu", config=M.TINY)
        assert pool is None
        optimizer = Adafactor(ft_model.param_groups(ft_model.trainable_named(slots)), lr=lr,
                              scale_parameter=False, relative_step=False, warmup_init=False)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
        return model, slots, optimizer, scheduler

    model, slots, optimizer, scheduler = make(1e-3)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    path = ft_state.write_state(str(tmp_path), "mode_b", model=model, slots=slots, pool=None,
                                optimizer=optimizer, lr_scheduler=scheduler, mode=ft_args.ADAFACTOR_FUSED,
                                progress=ft_state.TrainProgress(global_step=1, completed_epochs=1), args=args,
                                fsync=False)
    other, other_slots, other_opt, other_sched = make(5e-4)
    fresh_state = other_sched.state_dict()
    info = ft_state.load_state(path, model=other, slots=other_slots, pool=None, optimizer=other_opt,
                               lr_scheduler=other_sched, mode=ft_args.ADAFACTOR_FUSED, args=args, restore_rng=False)
    assert info.progress.global_step == 1
    assert other_sched.state_dict() == fresh_state
    assert all(g["lr"] == 5e-4 and g["initial_lr"] == 5e-4 for g in other_opt.param_groups)
    for p, q in zip(model.parameters(), other.parameters()):
        assert torch.equal(p, q)
        for key, value in optimizer.state.get(p, {}).items():
            actual = other_opt.state[q][key]
            assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value


def _keep_test_progress(global_step, completed_epochs, epoch_step=0):
    return ft_state.TrainProgress(global_step=global_step, completed_epochs=completed_epochs, epoch_step=epoch_step,
                                  steps_per_epoch=2, gradient_accumulation_steps=GA)


def test_keep_policy_after_resuming_an_earlier_state(tmp_path):
    """Review fix: re-running phase 2 from the pinned phase-1 state into the same folder (design 4.6 / 4.7) must not
    delete the states the new run writes. States numbered past the new run's position belong to the abandoned run:
    they are left alone (not deleted, not counted) until the new run passes them, and the state just written is never
    deleted. The refusal hint names the most recently written complete state."""
    base_path, _ = M.write_base(tmp_path)
    args = _args(save_last_n_epochs_state=1, save_last_n_steps_state=1, moss_ft_lr_phases="2:1e-3,2:4e-4")
    policy = ft_state.keep_policy_from_args(args)
    state_dir = str(tmp_path / "states")

    def names():
        return sorted(os.listdir(state_dir))

    run = Run(base_path, args)
    try:
        run.windows(1)
        # the abandoned run: the pinned end of phase 1 (epoch 2), epoch 4 and step 9
        for name, progress in ((ft_state.epoch_state_name("tiny", 2), _keep_test_progress(4, 2)),
                               (ft_state.epoch_state_name("tiny", 4), _keep_test_progress(8, 4)),
                               (ft_state.step_state_name("tiny", 9), _keep_test_progress(9, 4, 2))):
            ft_state.apply_keep_policy(state_dir, **policy, written=run.save(state_dir, name, progress=progress))
        # the new run, resumed from tiny-000002-state
        written = run.save(state_dir, ft_state.step_state_name("tiny", 5), progress=_keep_test_progress(5, 2, 2))
        assert ft_state.apply_keep_policy(state_dir, **policy, written=written) == []
        assert ft_state.complete_states(state_dir)[0] == written  # the newest = the last written, not step 9
        written = run.save(state_dir, ft_state.epoch_state_name("tiny", 3), progress=_keep_test_progress(6, 3))
        assert ft_state.apply_keep_policy(state_dir, **policy, written=written) == []
        written = run.save(state_dir, ft_state.step_state_name("tiny", 7), progress=_keep_test_progress(7, 3, 2))
        deleted = ft_state.apply_keep_policy(state_dir, **policy, written=written)
        assert [os.path.basename(p) for p in deleted] == ["tiny-step00000005-state"]
        assert names() == ["tiny-000002-state", "tiny-000003-state", "tiny-000004-state", "tiny-step00000007-state",
                           "tiny-step00000009-state"]
        # the new run reaches epoch 4 (the name is replaced atomically) and then passes step 9
        written = run.save(state_dir, ft_state.epoch_state_name("tiny", 4), progress=_keep_test_progress(8, 4))
        deleted = ft_state.apply_keep_policy(state_dir, **policy, written=written)
        assert [os.path.basename(p) for p in deleted] == ["tiny-000003-state"]
        written = run.save(state_dir, ft_state.step_state_name("tiny", 10), progress=_keep_test_progress(10, 5))
        deleted = ft_state.apply_keep_policy(state_dir, **policy, written=written)
        assert sorted(os.path.basename(p) for p in deleted) == ["tiny-step00000007-state", "tiny-step00000009-state"]
        assert names() == ["tiny-000002-state", "tiny-000004-state", "tiny-step00000010-state"]
    finally:
        run.close()


def test_the_state_just_written_is_never_deleted(tmp_path):
    """Even when it ranks below another of its kind (a state written under a lower number than an existing one)."""
    base_path, _ = M.write_base(tmp_path)
    args = _args(save_last_n_steps_state=1)
    state_dir = str(tmp_path / "states")
    run = Run(base_path, args)
    try:
        run.windows(1)
        run.save(state_dir, ft_state.step_state_name("tiny", 2), progress=_keep_test_progress(2, 1))
        written = run.save(state_dir, ft_state.step_state_name("tiny", 1), progress=_keep_test_progress(2, 1))
        policy = ft_state.keep_policy_from_args(args)
        assert written not in ft_state.apply_keep_policy(state_dir, **policy, written=written)
        assert os.path.isdir(written)
    finally:
        run.close()


def test_resume_pointing_at_the_run_folder_names_its_states(tmp_path):
    """Review fix: --resume <output_dir> (the folder holding the states) is not an interrupted state: the refusal says
    so and names the newest complete state inside it."""
    base_path, args, state_dir, older, newer = _two_states(tmp_path)
    with pytest.raises(ft_state.StateError) as error:
        ft_state.verify_state(state_dir)
    message = str(error.value)
    assert "interrupted" not in message and "tiny-step00000002-state" in message, message
    assert "folder of training states" in message
    run = Run(base_path, args)
    try:
        with pytest.raises(ft_state.StateError, match="folder of training states"):
            run.load(state_dir)
    finally:
        run.close()
