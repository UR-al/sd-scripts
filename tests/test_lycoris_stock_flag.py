"""The LyCORIS fixed paths are the default; --lycoris_stock (config lycoris_stock = true) gives LyCORIS as installed.

User decision 2026-10-24 ("flag ON -> original"): for network_module=lycoris.kohya the fixes (library/lycoris_algo_fixes.py)
run without any flag; --lycoris_stock turns them off (the module is not even imported); --lycoris_algo_fixes stays
accepted for old configs and changes nothing; with both, --lycoris_stock wins. Non-LyCORIS networks are never touched.

Each case runs NetworkTrainer._lycoris_fix_setup in a fresh interpreter, so "not imported / nothing patched" is real.
Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -B -m pytest tests/test_lycoris_stock_flag.py
"""
import json
import os
import subprocess
import sys

import pytest

SD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = r"""
import json, logging, sys, types
sys.path.insert(0, {sd!r})
records = []

class _H(logging.Handler):
    def emit(self, r):
        records.append(r.getMessage())

import train_network
train_network.logger.addHandler(_H())
args = types.SimpleNamespace(**json.loads({args!r}))
fast = {fast!r}
if fast:  # what prepare_anima_runtime / prepare_moss_runtime do before train() builds the network
    from library import lycoris_fast
    lycoris_fast.install_fast_lokr_forward()
trainer = types.SimpleNamespace(_lycoris_fixes=None)
on = train_network.NetworkTrainer._lycoris_fix_setup(trainer, args)
from lycoris.modules.loha import LohaModule
from lycoris.modules.lokr import LokrModule
patched = bool(getattr(LohaModule.__dict__["get_diff_weight"], "_lycoris_algo_fix", False))
lokr = getattr(LokrModule.__dict__["forward"], "__name__", "")
lokr_inner = ""
if "library.lycoris_algo_fixes" in sys.modules:
    inner = sys.modules["library.lycoris_algo_fixes"]._LOKR_STOCK["forward"]  # what _lokr_forward calls at multiplier 1
    lokr_inner = getattr(inner, "__name__", "")
print("RESULT" + json.dumps({{"on": on, "imported": "library.lycoris_algo_fixes" in sys.modules, "patched": patched,
                             "lokr_forward": lokr, "lokr_inner": lokr_inner, "logs": records}}))
"""


def _setup(fast=False, **args):
    code = _PROBE.format(sd=SD, args=json.dumps(args), fast=fast)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=SD, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=600)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT")]
    assert line, out.stdout[-2000:] + out.stderr[-4000:]
    return json.loads(line[-1][len("RESULT"):])


def test_default_is_on_for_lycoris():
    r = _setup(network_module="lycoris.kohya")
    assert r["on"] and r["imported"] and r["patched"]
    assert not any("--lycoris_algo_fixes" in m for m in r["logs"])


def test_lycoris_stock_imports_and_patches_nothing():
    r = _setup(network_module="lycoris.kohya", lycoris_stock=True)
    assert not r["on"] and not r["imported"] and not r["patched"]
    assert any("--lycoris_stock" in m for m in r["logs"])


def test_old_flag_is_accepted_and_changes_nothing():
    r = _setup(network_module="lycoris.kohya", lycoris_algo_fixes=True)
    assert r["on"] and r["imported"] and r["patched"]
    assert sum("--lycoris_algo_fixes is the default now" in m for m in r["logs"]) == 1


def test_stock_wins_over_the_old_flag():
    r = _setup(network_module="lycoris.kohya", lycoris_stock=True, lycoris_algo_fixes=True)
    assert not r["on"] and not r["imported"] and not r["patched"]
    assert any("both given" in m for m in r["logs"])


def test_production_order_keeps_the_fast_lokr_forward():
    """Anima / MOSS install library/lycoris_fast.py's LoKr+DoRA forward before train() applies the fixed paths: the
    fixed LoKr forward (multiplier != 1 fix) wraps it and hands every multiplier-1 call to it; --lycoris_stock leaves
    the fast forward itself in place and imports nothing."""
    r = _setup(fast=True, network_module="lycoris.kohya")
    assert r["on"] and r["lokr_forward"] == "_lokr_forward" and r["lokr_inner"] == "_fast_forward", r
    r = _setup(fast=True, network_module="lycoris.kohya", lycoris_stock=True)
    assert not r["on"] and not r["imported"] and r["lokr_forward"] == "_fast_forward", r


@pytest.mark.parametrize("flags", [{}, {"lycoris_algo_fixes": True}, {"lycoris_stock": True}])
def test_non_lycoris_networks_untouched(flags):
    r = _setup(network_module="networks.lora", **flags)
    assert not r["on"] and not r["imported"] and not r["patched"]


def test_parser_and_config_key(tmp_path):
    sys.path.insert(0, SD)
    import train_network
    from library import train_util

    parser = train_network.setup_parser()
    a = parser.parse_args([])
    assert a.lycoris_stock is False and a.lycoris_algo_fixes is False
    a = parser.parse_args(["--lycoris_stock", "--lycoris_algo_fixes"])
    assert a.lycoris_stock is True and a.lycoris_algo_fixes is True
    cfg = tmp_path / "c.toml"
    cfg.write_text('network_module = "lycoris.kohya"\nlycoris_stock = true\n', encoding="utf-8")
    argv = ["train_network.py", "--config_file", str(cfg)]
    saved = sys.argv
    sys.argv = argv  # read_config_from_file re-parses sys.argv
    try:
        a = train_util.read_config_from_file(parser.parse_args(argv[1:]), parser)
    finally:
        sys.argv = saved
    assert a.lycoris_stock is True and a.network_module == "lycoris.kohya"
