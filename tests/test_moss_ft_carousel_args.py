"""CPU tests of the Carousel arguments (library/moss/ft_args.py; docs/moss_ft_carousel_plan.md 6, items 8-9): the
optimizer kinds, every refusal and ignore of plan 4.3, CAME's bias correction default, and that every flag, log key
and user-facing string derives from the one name constant (ft_budget.CAROUSEL).

Run: CUDA_VISIBLE_DEVICES=-1 venv_t214/Scripts/python.exe -B -m pytest tests/test_moss_ft_carousel_args.py -q -p no:cacheprovider
"""
import ast
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest

from library.moss import ft_args, ft_budget
from test_moss_ft_args import logs, make_args  # noqa: F401 - the logs fixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = ft_args.carousel_flag


@pytest.fixture(scope="module")
def parser():
    return ft_args.setup_parser()


def carousel(parser, *argv, **overrides):
    return make_args(parser, F(), *argv, **overrides)


@pytest.mark.parametrize("argv,kind", [
    ((), "adamw"), (("--optimizer_type", "adamw"), "adamw"), (("--optimizer_type", "AdamW8bit"), "adamw8bit"),
    (("--optimizer_type", "bitsandbytes.optim.AdamW8bit"), "adamw8bit"), (("--use_8bit_adam",), "adamw8bit"),
    (("--optimizer_type", "CAME"), "came"), (("--optimizer_type", "LoraEasyCustomOptimizer.came.CAME"), "came"),
])
def test_optimizer_kinds(parser, argv, kind):
    args = carousel(parser, *argv)
    assert ft_args.optimizer_mode(args) == ft_args.CAROUSEL and ft_args.carousel_kind(args) == kind


def test_optimizer_args_per_kind(parser):
    came = ft_args.parse_optimizer_args(carousel(parser, "--optimizer_type", "CAME"), ft_args.CAROUSEL, quiet=True)
    assert came["bias_correction"] is True and came["betas"] == (0.9, 0.999, 0.99)
    explicit = carousel(parser, "--optimizer_type", "CAME", "--optimizer_args", "bias_correction=False",
                        "betas=(0.9,0.999,0.99)")
    assert ft_args.parse_optimizer_args(explicit, ft_args.CAROUSEL, quiet=True)["bias_correction"] is False
    adamw = ft_args.parse_optimizer_args(carousel(parser, "--optimizer_args", "weight_decay=0.01"), ft_args.CAROUSEL)
    assert adamw == {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}


REFUSED = [
    (("--optimizer_type", "PagedAdamW8bit"), {}, "paged optimizers are refused"),
    (("--optimizer_type", "Adafactor", "--fused_backward_pass"), {}, "fused_backward_pass is not used"),
    (("--optimizer_type", "Prodigy"), {}, "not supported with"),
    (("--use_lion_optimizer",), {}, "Lion"),
    (("--optimizer_type", "CAME", "--optimizer_args", "bias_correction=False", "betas=(0.9,0.999,0.9999)"), {},
     "needs bias_correction=True"),
    (("--optimizer_type", "CAME", "--optimizer_args", "ams_bound=True"), {}, "ams_bound"),
    (("--optimizer_type", "AdamW8bit", "--optimizer_args", "is_paged=True"), {"cuda": True}, "is_paged"),
    (("--optimizer_type", "AdamW8bit"), {}, "needs CUDA"),
    ((F("host_pinned"), "masters"), {}, "needs CUDA"),
    ((F("active"), "0"), {}, "active"),
    ((F("active"), "45"), {}, "active"),
    ((F("active"), "4", F("vram_gib"), "20"), {}, "choose one"),
    ((F("vram_gib"), "-1"), {}, "vram_gib"),
    ((F("period"), "0"), {}, "period"),
    ((F("state_dtype"), "bf16"), {}, "came only"),
    (("--optimizer_type", "CAME", F("state_dtype"), "bf16", F("state"), "reset"), {}, "keep"),
    (("--moss_ft_freeze", r"layers\..*,text_embedder\.blocks\..*"), {}, "no block trainable"),
]


@pytest.mark.parametrize("argv,kw,match", REFUSED)
def test_refusals(parser, argv, kw, match):
    args = carousel(parser, *argv)
    with pytest.raises(ValueError, match=match):
        ft_args.validate_moss_ft_args(args, cuda=kw.get("cuda", False))


@pytest.mark.parametrize("flag,value", [("active", "4"), ("period", "10"), ("select", "window"), ("always", "head"),
                                        ("state", "reset"), ("host_pinned", "masters"), ("vram_gib", "20")])
def test_carousel_flags_need_the_mode(parser, flag, value):
    args = make_args(parser, F(flag), value)
    with pytest.raises(ValueError, match="only used with"):
        ft_args.validate_moss_ft_args(args, cuda=False)


def test_ignored_options_are_reported(parser, logs):  # noqa: F811 - the fixture
    args = carousel(parser, "--moss_ft_clip_mode", "exact", "--max_grad_norm", "1.0", "--moss_ft_host_threads", "4",
                    "--mixed_precision", "fp16")
    ft_args.validate_moss_ft_args(args, cuda=False)
    text = "\n".join(logs["info"])
    assert "--moss_ft_clip_mode exact is not used" in text and "--moss_ft_host_threads is not used" in text
    assert "--moss_ft_overflow_policy partial is not used" in text
    assert ft_args.optimizer_mode(args) == ft_args.CAROUSEL


def test_every_name_derives_from_the_constant(monkeypatch):
    """A rename is ft_budget.CAROUSEL alone: with another value every flag and dest follows; no module hard-codes the
    name in a flag, a log key or a message (string constants outside docstrings / comments)."""
    for module in (ft_budget, ft_args):
        monkeypatch.setattr(module, "CAROUSEL", "spin")
    parser = ft_args.setup_parser()
    dests = {action.dest for action in parser._actions}
    assert {"moss_spin", "moss_spin_active", "moss_spin_period", "moss_spin_always", "moss_spin_state"} <= dests
    assert not any("carousel" in dest for dest in dests)
    args = parser.parse_args(["--moss_spin", "--moss_spin_active", "3", "--optimizer_type", "CAME"])
    assert ft_args.carousel_enabled(args) and ft_args.carousel_arg(args, "active") == 3
    assert ft_args.optimizer_mode(args) == "spin"
    monkeypatch.undo()

    files = [os.path.join(ROOT, "library", "moss", name) for name in os.listdir(os.path.join(ROOT, "library", "moss"))
             if name.startswith("ft_") and name.endswith(".py")] + [os.path.join(ROOT, "moss_train.py")]
    found = []
    for path in files:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docstrings.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                text = node.value
                if text.isidentifier() and text.startswith("Carousel"):
                    continue  # a class name in an annotation (internal, not user-facing)
                if "moss_carousel" in text or "carousel/" in text or text == "carousel" or "Carousel" in text:
                    found.append((os.path.basename(path), node.lineno, text[:60]))
    assert found == [("ft_budget.py", found[0][1], "carousel")] if found else True, found


def guide_example_toml() -> str:
    """The "# carousel example" TOML block of docs/moss_train.md section 9."""
    with open(os.path.join(ROOT, "docs", "moss_train.md"), encoding="utf-8") as f:
        text = f.read()
    start = text.index("```toml\n# carousel example")
    return text[start + len("```toml\n"):text.index("```", start + 7)]


def test_the_guide_example_config_reads_and_validates(parser, tmp_path, monkeypatch, logs):  # noqa: F811 - fixture
    import sys

    from library import train_util

    config = tmp_path / "carousel.toml"
    config.write_text(guide_example_toml(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["moss_train.py", "--config_file", str(config)])
    args = train_util.read_config_from_file(parser.parse_args(), parser, unknown_keys="error")
    ft_args.validate_moss_ft_args(args, cuda=True)
    assert ft_args.optimizer_mode(args) == ft_args.CAROUSEL and ft_args.carousel_kind(args) == "came"
    kwargs = ft_args.parse_optimizer_args(args, ft_args.CAROUSEL, quiet=True)
    assert kwargs["bias_correction"] is True and kwargs["betas"] == (0.9, 0.999, 0.9999)
    assert ft_args.carousel_arg(args, "period") == 50 and ft_args.carousel_arg(args, "always") == "small"
    assert not logs["warning"], logs["warning"]
