"""CPU tests: the git dependencies in requirements.txt are pinned to commits, backend/installer.py installs the same
commits, and the git packages installed in this venv are those commits.

Run: CUDA_VISIBLE_DEVICES=-1 venv/Scripts/python.exe -m pytest tests/test_requirements_pins.py
"""
import json
import os
import re
from importlib import metadata

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(HERE, "..", "requirements.txt")
INSTALLER = os.path.join(HERE, "..", "..", "installer.py")  # backend/installer.py: sd_scripts is its submodule
GIT_URL = re.compile(r"git\+(https://[^\s@\"'#]+)(?:@([^\s\"'#]+))?")
SHA = re.compile(r"[0-9a-f]{40}")


def _repo(url):
    url = url.rstrip("/").lower()
    return url[:-4] if url.endswith(".git") else url


def git_refs(path):
    """{repo url: ref, or None for the default branch} of every git+ URL outside comments."""
    refs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for url, ref in GIT_URL.findall(line.split("#", 1)[0]):
                refs[_repo(url)] = ref or None
    return refs


def test_git_requirements_are_pinned_to_commits():
    refs = git_refs(REQUIREMENTS)
    assert refs and all(SHA.fullmatch(ref or "") for ref in refs.values()), refs


def test_installer_installs_the_same_commits():
    if not os.path.isfile(INSTALLER):
        pytest.skip("installer.py lives in the LoRA_Easy_Training_Scripts backend checkout")
    pinned, installed = git_refs(REQUIREMENTS), git_refs(INSTALLER)
    assert installed and all(SHA.fullmatch(ref or "") for ref in installed.values()), installed
    assert {url: pinned.get(url) for url in installed} == installed, (installed, pinned)


def test_installed_git_packages_are_the_pinned_commits():
    pinned = git_refs(REQUIREMENTS)
    seen = {}
    for dist in metadata.distributions():
        text = dist.read_text("direct_url.json")
        info = json.loads(text) if text else {}
        if "vcs_info" in info and _repo(info["url"]) in pinned:
            seen[_repo(info["url"])] = info["vcs_info"].get("commit_id")
    if not seen:
        pytest.skip("no git dependency of requirements.txt is installed from its git URL here")
    assert seen == {url: pinned[url] for url in seen}, (seen, pinned)
