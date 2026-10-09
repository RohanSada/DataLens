"""YAML config loading and command-line overrides (``--set trainer.learning_rate=2e-5``)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml


class _Loader(yaml.SafeLoader):
    """``SafeLoader`` that also reads ``2e-5`` as a float.

    PyYAML implements YAML 1.1, where a float needs a decimal point, so ``2e-5``
    would load as the string ``"2e-5"`` and reach the optimizer as one.
    """


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Loader)


def load_yaml(path: str | Path) -> dict[str, Any]:
    raw = parse_yaml(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return raw


def apply_overrides(raw: dict[str, Any], assignments: Sequence[str]) -> dict[str, Any]:
    """Set dotted keys in a nested dict. Values are parsed as YAML (numbers, bools, null, lists)."""
    for assignment in assignments:
        key, sep, value = assignment.partition("=")
        if not sep or not key:
            raise ValueError(f"override {assignment!r} must look like key.path=value")
        node = raw
        *parents, leaf = key.strip().split(".")
        for part in parents:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            if not isinstance(child, dict):
                raise ValueError(f"cannot set {key!r}: {part!r} is not a mapping")
            node = child
        node[leaf] = parse_yaml(value)
    return raw
