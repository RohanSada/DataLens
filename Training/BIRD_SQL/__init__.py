"""BIRD Text-to-SQL training-data builder.

This package converts a BIRD-format question file (``question`` + ``evidence`` +
gold ``SQL`` + ``db_id``) into JSONL ``{prompt, completion}`` records for
supervised fine-tuning / QLoRA of Qwen2.5-Coder.

The schema-context machinery is reused from the sibling ``schema_retrieval``
package. That package uses *flat* imports (``from config import ...``,
``import schema_loader``) and ships no ``__init__.py``, so it cannot be imported
as ``schema_retrieval.x``. To make its modules importable from here we prepend
its directory to ``sys.path`` at package import time. Because Python imports the
parent package before any submodule, this runs before
``schema_context``/``gold_sql`` attempt to import ``schema_loader`` etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Training/BIRD_SQL/__init__.py -> Training/BIRD_SQL -> Training -> <project root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCHEMA_RETRIEVAL_DIR = _PROJECT_ROOT / "schema_retrieval"


def _ensure_schema_retrieval_on_path() -> None:
    path = str(_SCHEMA_RETRIEVAL_DIR)
    if _SCHEMA_RETRIEVAL_DIR.is_dir() and path not in sys.path:
        # Append (not insert at 0) so we don't shadow this package's own
        # modules; ``schema_retrieval`` module names (schema_loader, formatters,
        # graph_store, config, ...) don't collide with ours by accident because
        # ours are imported via the ``Training.BIRD_SQL`` package namespace.
        sys.path.append(path)


_ensure_schema_retrieval_on_path()

PROJECT_ROOT = _PROJECT_ROOT
SCHEMA_RETRIEVAL_DIR = _SCHEMA_RETRIEVAL_DIR

__all__ = ["PROJECT_ROOT", "SCHEMA_RETRIEVAL_DIR"]
