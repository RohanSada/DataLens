"""Download BIRD and normalise it into the layout :func:`bird_spec` expects.

The official archives nest zips inside zips (``dev.zip`` contains
``dev_databases.zip``) and their top-level folder names change between releases,
so instead of hard-coding paths this module extracts everything and then looks
for ``<split>.json`` and the folder of ``<db>/<db>.sqlite`` files.
"""

from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BIRD_URLS = {
    "train": "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip",
    "dev": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
}


def download(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (skipped if it already exists)."""
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, tmp.open("wb") as fh:
        shutil.copyfileobj(response, fh, length=1 << 20)
    tmp.rename(dest)
    return dest


def _safe_extract(archive: Path, target: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            resolved = (target / member.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ValueError(f"unsafe path in {archive.name}: {member.filename}")
        zf.extractall(target)


def extract_recursive(archive: Path, target: Path) -> None:
    """Extract ``archive`` into ``target``, then any zips found inside it."""
    _safe_extract(archive, target)
    while True:
        nested = [p for p in target.rglob("*.zip") if "__MACOSX" not in p.parts]
        if not nested:
            return
        for inner in nested:
            _safe_extract(inner, inner.parent)
            inner.unlink()


def normalise_bird_split(extracted: Path, split: str, dest_root: Path) -> Path:
    """Move ``<split>.json`` and its databases into ``dest_root/<split>/``."""
    questions = [p for p in extracted.rglob(f"{split}.json") if "__MACOSX" not in p.parts]
    if not questions:
        raise FileNotFoundError(f"no {split}.json inside the extracted archive")

    sqlite_files = [p for p in extracted.rglob("*.sqlite") if "__MACOSX" not in p.parts]
    db_dirs = {p.parent.parent for p in sqlite_files if p.stem == p.parent.name}
    if len(db_dirs) != 1:
        raise FileNotFoundError(
            f"expected one folder of <db>/<db>.sqlite databases, found {sorted(map(str, db_dirs))}"
        )

    out = dest_root / split
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(questions[0], out / f"{split}.json")
    for extra in questions[0].parent.glob(f"{split}*"):
        if extra.is_file() and extra.suffix in {".json", ".sql"}:
            shutil.copy2(extra, out / extra.name)
    db_out = out / f"{split}_databases"
    if db_out.exists():
        shutil.rmtree(db_out)
    shutil.move(str(db_dirs.pop()), db_out)
    return out


def is_ready(dest_root: Path, split: str) -> bool:
    out = dest_root / split
    return (out / f"{split}.json").is_file() and (out / f"{split}_databases").is_dir()


def download_bird(
    dest_root: str | Path, splits: list[str], *, force: bool = False, keep_archives: bool = False
) -> list[Path]:
    """Download and normalise the given BIRD splits under ``dest_root``.

    Splits that are already in place are skipped unless ``force``. Archives are
    deleted once extracted (train is tens of GB) unless ``keep_archives``.
    """
    dest_root = Path(dest_root)
    outputs = []
    for split in splits:
        if split not in BIRD_URLS:
            raise ValueError(f"unknown BIRD split {split!r}; choose from {sorted(BIRD_URLS)}")
        if is_ready(dest_root, split) and not force:
            outputs.append(dest_root / split)
            continue
        archive = download(BIRD_URLS[split], dest_root / "_downloads" / f"{split}.zip")
        with tempfile.TemporaryDirectory(dir=dest_root) as tmp:
            extract_recursive(archive, Path(tmp))
            outputs.append(normalise_bird_split(Path(tmp), split, dest_root))
        if not keep_archives:
            archive.unlink()
    return outputs
