from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from datalens.data import download as download_module
from datalens.data.benchmarks import benchmark_label, bird_spec, load_examples, resolve_spec, spider_spec
from datalens.data.download import extract_recursive, normalise_bird_split
from datalens.data.schema import DatabaseSchema, introspect_sqlite, load_schema, quote_ident
from datalens.prompts import PromptConfig, build_prompt, format_answer, prompt_for_example


def test_load_bird_examples(bird_root):
    examples = load_examples(bird_spec(bird_root, "dev"))
    assert [e.id for e in examples] == ["bird-dev-0", "bird-dev-1", "bird-dev-2", "bird-dev-3"]
    first = examples[1]
    assert first.evidence.startswith("Germany")
    assert first.difficulty == "simple"
    assert first.db_path.is_file()


def test_load_spider_format(tmp_path, shop_db):
    root = tmp_path / "spider"
    (root / "database" / "shop").mkdir(parents=True)
    (root / "database" / "shop" / "shop.sqlite").write_bytes(shop_db.read_bytes())
    (root / "dev.json").write_text(
        json.dumps([{"db_id": "shop", "question": "How many?", "query": "SELECT COUNT(*) FROM orders"}]),
        encoding="utf-8",
    )
    [example] = load_examples(spider_spec(root))
    assert example.id == "spider-dev-0"
    assert example.gold_sql == "SELECT COUNT(*) FROM orders"
    assert example.evidence == "" and example.difficulty is None


def test_resolve_spec_and_missing_file(tmp_path):
    assert resolve_spec("bird-train", tmp_path).name == "bird-train"
    with pytest.raises(ValueError):
        resolve_spec("wikisql", tmp_path)
    with pytest.raises(FileNotFoundError, match="datalens data download"):
        load_examples(bird_spec(tmp_path, "dev"))


@pytest.mark.parametrize(
    ("benchmark", "label"),
    [("bird-dev", "BIRD dev"), ("bird-train", "BIRD train"), ("spider-dev", "Spider dev")],
)
def test_benchmark_label(benchmark, label):
    assert benchmark_label(benchmark) == label


def test_introspection_reads_keys_examples_and_descriptions(shop_db):
    schema = introspect_sqlite(shop_db)
    assert schema.table_names() == ["customers", "products", "orders"]
    customers = schema.tables[0]
    assert customers.columns[0].primary_key
    country = customers.columns[2]
    assert country.examples == ["DE", "US"]
    assert country.description == "ISO country code of the customer"
    assert customers.columns[0].description is None  # restates the name, dropped
    orders = schema.tables[2]
    assert {(fk.column, fk.ref_table) for fk in orders.foreign_keys} == {
        ("customer_id", "customers"),
        ("product_id", "products"),
    }


def test_render_is_valid_ddl(shop_db, tmp_path):
    import sqlite3

    schema = introspect_sqlite(shop_db)
    text = schema.render(descriptions=True)
    assert "`order date` TEXT" in text
    assert '-- example: ["DE", "US"]' in text or '"DE"' in text
    assert "values: DE: Germany" in text
    # Comments and quoting must still parse as SQL.
    sqlite3.connect(tmp_path / "copy.sqlite").executescript(text.replace("PRIMARY KEY", "UNIQUE"))


def test_schema_cache_roundtrip(shop_db, tmp_path):
    first = load_schema(shop_db, cache_dir=tmp_path / "cache")
    assert list((tmp_path / "cache").glob("*.json"))
    second = load_schema(shop_db, cache_dir=tmp_path / "cache")
    assert second.render() == first.render()
    assert DatabaseSchema.from_json(first.to_json()).render() == first.render()


def test_quote_ident():
    assert quote_ident("name") == "name"
    assert quote_ident("orders") == "orders"
    assert quote_ident("order") == "`order`"
    assert quote_ident("Group") == "`Group`"
    assert quote_ident("order date") == "`order date`"
    assert quote_ident("we`ird") == "`we``ird`"


def test_introspection_handles_keyword_names(tmp_path):
    """BIRD's financial database has a table called ``order``."""
    import sqlite3

    db = tmp_path / "financial.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE account (account_id INTEGER PRIMARY KEY, "group" TEXT);
        CREATE TABLE "order" (
            order_id INTEGER PRIMARY KEY,
            account_id INTEGER REFERENCES account (account_id),
            "Free Meal Count (K-12)" REAL
        );
        INSERT INTO account VALUES (1, 'A');
        INSERT INTO "order" VALUES (10, 1, 2.5);
        """
    )
    conn.commit()
    conn.close()

    schema = introspect_sqlite(db)
    assert schema.table_names() == ["account", "order"]
    order = schema.tables[1]
    assert [c.examples for c in order.columns] == [[10], [1], [2.5]]
    assert order.foreign_keys[0].ref_table == "account"
    text = schema.render()
    assert "CREATE TABLE `order` (" in text and "`group` TEXT" in text
    sqlite3.connect(tmp_path / "copy.sqlite").executescript(text)


def test_prompt_parts_join_to_the_full_prompt(bird_root):
    example = load_examples(bird_spec(bird_root, "dev"))[1]
    parts = prompt_for_example(example, PromptConfig(reasoning=True))
    assert parts.user == parts.schema_block + parts.question_block
    assert "CREATE TABLE customers" in parts.schema_block
    assert "External knowledge: Germany refers to country = 'DE'" in parts.question_block
    assert "<think>" in parts.question_block
    messages = parts.messages()
    assert [m["role"] for m in messages] == ["system", "user"]
    direct = build_prompt("CREATE TABLE t (a INT);", "q?", reasoning=False)
    assert "<think>" not in direct.user and "External knowledge: None" in direct.user


def test_format_answer():
    assert format_answer("SELECT 1;") == "```sql\nSELECT 1;\n```"
    assert format_answer("SELECT 1", reasoning="because").startswith("<think>\nbecause\n</think>")


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def test_download_normalises_nested_archives(tmp_path, shop_db):
    inner = _zip_bytes({"dev_databases/shop/shop.sqlite": shop_db.read_bytes()})
    outer = _zip_bytes(
        {
            "dev_20240627/dev.json": json.dumps([]).encode(),
            "dev_20240627/dev_tied_append.json": b"[]",
            "dev_20240627/dev_databases.zip": inner,
            "__MACOSX/dev_20240627/._dev.json": b"junk",
        }
    )
    archive = tmp_path / "dev.zip"
    archive.write_bytes(outer)
    extracted = tmp_path / "x"
    extracted.mkdir()
    extract_recursive(archive, extracted)
    out = normalise_bird_split(extracted, "dev", tmp_path / "bird")
    assert (out / "dev.json").is_file()
    assert (out / "dev_tied_append.json").is_file()
    assert (out / "dev_databases" / "shop" / "shop.sqlite").is_file()


def test_download_bird_skips_ready_splits_and_removes_archives(tmp_path, shop_db, monkeypatch):
    fetched = []

    def fake_download(url, dest):
        fetched.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        files = {"dev/dev.json": b"[]", "dev/dev_databases/shop/shop.sqlite": shop_db.read_bytes()}
        dest.write_bytes(_zip_bytes(files))
        return dest

    monkeypatch.setattr(download_module, "download", fake_download)
    root = tmp_path / "bird"
    assert download_module.download_bird(root, ["dev"]) == [root / "dev"]
    assert (root / "dev" / "dev_databases" / "shop" / "shop.sqlite").is_file()
    assert not (root / "_downloads" / "dev.zip").exists()
    download_module.download_bird(root, ["dev"])
    assert len(fetched) == 1


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    archive.write_bytes(_zip_bytes({"../escape.txt": b"x"}))
    target = tmp_path / "t"
    target.mkdir()
    with pytest.raises(ValueError, match="unsafe"):
        extract_recursive(archive, target)
    assert not Path(tmp_path / "escape.txt").exists()
