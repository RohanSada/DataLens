import sqlite3

from app.services.schema_extractor import derive_db_id, extract_bird_schema


def test_derive_db_id_from_explicit():
    assert derive_db_id("/tmp/foo.sqlite", "custom") == "custom"


def test_extract_bird_schema_shape():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
    conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, FOREIGN KEY(user_id) REFERENCES users(id))")
    conn.commit()

    schema = extract_bird_schema(conn, "demo")

    assert schema["db_id"] == "demo"
    assert "users" in schema["table_names_original"]
    assert "orders" in schema["table_names_original"]
    assert schema["column_types"]
    conn.close()
