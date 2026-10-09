"""Shared fixtures: a tiny shop database laid out like BIRD."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

QUESTIONS = [
    {
        "question_id": 0,
        "db_id": "shop",
        "question": "How many customers are there?",
        "evidence": "",
        "SQL": "SELECT COUNT(*) FROM customers",
        "difficulty": "simple",
    },
    {
        "question_id": 1,
        "db_id": "shop",
        "question": "List the names of customers from Germany.",
        "evidence": "Germany refers to country = 'DE'",
        "SQL": "SELECT name FROM customers WHERE country = 'DE'",
        "difficulty": "simple",
    },
    {
        "question_id": 2,
        "db_id": "shop",
        "question": "What is the total revenue of orders for the product named Lamp?",
        "evidence": "revenue = quantity * price",
        "SQL": (
            "SELECT SUM(o.quantity * p.price) FROM orders AS o "
            "JOIN products AS p ON o.product_id = p.id WHERE p.name = 'Lamp'"
        ),
        "difficulty": "moderate",
    },
    {
        "question_id": 3,
        "db_id": "shop",
        "question": "Which customer placed the most orders?",
        "evidence": "",
        "SQL": (
            "SELECT c.name FROM customers AS c JOIN orders AS o ON o.customer_id = c.id "
            "GROUP BY c.id ORDER BY COUNT(*) DESC LIMIT 1"
        ),
        "difficulty": "challenging",
    },
]


def make_shop_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT NOT NULL, country TEXT);
        CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customers(id),
            product_id INTEGER REFERENCES products(id),
            quantity INTEGER,
            `order date` TEXT
        );
        INSERT INTO customers VALUES (1, 'Ada', 'DE'), (2, 'Grace', 'US'), (3, 'Linus', 'DE');
        INSERT INTO products VALUES (1, 'Lamp', 20.0), (2, 'Desk', 150.0);
        INSERT INTO orders VALUES
            (1, 1, 1, 2, '2024-01-01'),
            (2, 1, 2, 1, '2024-01-02'),
            (3, 2, 1, 1, '2024-02-01'),
            (4, 1, 1, 3, '2024-03-01');
        """
    )
    conn.commit()
    conn.close()
    desc = path.parent / "database_description"
    desc.mkdir(exist_ok=True)
    (desc / "customers.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "id,id,id,integer,\n"
        "country,country,ISO country code of the customer,text,DE: Germany; US: United States\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def shop_db(tmp_path: Path) -> Path:
    return make_shop_db(tmp_path / "shop" / "shop.sqlite")


@pytest.fixture
def bird_root(tmp_path: Path) -> Path:
    """``tmp/bird`` with a ``dev`` split holding the shop database."""
    root = tmp_path / "bird"
    make_shop_db(root / "dev" / "dev_databases" / "shop" / "shop.sqlite")
    (root / "dev" / "dev.json").write_text(json.dumps(QUESTIONS), encoding="utf-8")
    return root
