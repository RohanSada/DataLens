"""Create a small SQLite database so the API has something to query out of the box.

Usage: python scripts/make_demo_db.py <db_dir>
Writes <db_dir>/music_store/music_store.sqlite with deterministic synthetic data.
"""

from __future__ import annotations

import random
import sqlite3
import sys
from pathlib import Path

ARTISTS = ["Nina Simone", "Radiohead", "Daft Punk", "Miles Davis", "Björk", "Kendrick Lamar", "Fleetwood Mac"]
GENRES = ["Jazz", "Rock", "Electronic", "Hip Hop", "Pop"]
COUNTRIES = ["USA", "Germany", "France", "Brazil", "India", "Japan", "Canada"]
FIRST = ["Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Goran", "Hana", "Ivan", "Jules", "Kofi", "Lena"]
LAST = ["Silva", "Okafor", "Müller", "Tanaka", "Dubois", "Patel", "Kim", "Novak", "Haddad", "Smith"]


def build(path: Path, seed: int = 7) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE artists (artist_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE genres (genre_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE albums (
            album_id INTEGER PRIMARY KEY, title TEXT NOT NULL, release_year INTEGER,
            artist_id INTEGER NOT NULL REFERENCES artists(artist_id)
        );
        CREATE TABLE tracks (
            track_id INTEGER PRIMARY KEY, name TEXT NOT NULL, milliseconds INTEGER, unit_price REAL,
            album_id INTEGER REFERENCES albums(album_id), genre_id INTEGER REFERENCES genres(genre_id)
        );
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, country TEXT, signup_date TEXT
        );
        CREATE TABLE invoices (
            invoice_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
            invoice_date TEXT, total REAL
        );
        CREATE TABLE invoice_items (
            item_id INTEGER PRIMARY KEY, invoice_id INTEGER REFERENCES invoices(invoice_id),
            track_id INTEGER REFERENCES tracks(track_id), quantity INTEGER, unit_price REAL
        );
        """
    )
    conn.executemany("INSERT INTO artists VALUES (?, ?)", enumerate(ARTISTS, 1))
    conn.executemany("INSERT INTO genres VALUES (?, ?)", enumerate(GENRES, 1))

    album_id = track_id = 0
    for artist_id, artist in enumerate(ARTISTS, 1):
        genre_id = rng.randint(1, len(GENRES))
        for n in range(rng.randint(2, 4)):
            album_id += 1
            conn.execute(
                "INSERT INTO albums VALUES (?, ?, ?, ?)",
                (album_id, f"{artist.split()[0]} Sessions Vol. {n + 1}", rng.randint(1965, 2024), artist_id),
            )
            for t in range(rng.randint(6, 12)):
                track_id += 1
                conn.execute(
                    "INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        track_id,
                        f"Track {t + 1}",
                        rng.randint(120_000, 420_000),
                        rng.choice([0.99, 1.29]),
                        album_id,
                        genre_id,
                    ),
                )

    for customer_id in range(1, 61):
        conn.execute(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
            (
                customer_id,
                rng.choice(FIRST),
                rng.choice(LAST),
                rng.choice(COUNTRIES),
                f"20{rng.randint(19, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            ),
        )

    item_id = 0
    for invoice_id in range(1, 301):
        items = []
        for _ in range(rng.randint(1, 5)):
            item_id += 1
            track = rng.randint(1, track_id)
            price = conn.execute("SELECT unit_price FROM tracks WHERE track_id = ?", (track,)).fetchone()[0]
            items.append((item_id, invoice_id, track, rng.randint(1, 2), price))
        total = round(sum(q * p for *_, q, p in items), 2)
        date = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        customer = rng.randint(1, 60)
        conn.execute("INSERT INTO invoices VALUES (?, ?, ?, ?)", (invoice_id, customer, date, total))
        conn.executemany("INSERT INTO invoice_items VALUES (?, ?, ?, ?, ?)", items)
    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/databases")
    print(build(root / "music_store" / "music_store.sqlite"))
