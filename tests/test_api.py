from io import BytesIO


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "version" in response.json()


def test_connect_sqlite_via_upload(client, sqlite_db):
    with sqlite_db.open("rb") as f:
        upload = client.post(
            "/upload",
            files={"file": ("sample.sqlite", f, "application/octet-stream")},
        )
    assert upload.status_code == 200
    file_id = upload.json()["file_id"]

    connect = client.post(
        "/connect",
        json={"db_type": "sqlite", "file_id": file_id, "db_id": "sample"},
    )
    assert connect.status_code == 200
    body = connect.json()
    assert body["schema_ready"] is True
    assert body["table_count"] >= 1
    session_id = body["session_id"]

    query = client.post(
        "/query",
        json={"session_id": session_id, "question": "How many users?"},
    )
    assert query.status_code == 200
    assert "generated_sql" in query.json()

    schema = client.get(f"/schema/{session_id}")
    assert schema.status_code == 200
    assert schema.json()["db_id"] == "sample"

    disconnect = client.post("/disconnect", json={"session_id": session_id})
    assert disconnect.status_code == 200


def test_rejects_raw_db_path(client):
    response = client.post(
        "/connect",
        json={"db_type": "sqlite", "db_path": "/etc/passwd"},
    )
    assert response.status_code == 400


def test_upload_rejects_invalid_extension(client):
    response = client.post(
        "/upload",
        files={"file": ("bad.txt", BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400
