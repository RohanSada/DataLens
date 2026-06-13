"""
DataLens Streamlit frontend (development / internal use only).

For production SaaS, use the web/ Next.js frontend.

Run the API first:
    AUTH_REQUIRE_ENABLED=false DEBUG_MODE=true uvicorn app.main:app --reload --port 8000

Then run this app:
    streamlit run app/frontend/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from app.frontend.api_client import DataLensAPIError, DataLensClient
from app.frontend.sql_importer import import_sql_file_to_sqlite

DB_FILE_TYPES = ["sqlite", "db"]
SQL_FILE_TYPES = ["sql"]
DEFAULT_API_URL = "http://127.0.0.1:8000"


def init_session_state() -> None:
    defaults: dict[str, Any] = {
        "api_url": DEFAULT_API_URL,
        "access_token": None,
        "session_id": None,
        "db_id": None,
        "db_type": None,
        "table_count": None,
        "connection_message": None,
        "chat_messages": [],
        "connected": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_connection(client: DataLensClient) -> None:
    if st.session_state.session_id:
        try:
            client.disconnect(st.session_state.session_id)
        except DataLensAPIError:
            pass
    st.session_state.session_id = None
    st.session_state.db_id = None
    st.session_state.db_type = None
    st.session_state.table_count = None
    st.session_state.connection_message = None
    st.session_state.chat_messages = []
    st.session_state.connected = False


def get_client() -> DataLensClient:
    client = DataLensClient(
        base_url=st.session_state.api_url,
        access_token=st.session_state.access_token,
    )
    return client


def connect_with_payload(client: DataLensClient, payload: dict[str, Any]) -> None:
    result = client.connect(payload)
    st.session_state.session_id = result["session_id"]
    st.session_state.db_id = result["db_id"]
    st.session_state.db_type = result["db_type"]
    st.session_state.table_count = result["table_count"]
    st.session_state.connection_message = result["message"]
    st.session_state.connected = True
    st.session_state.chat_messages = []


def upload_and_connect_sqlite(
    client: DataLensClient,
    file_bytes: bytes,
    filename: str,
    db_id: str | None,
) -> None:
    upload = client.upload_db(filename, file_bytes)
    payload = {
        "db_type": "sqlite",
        "file_id": upload["file_id"],
        "db_id": db_id or Path(filename).stem,
    }
    connect_with_payload(client, payload)


def render_auth_panel(client: DataLensClient) -> bool:
    st.sidebar.subheader("Authentication")
    email = st.sidebar.text_input("Email", key="auth_email")
    password = st.sidebar.text_input("Password", type="password", key="auth_password")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Login", key="btn_login"):
        try:
            token = client.login(email, password)
            st.session_state.access_token = token["access_token"]
            st.sidebar.success("Logged in")
            st.rerun()
        except DataLensAPIError as exc:
            st.sidebar.error(str(exc))

    if col2.button("Sign up", key="btn_signup"):
        try:
            client.signup(email, password, tenant_name=email.split("@")[0])
            token = client.login(email, password)
            st.session_state.access_token = token["access_token"]
            st.sidebar.success("Account created")
            st.rerun()
        except DataLensAPIError as exc:
            st.sidebar.error(str(exc))

    return st.session_state.access_token is not None


def render_connection_panel(client: DataLensClient) -> None:
    st.subheader("Connect to a database")
    st.caption(
        "Upload a SQLite database file, upload a SQL script, or enter Postgres/MySQL credentials."
    )

    tab_db, tab_sql, tab_creds = st.tabs(
        ["Upload database file", "Upload SQL file", "Database credentials"]
    )

    with tab_db:
        uploaded_db = st.file_uploader(
            "Database file",
            type=DB_FILE_TYPES,
            key="upload_db_file",
        )
        db_id = st.text_input(
            "Database label (optional)",
            placeholder="e.g. financial",
            key="upload_db_label",
        )
        if st.button("Connect with database file", type="primary", key="btn_db_file"):
            if uploaded_db is None:
                st.error("Please upload a database file first.")
            else:
                try:
                    upload_and_connect_sqlite(
                        client,
                        uploaded_db.getvalue(),
                        uploaded_db.name,
                        db_id or None,
                    )
                    st.success(st.session_state.connection_message)
                    st.rerun()
                except DataLensAPIError as exc:
                    st.error(str(exc))

    with tab_sql:
        uploaded_sql = st.file_uploader(
            "SQL file",
            type=SQL_FILE_TYPES,
            key="upload_sql_file",
        )
        sql_db_id = st.text_input(
            "Database label (optional)",
            placeholder="e.g. my_database",
            key="upload_sql_label",
        )
        if st.button("Connect with SQL file", type="primary", key="btn_sql_file"):
            if uploaded_sql is None:
                st.error("Please upload a SQL file first.")
            else:
                try:
                    db_path = import_sql_file_to_sqlite(
                        uploaded_sql.getvalue(),
                        uploaded_sql.name,
                    )
                    upload_and_connect_sqlite(
                        client,
                        Path(db_path).read_bytes(),
                        Path(db_path).name,
                        sql_db_id or None,
                    )
                    st.success(st.session_state.connection_message)
                    st.rerun()
                except (ValueError, DataLensAPIError) as exc:
                    st.error(str(exc))

    with tab_creds:
        db_type = st.selectbox(
            "Database type",
            options=["postgres", "mysql"],
            key="cred_db_type",
        )
        col1, col2 = st.columns(2)
        with col1:
            host = st.text_input("Host", key="cred_host")
            port = st.number_input(
                "Port",
                min_value=1,
                max_value=65535,
                value=5432 if db_type == "postgres" else 3306,
            )
            database = st.text_input("Database name", key="cred_database")
        with col2:
            username = st.text_input("Username", key="cred_username")
            password = st.text_input("Password", type="password", key="cred_password")
            cred_db_id_other = st.text_input(
                "Database label (optional)", key="cred_other_label"
            )

        if st.button(f"Connect to {db_type}", type="primary", key="btn_other_db"):
            try:
                payload = {
                    "db_type": db_type,
                    "host": host,
                    "port": int(port),
                    "database": database,
                    "username": username,
                    "password": password,
                    "db_id": cred_db_id_other or database,
                }
                connect_with_payload(client, payload)
                st.success(st.session_state.connection_message)
                st.rerun()
            except DataLensAPIError as exc:
                st.error(str(exc))


def format_assistant_response(result: dict[str, Any]) -> str:
    lines = [
        "Here are the results for your question.",
        "",
        f"**Question:** {result['question']}",
        "",
        "**Generated SQL:**",
        f"```sql\n{result['generated_sql']}\n```",
        "",
        f"**Rows returned:** {result['row_count']}",
    ]
    if result["row_count"] == 0:
        lines.append("")
        lines.append("The query ran successfully but returned no rows.")
    return "\n".join(lines)


def render_chat(client: DataLensClient) -> None:
    st.subheader("Ask questions about your data")

    info_cols = st.columns(4)
    info_cols[0].metric("Database", st.session_state.db_id or "—")
    info_cols[1].metric("Type", st.session_state.db_type or "—")
    info_cols[2].metric("Tables", st.session_state.table_count or "—")
    info_cols[3].metric("Session", (st.session_state.session_id or "—")[:8] + "...")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("result"):
                result = message["result"]
                st.code(result["generated_sql"], language="sql")
                if result["row_count"] > 0:
                    df = pd.DataFrame(result["rows"], columns=result["columns"])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("Query returned 0 rows.")

    question = st.chat_input("Ask a question in plain English...")
    if question:
        st.session_state.chat_messages.append({"role": "user", "content": question})
        with st.spinner("Generating SQL and fetching results..."):
            try:
                result = client.query(st.session_state.session_id, question)
                assistant_text = format_assistant_response(result)
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_text,
                        "result": result,
                    }
                )
                st.rerun()
            except DataLensAPIError as exc:
                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": f"Sorry, something went wrong:\n\n{exc}",
                    }
                )
                st.rerun()


def render_sidebar(client: DataLensClient) -> None:
    with st.sidebar:
        st.title("DataLens")
        st.caption("Dev UI — use web/ for production")

        st.session_state.api_url = st.text_input(
            "API base URL",
            value=st.session_state.api_url,
        )
        client = get_client()

        try:
            health = client.health()
            st.success(f"API online (v{health.get('version', '?')})")
        except DataLensAPIError as exc:
            st.error(str(exc))

        render_auth_panel(client)

        st.divider()

        if st.session_state.connected:
            st.markdown("**Connected**")
            st.write(f"DB: `{st.session_state.db_id}`")
            st.write(f"Tables: `{st.session_state.table_count}`")
            if st.button("Disconnect", type="secondary"):
                reset_connection(client)
                st.rerun()
        else:
            st.markdown("**Not connected**")


def main() -> None:
    st.set_page_config(
        page_title="DataLens (Dev)",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_session_state()
    client = get_client()
    render_sidebar(client)
    client = get_client()

    st.title("DataLens")
    st.markdown(
        "**Development UI.** For production SaaS, run the Next.js app in `web/`."
    )

    if st.session_state.connected:
        render_chat(client)
    else:
        render_connection_panel(client)


if __name__ == "__main__":
    main()
