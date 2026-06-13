"use client";

import { useEffect, useState } from "react";
import { api, setToken } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
  sql?: string;
  columns?: string[];
  rows?: unknown[][];
};

export default function HomePage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loggedIn, setLoggedIn] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [dbId, setDbId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [dbType, setDbType] = useState<"sqlite" | "postgres" | "mysql">("sqlite");
  const [host, setHost] = useState("");
  const [port, setPort] = useState("5432");
  const [database, setDatabase] = useState("");
  const [username, setUsername] = useState("");
  const [dbPassword, setDbPassword] = useState("");

  useEffect(() => {
    setLoggedIn(!!localStorage.getItem("datalens_token"));
  }, []);

  async function handleLogin(signup = false) {
    setError(null);
    try {
      if (signup) await api.signup(email, password, email.split("@")[0]);
      const { access_token } = await api.login(email, password);
      setToken(access_token);
      setLoggedIn(true);
    } catch (e) {
      setError((e as { message: string }).message);
    }
  }

  async function handleUpload(file: File | null) {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const upload = await api.upload(file);
      const result = await api.connect({
        db_type: "sqlite",
        file_id: upload.file_id,
        db_id: file.name.replace(/\.[^.]+$/, ""),
      }) as { session_id: string; db_id: string };
      setSessionId(result.session_id);
      setDbId(result.db_id);
      setMessages([]);
    } catch (e) {
      setError((e as { message: string }).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleRemoteConnect() {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        db_type: dbType,
        host,
        port: Number(port),
        database,
        username,
        password: dbPassword,
        db_id: database,
      };
      if (dbType !== "sqlite") {
        const test = await api.testConnection(payload);
        if (!test.success) throw new Error(test.message);
      }
      const result = await api.connect(payload) as { session_id: string; db_id: string };
      setSessionId(result.session_id);
      setDbId(result.db_id);
      setMessages([]);
    } catch (e) {
      setError((e as { message: string }).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleAsk(e: React.FormEvent) {
    e.preventDefault();
    if (!sessionId || !question.trim()) return;
    setLoading(true);
    setError(null);
    const q = question.trim();
    setQuestion("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    try {
      const result = await api.query(sessionId, q);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: `Returned ${result.row_count} row(s).`,
          sql: result.generated_sql,
          columns: result.columns,
          rows: result.rows,
        },
      ]);
    } catch (err) {
      setError((err as { message: string }).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleDisconnect() {
    if (sessionId) {
      try {
        await api.disconnect(sessionId);
      } catch {
        /* ignore */
      }
    }
    setSessionId(null);
    setDbId(null);
    setMessages([]);
  }

  function logout() {
    setToken(null);
    setLoggedIn(false);
    handleDisconnect();
  }

  if (!loggedIn) {
    return (
      <main style={{ maxWidth: 420, margin: "4rem auto", padding: "0 1rem" }}>
        <h1>DataLens</h1>
        <p>Natural language to SQL for your databases.</p>
        {error && <p style={{ color: "#f87171" }}>{error}</p>}
        <input
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={inputStyle}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={inputStyle}
        />
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button style={buttonStyle} onClick={() => handleLogin(false)}>Log in</button>
          <button style={buttonStyle} onClick={() => handleLogin(true)}>Sign up</button>
        </div>
      </main>
    );
  }

  return (
    <main style={{ maxWidth: 960, margin: "0 auto", padding: "1.5rem" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 style={{ margin: 0 }}>DataLens</h1>
          <p style={{ margin: "0.25rem 0 0", color: "#94a3b8" }}>
            {sessionId ? `Connected to ${dbId}` : "Connect a database to begin"}
          </p>
        </div>
        <button style={buttonStyle} onClick={logout}>Log out</button>
      </header>

      {error && <p style={{ color: "#f87171" }}>{error}</p>}

      {!sessionId ? (
        <section style={panelStyle}>
          <h2>Connect</h2>
          <label style={labelStyle}>
            Upload SQLite (.sqlite / .db)
            <input
              type="file"
              accept=".sqlite,.db"
              onChange={(e) => handleUpload(e.target.files?.[0] || null)}
            />
          </label>

          <hr style={{ borderColor: "#334155", margin: "1.5rem 0" }} />

          <h3>Remote database</h3>
          <select value={dbType} onChange={(e) => setDbType(e.target.value as typeof dbType)} style={inputStyle}>
            <option value="postgres">Postgres</option>
            <option value="mysql">MySQL</option>
          </select>
          <input placeholder="Host" value={host} onChange={(e) => setHost(e.target.value)} style={inputStyle} />
          <input placeholder="Port" value={port} onChange={(e) => setPort(e.target.value)} style={inputStyle} />
          <input placeholder="Database" value={database} onChange={(e) => setDatabase(e.target.value)} style={inputStyle} />
          <input placeholder="Username" value={username} onChange={(e) => setUsername(e.target.value)} style={inputStyle} />
          <input type="password" placeholder="Password" value={dbPassword} onChange={(e) => setDbPassword(e.target.value)} style={inputStyle} />
          <button style={buttonStyle} disabled={loading} onClick={handleRemoteConnect}>
            {loading ? "Connecting..." : "Connect"}
          </button>
        </section>
      ) : (
        <>
          <button style={{ ...buttonStyle, marginBottom: "1rem" }} onClick={handleDisconnect}>
            Disconnect
          </button>
          <section style={panelStyle}>
            {messages.map((msg, i) => (
              <div key={i} style={{ marginBottom: "1rem" }}>
                <strong>{msg.role === "user" ? "You" : "DataLens"}</strong>
                <p>{msg.content}</p>
                {msg.sql && (
                  <pre style={codeStyle}>{msg.sql}</pre>
                )}
                {msg.columns && msg.rows && msg.rows.length > 0 && (
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          {msg.columns.map((c) => (
                            <th key={c} style={thStyle}>{c}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {msg.rows.map((row, ri) => (
                          <tr key={ri}>
                            {(row as unknown[]).map((cell, ci) => (
                              <td key={ci} style={tdStyle}>{String(cell ?? "")}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            ))}
            <form onSubmit={handleAsk} style={{ display: "flex", gap: "0.5rem" }}>
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask a question in plain English..."
                style={{ ...inputStyle, flex: 1 }}
              />
              <button style={buttonStyle} disabled={loading} type="submit">
                {loading ? "..." : "Ask"}
              </button>
            </form>
          </section>
        </>
      )}
    </main>
  );
}

const inputStyle: React.CSSProperties = {
  display: "block",
  width: "100%",
  marginBottom: "0.75rem",
  padding: "0.6rem 0.75rem",
  borderRadius: 8,
  border: "1px solid #334155",
  background: "#1e293b",
  color: "#e2e8f0",
  boxSizing: "border-box",
};

const buttonStyle: React.CSSProperties = {
  padding: "0.6rem 1rem",
  borderRadius: 8,
  border: "none",
  background: "#3b82f6",
  color: "white",
  cursor: "pointer",
};

const panelStyle: React.CSSProperties = {
  marginTop: "1.5rem",
  padding: "1.25rem",
  borderRadius: 12,
  background: "#1e293b",
  border: "1px solid #334155",
};

const labelStyle: React.CSSProperties = { display: "block", marginBottom: "1rem" };
const codeStyle: React.CSSProperties = {
  background: "#0f172a",
  padding: "0.75rem",
  borderRadius: 8,
  overflowX: "auto",
};
const thStyle: React.CSSProperties = {
  textAlign: "left",
  padding: "0.5rem",
  borderBottom: "1px solid #334155",
};
const tdStyle: React.CSSProperties = {
  padding: "0.5rem",
  borderBottom: "1px solid #1e293b",
};
