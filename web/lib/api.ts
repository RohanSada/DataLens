const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export type ApiError = { message: string; status?: number };

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("datalens_token");
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem("datalens_token", token);
  else localStorage.removeItem("datalens_token");
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  auth = true,
): Promise<T> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  if (auth) {
    const token = getToken();
    if (token) headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const data = await response.json();
      message = data.detail || message;
    } catch {
      /* ignore */
    }
    throw { message: String(message), status: response.status } as ApiError;
  }
  return response.json();
}

export const api = {
  signup: (email: string, password: string, tenant_name: string) =>
    request("/auth/signup", {
      method: "POST",
      body: JSON.stringify({ email, password, tenant_name }),
    }, false),
  login: (email: string, password: string) =>
    request<{ access_token: string }>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }, false),
  upload: async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ file_id: string; filename: string }>("/upload", {
      method: "POST",
      body: form,
      headers: {},
    });
  },
  connect: (payload: Record<string, unknown>) =>
    request("/connect", { method: "POST", body: JSON.stringify(payload) }),
  disconnect: (session_id: string) =>
    request("/disconnect", { method: "POST", body: JSON.stringify({ session_id }) }),
  query: (session_id: string, question: string) =>
    request<{
      question: string;
      generated_sql: string;
      columns: string[];
      rows: unknown[][];
      row_count: number;
    }>("/query", { method: "POST", body: JSON.stringify({ session_id, question }) }),
  testConnection: (payload: Record<string, unknown>) =>
    request<{ success: boolean; message: string }>("/connect/test", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  health: () => request<{ status: string; version: string }>("/health", {}, false),
};
