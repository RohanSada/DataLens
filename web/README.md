# DataLens production web frontend (Next.js)

Run against a local API:

```bash
cd web
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev
```

Build for production:

```bash
npm run build
npm start
```

For local development without auth, start the API with `AUTH_REQUIRE_ENABLED=false`.

The Streamlit app in `app/frontend/` is retained for internal debugging only.
