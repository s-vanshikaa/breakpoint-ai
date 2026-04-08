# BreakPoint AI dashboard

React + TypeScript + Vite dashboard for browsing stored benchmark results.
It reads from the FastAPI backend (`/results/summary`, `/results/records`).

```bash
npm install
cp .env.example .env   # optional; only needed if the API is not on localhost:8000
npm run dev            # http://localhost:5173
```

| Script | Purpose |
| --- | --- |
| `npm run dev` | Start the dev server |
| `npm run build` | Type-check and build for production |
| `npm run lint` | Lint with oxlint |
