import type { ComparisonReport, RunName, TestRecord } from "./types";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export class ApiError extends Error {
  status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function getJson<T>(path: string, what: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`);
  } catch {
    throw new ApiError(
      `Could not reach the backend at ${API_BASE}. Is it running? (uvicorn api.main:app)`,
      null
    );
  }
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail ?? "";
    } catch {
      // Response body was not JSON; fall back to the status code alone.
    }
    throw new ApiError(
      `Failed to load ${what} (HTTP ${res.status})${detail ? `: ${detail}` : ""}`,
      res.status
    );
  }
  return res.json();
}

export function fetchSummary(): Promise<ComparisonReport> {
  return getJson("/results/summary", "summary");
}

export function fetchRecords(run: RunName): Promise<TestRecord[]> {
  return getJson(`/results/records?run=${run}`, `${run} test records`);
}
