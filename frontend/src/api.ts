import type { ComparisonReport, RunName, TestRecord } from "./types";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

async function getJson<T>(path: string, what: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`);
  } catch {
    throw new Error(
      `Could not reach the backend at ${API_BASE}. Is it running? (uvicorn api.main:app)`
    );
  }
  if (!res.ok) {
    let detail = "";
    try {
      detail = (await res.json()).detail ?? "";
    } catch {
      // Response body was not JSON; fall back to the status code alone.
    }
    throw new Error(`Failed to load ${what} (HTTP ${res.status})${detail ? `: ${detail}` : ""}`);
  }
  return res.json();
}

export function fetchSummary(): Promise<ComparisonReport> {
  return getJson("/results/summary", "summary");
}

export function fetchRecords(
  run: RunName,
  category?: string
): Promise<TestRecord[]> {
  const params = new URLSearchParams({ run });
  if (category) params.set("category", category);
  return getJson(`/results/records?${params}`, "test records");
}
