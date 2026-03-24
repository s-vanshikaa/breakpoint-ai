import type { ComparisonReport, RunName, TestRecord } from "./types";

const API_BASE = "http://localhost:8000";

export async function fetchSummary(): Promise<ComparisonReport> {
  const res = await fetch(`${API_BASE}/results/summary`);
  if (!res.ok) throw new Error(`Failed to load summary: ${res.status}`);
  return res.json();
}

export async function fetchRecords(
  run: RunName,
  category?: string
): Promise<TestRecord[]> {
  const params = new URLSearchParams({ run });
  if (category) params.set("category", category);
  const res = await fetch(`${API_BASE}/results/records?${params}`);
  if (!res.ok) throw new Error(`Failed to load records: ${res.status}`);
  return res.json();
}
