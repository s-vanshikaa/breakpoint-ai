import type { TestRecord } from "./types";
import { isBenign } from "./labels";

export interface TestRow {
  id: string;
  category: string;
  target: string;
  prompt: string;
  baseline?: TestRecord;
  guarded?: TestRecord;
}

export type OutcomeFilter =
  | "all"
  | "beat_baseline"
  | "beat_guardrails"
  | "stopped_by_guardrails"
  | "benign_failed_guarded";

export const OUTCOME_FILTERS: { value: OutcomeFilter; label: string }[] = [
  { value: "all", label: "All tests" },
  { value: "beat_baseline", label: "Attacks that succeeded without guardrails" },
  { value: "beat_guardrails", label: "Attacks that still succeed with guardrails" },
  { value: "stopped_by_guardrails", label: "Attacks stopped only by guardrails" },
  { value: "benign_failed_guarded", label: "Benign tests that fail with guardrails" },
];

export function buildRows(baseline: TestRecord[], guarded: TestRecord[]): TestRow[] {
  const rows = new Map<string, TestRow>();
  for (const [run, records] of [["baseline", baseline], ["guarded", guarded]] as const) {
    for (const r of records) {
      const row = rows.get(r.test_id) ?? {
        id: r.test_id,
        category: r.category,
        target: r.target,
        prompt: r.prompt,
      };
      row[run] = r;
      rows.set(r.test_id, row);
    }
  }
  return [...rows.values()];
}

export function matchesOutcome(row: TestRow, filter: OutcomeFilter): boolean {
  const attack = !isBenign(row.category);
  switch (filter) {
    case "all":
      return true;
    case "beat_baseline":
      return attack && row.baseline?.passed === false;
    case "beat_guardrails":
      return attack && row.guarded?.passed === false;
    case "stopped_by_guardrails":
      return attack && row.baseline?.passed === false && row.guarded?.passed === true;
    case "benign_failed_guarded":
      return !attack && row.guarded?.passed === false;
  }
}

export function countRows(rows: TestRow[], filter: OutcomeFilter): number {
  return rows.filter((r) => matchesOutcome(r, filter)).length;
}
