import { useCallback, useEffect, useState } from "react";
import { ApiError, fetchRecords, fetchSummary } from "./api";
import type { ComparisonReport, TestRecord } from "./types";

export type DashboardState =
  | { status: "loading" }
  | { status: "error"; message: string; missingResults: boolean }
  | {
      status: "ready";
      summary: ComparisonReport;
      baseline: TestRecord[];
      guarded: TestRecord[];
    };

export function useDashboardData() {
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<DashboardState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchSummary(), fetchRecords("baseline"), fetchRecords("guarded")])
      .then(([summary, baseline, guarded]) => {
        if (!cancelled) setState({ status: "ready", summary, baseline, guarded });
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setState({
          status: "error",
          message: e instanceof Error ? e.message : String(e),
          missingResults: e instanceof ApiError && e.status === 404,
        });
      });
    return () => {
      cancelled = true;
    };
  }, [attempt]);

  const retry = useCallback(() => {
    setState({ status: "loading" });
    setAttempt((a) => a + 1);
  }, []);

  return [state, retry] as const;
}
