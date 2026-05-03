export function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Signed change in percentage points, e.g. "-27.5 pp". */
export function formatPoints(delta: number): string {
  const points = delta * 100;
  const sign = points > 0 ? "+" : points < 0 ? "−" : "";
  return `${sign}${Math.abs(points).toFixed(1)} pp`;
}

/** Relative reduction; "n/a" when there was nothing to reduce. */
export function formatReduction(baseline: number, reduction: number): string {
  return baseline === 0 ? "n/a" : formatPct(reduction);
}
