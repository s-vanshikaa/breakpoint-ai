import { formatPct } from "../format";

interface Props {
  title: string;
  description: string;
  baseline: number;
  guarded: number;
  /** "lower" when a drop is an improvement (attacks); "higher" when a rise is (benign tasks). */
  better: "lower" | "higher";
  badge?: { text: string; tone: "good" | "bad" | "neutral" };
}

function Bar({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="metric-row">
      <span className="metric-row-label">{label}</span>
      <div
        className="bar-track"
        role="img"
        aria-label={`${label}: ${formatPct(value)}`}
      >
        <div className={`bar-fill ${tone}`} style={{ width: `${Math.min(100, value * 100)}%` }} />
      </div>
      <span className="metric-row-value">{formatPct(value)}</span>
    </div>
  );
}

export function MetricCard({ title, description, baseline, guarded, better, badge }: Props) {
  return (
    <section className="card metric-card">
      <h3>{title}</h3>
      <p className="muted small">{description}</p>
      <p className="metric-headline">
        <span className="metric-big">{formatPct(guarded)}</span>
        <span className="muted small"> guarded (was {formatPct(baseline)})</span>
      </p>
      <Bar label="Baseline" value={baseline} tone="tone-baseline" />
      <Bar label="Guarded" value={guarded} tone="tone-guarded" />
      <p className="muted small">{better === "lower" ? "Lower is better" : "Higher is better"}</p>
      {badge && <span className={`badge badge-${badge.tone}`}>{badge.text}</span>}
    </section>
  );
}
