import { formatPct, formatPoints, formatReduction } from "../format";
import { isBenign } from "../labels";
import type { OutcomeFilter } from "../rows";
import type { ComparisonReport, TestRecord } from "../types";
import { MetricCard } from "./MetricCard";

interface Props {
  summary: ComparisonReport;
  baseline: TestRecord[];
  guarded: TestRecord[];
  onExplore: (filter: OutcomeFilter) => void;
}

export function SummaryView({ summary, baseline, guarded, onExplore }: Props) {
  const total = summary.baseline.total_tests;
  const benignCount = baseline.filter((r) => isBenign(r.category)).length;
  const attackCount = baseline.length - benignCount;
  const categoryCount = new Set(
    baseline.filter((r) => !isBenign(r.category)).map((r) => r.category)
  ).size;

  const stillSucceeding = guarded.filter((r) => !isBenign(r.category) && !r.passed).length;
  const benignBroken = guarded.filter((r) => isBenign(r.category) && !r.passed).length;

  const asrDelta = summary.guarded.overall_asr - summary.baseline.overall_asr;
  const benignDelta = summary.benign_success_after - summary.benign_success_before;

  return (
    <div>
      <h2>Baseline vs. guarded</h2>
      <p className="muted">
        {total} tests: {attackCount} malicious across {categoryCount} attack categories, plus{" "}
        {benignCount} benign tasks. The same tests are run twice: once against the unprotected
        targets (baseline) and once with input, retrieval and tool-permission guardrails enabled
        (guarded).
      </p>

      <div className="cards">
        <MetricCard
          title="Attack success rate"
          description="Share of malicious tests where the attack worked."
          baseline={summary.baseline.overall_asr}
          guarded={summary.guarded.overall_asr}
          better="lower"
          badge={{
            text: `${formatReduction(summary.baseline.overall_asr, summary.overall_relative_asr_reduction)} relative reduction (${formatPoints(asrDelta)})`,
            tone: asrDelta < 0 ? "good" : "neutral",
          }}
        />
        <MetricCard
          title="Tool misuse rate"
          description="Share of tool-abuse tests where a restricted or forbidden tool actually ran."
          baseline={summary.tool_misuse_rate_before}
          guarded={summary.tool_misuse_rate_after}
          better="lower"
          badge={{
            text: formatPoints(summary.tool_misuse_rate_after - summary.tool_misuse_rate_before),
            tone:
              summary.tool_misuse_rate_after < summary.tool_misuse_rate_before ? "good" : "neutral",
          }}
        />
        <MetricCard
          title="Benign task success"
          description="Share of legitimate tasks still completed correctly. Guardrails should not break these."
          baseline={summary.benign_success_before}
          guarded={summary.benign_success_after}
          better="higher"
          badge={{
            text: formatPoints(benignDelta),
            tone: benignDelta < 0 ? "bad" : "neutral",
          }}
        />
      </div>

      <h3>Where to look next</h3>
      <ul className="link-list">
        <li>
          <button type="button" className="link" onClick={() => onExplore("beat_guardrails")}>
            {stillSucceeding} attack{stillSucceeding === 1 ? "" : "s"} still succeed
          </button>{" "}
          with guardrails on ({formatPct(summary.guarded.overall_asr)} of malicious tests).
        </li>
        <li>
          <button type="button" className="link" onClick={() => onExplore("benign_failed_guarded")}>
            {benignBroken} benign task{benignBroken === 1 ? "" : "s"} fail
          </button>{" "}
          with guardrails on (the cost of the defenses).
        </li>
      </ul>
    </div>
  );
}
