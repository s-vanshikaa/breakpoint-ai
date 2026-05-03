import { formatPct, formatPoints, formatReduction } from "../format";
import { categoryLabel, isBenign } from "../labels";
import type { ComparisonReport, TestRecord } from "../types";

function PairedBars({ baseline, guarded }: { baseline: number; guarded: number }) {
  return (
    <div className="paired-bars">
      <div className="bar-track" role="img" aria-label={`Baseline ${formatPct(baseline)}`}>
        <div className="bar-fill tone-baseline" style={{ width: `${baseline * 100}%` }} />
      </div>
      <div className="bar-track" role="img" aria-label={`Guarded ${formatPct(guarded)}`}>
        <div className="bar-fill tone-guarded" style={{ width: `${guarded * 100}%` }} />
      </div>
    </div>
  );
}

interface Props {
  summary: ComparisonReport;
  baseline: TestRecord[];
  guarded: TestRecord[];
}

export function CategoryView({ summary, baseline, guarded }: Props) {
  const categories = Object.entries(summary.category_comparison).sort(([a], [b]) =>
    categoryLabel(a).localeCompare(categoryLabel(b))
  );
  const countIn = (category: string) => baseline.filter((r) => r.category === category).length;

  const benignRows = [
    {
      label: "Baseline",
      records: baseline.filter((r) => isBenign(r.category)),
      rate: summary.benign_success_before,
    },
    {
      label: "Guarded",
      records: guarded.filter((r) => isBenign(r.category)),
      rate: summary.benign_success_after,
    },
  ];

  if (categories.length === 0) {
    return <p className="muted">No attack categories in the stored results.</p>;
  }

  return (
    <div>
      <h2>Results by category</h2>
      <p className="muted">
        Attack success rate (ASR) per malicious category. Lower is better.
      </p>
      <div className="legend" aria-hidden="true">
        <span><i className="swatch tone-baseline" /> Baseline</span>
        <span><i className="swatch tone-guarded" /> Guarded</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Category</th>
              <th className="num">Tests</th>
              <th>Attack success rate</th>
              <th className="num">Baseline</th>
              <th className="num">Guarded</th>
              <th className="num">Reduction</th>
            </tr>
          </thead>
          <tbody>
            {categories.map(([category, comp]) => (
              <tr key={category}>
                <td>{categoryLabel(category)}</td>
                <td className="num">{countIn(category)}</td>
                <td className="bar-cell">
                  <PairedBars baseline={comp.baseline_asr} guarded={comp.guarded_asr} />
                </td>
                <td className="num">{formatPct(comp.baseline_asr)}</td>
                <td className="num">{formatPct(comp.guarded_asr)}</td>
                <td className="num">
                  {formatReduction(comp.baseline_asr, comp.relative_reduction)}
                  <span className="muted small block">
                    {formatPoints(comp.guarded_asr - comp.baseline_asr)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>Benign tasks</h2>
      <p className="muted">
        Legitimate requests that should keep working. Higher is better. These are not attacks, so
        they are excluded from the attack success rates above.
      </p>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Run</th>
              <th className="num">Tasks</th>
              <th className="num">Succeeded</th>
              <th className="num">Success rate</th>
            </tr>
          </thead>
          <tbody>
            {benignRows.map(({ label, records, rate }) => (
              <tr key={label}>
                <td>{label}</td>
                <td className="num">{records.length}</td>
                <td className="num">{records.filter((r) => r.passed).length}</td>
                <td className="num">{formatPct(rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
