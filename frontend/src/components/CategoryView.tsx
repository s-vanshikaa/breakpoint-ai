import type { ComparisonReport } from "../types";
import { formatPct } from "../format";

export function CategoryView({ summary }: { summary: ComparisonReport }) {
  const categories = Object.entries(summary.category_comparison).sort(([a], [b]) =>
    a.localeCompare(b)
  );

  return (
    <div>
      <h2>Category Breakdown</h2>
      <table>
        <thead>
          <tr>
            <th>Category</th>
            <th>Baseline ASR</th>
            <th>Guarded ASR</th>
            <th>Relative reduction</th>
          </tr>
        </thead>
        <tbody>
          {categories.map(([category, comp]) => (
            <tr key={category}>
              <td>{category}</td>
              <td>{formatPct(comp.baseline_asr)}</td>
              <td>{formatPct(comp.guarded_asr)}</td>
              <td>{formatPct(comp.relative_reduction)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">
        ASR = attack success rate. Lower is better. This table only covers
        adversarial categories; benign task success is on the Overview tab.
      </p>
    </div>
  );
}
