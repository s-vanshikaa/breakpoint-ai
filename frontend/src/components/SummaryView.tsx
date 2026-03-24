import type { ComparisonReport } from "../types";
import { formatPct } from "../format";

export function SummaryView({ summary }: { summary: ComparisonReport }) {
  return (
    <div>
      <h2>Overview</h2>
      <table>
        <tbody>
          <tr>
            <th>Total tests</th>
            <td>{summary.baseline.total_tests}</td>
          </tr>
          <tr>
            <th>Baseline attack success rate (ASR)</th>
            <td>{formatPct(summary.baseline.overall_asr)}</td>
          </tr>
          <tr>
            <th>Guarded attack success rate (ASR)</th>
            <td>{formatPct(summary.guarded.overall_asr)}</td>
          </tr>
          <tr>
            <th>Overall relative ASR reduction</th>
            <td>{formatPct(summary.overall_relative_asr_reduction)}</td>
          </tr>
          <tr>
            <th>Tool misuse rate (baseline → guarded)</th>
            <td>
              {formatPct(summary.tool_misuse_rate_before)} →{" "}
              {formatPct(summary.tool_misuse_rate_after)}
            </td>
          </tr>
          <tr>
            <th>Benign task success (baseline → guarded)</th>
            <td>
              {formatPct(summary.benign_success_before)} →{" "}
              {formatPct(summary.benign_success_after)}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}
