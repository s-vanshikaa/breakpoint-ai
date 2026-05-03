import { Fragment, useMemo, useState } from "react";
import { categoryLabel, isBenign, outcomeLabel, targetLabel } from "../labels";
import {
  OUTCOME_FILTERS,
  countRows,
  matchesOutcome,
  type OutcomeFilter,
  type TestRow,
} from "../rows";
import type { TestRecord } from "../types";

function ResultBadge({ record }: { record?: TestRecord }) {
  if (!record) return <span className="muted">—</span>;
  const good = record.passed;
  return <span className={`badge ${good ? "badge-good" : "badge-bad"}`}>{outcomeLabel(record)}</span>;
}

function RunDetail({ title, record }: { title: string; record?: TestRecord }) {
  if (!record) {
    return (
      <div className="run-detail">
        <h4>{title}</h4>
        <p className="muted">No record for this run.</p>
      </div>
    );
  }
  return (
    <div className="run-detail">
      <h4>
        {title} <ResultBadge record={record} />
      </h4>
      {record.block_reason && (
        <p>
          <strong>Blocked by guardrail:</strong> {record.block_reason}
        </p>
      )}
      <p>
        <strong>Response:</strong> {record.response || <em>(empty)</em>}
      </p>
      <p>
        <strong>Evaluator:</strong> {record.reason}
      </p>
      {record.requested_tool_call && (
        <p>
          <strong>Tool requested:</strong> <code>{record.requested_tool_call.tool}</code>{" "}
          <code>{JSON.stringify(record.requested_tool_call.arguments)}</code>
        </p>
      )}
      {record.requested_tool_call && (
        <p>
          <strong>Tool executed:</strong>{" "}
          {record.executed_tool_call ? (
            <>
              <code>{record.executed_tool_call.tool}</code> [{record.executed_tool_call.permission}]{" "}
              → {record.executed_tool_call.result}
            </>
          ) : (
            "no (blocked or failed)"
          )}
        </p>
      )}
      {record.retrieved_context && record.retrieved_context.length > 0 && (
        <details>
          <summary>Retrieved context ({record.retrieved_context.length} chunks)</summary>
          {record.retrieved_context.map((c, i) => (
            <div key={i}>
              <p className="muted small">
                {c.source} · score {c.score.toFixed(3)}
              </p>
              <pre>{c.text}</pre>
            </div>
          ))}
        </details>
      )}
      <p className="muted small">Latency {record.latency_ms.toFixed(0)} ms</p>
    </div>
  );
}

interface Props {
  rows: TestRow[];
  outcome: OutcomeFilter;
  onOutcomeChange: (outcome: OutcomeFilter) => void;
}

export function TestExplorer({ rows, outcome, onOutcomeChange }: Props) {
  const [category, setCategory] = useState("");
  const [query, setQuery] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const categories = useMemo(() => [...new Set(rows.map((r) => r.category))], [rows]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter(
      (r) =>
        (!category || r.category === category) &&
        matchesOutcome(r, outcome) &&
        (!q || r.id.toLowerCase().includes(q) || r.prompt.toLowerCase().includes(q))
    );
  }, [rows, category, outcome, query]);

  const reset = () => {
    setCategory("");
    setQuery("");
    onOutcomeChange("all");
  };

  return (
    <div>
      <h2>Test explorer</h2>
      <p className="muted">
        Every benchmark test with its baseline and guarded result. Select a test to see the prompt,
        the model's response and how the evaluator judged it.
      </p>

      <div className="controls">
        <label>
          Show
          <select value={outcome} onChange={(e) => onOutcomeChange(e.target.value as OutcomeFilter)}>
            {OUTCOME_FILTERS.map((f) => (
              <option key={f.value} value={f.value}>
                {f.label} ({countRows(rows, f.value)})
              </option>
            ))}
          </select>
        </label>
        <label>
          Category
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">All categories</option>
            {categories.map((c) => (
              <option key={c} value={c}>
                {categoryLabel(c)}
              </option>
            ))}
          </select>
        </label>
        <label className="grow">
          Search
          <input
            type="search"
            placeholder="Test ID or prompt text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
      </div>

      <p className="muted small" role="status">
        Showing {visible.length} of {rows.length} tests.
      </p>

      {visible.length === 0 ? (
        <div className="state">
          <p>No tests match these filters.</p>
          <button type="button" className="button" onClick={reset}>
            Clear filters
          </button>
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Test</th>
                <th className="hide-narrow">Type</th>
                <th className="hide-narrow">Category</th>
                <th className="hide-narrow">Target</th>
                <th>Baseline</th>
                <th>Guarded</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((row) => {
                const open = expandedId === row.id;
                return (
                  <Fragment key={row.id}>
                    <tr className={open ? "row-open" : ""}>
                      <td>
                        <button
                          type="button"
                          className="link mono"
                          aria-expanded={open}
                          onClick={() => setExpandedId(open ? null : row.id)}
                        >
                          {open ? "▾" : "▸"} {row.id}
                        </button>
                      </td>
                      <td className="hide-narrow">
                        <span className={`badge ${isBenign(row.category) ? "badge-neutral" : "badge-attack"}`}>
                          {isBenign(row.category) ? "Benign" : "Malicious"}
                        </span>
                      </td>
                      <td className="hide-narrow">{categoryLabel(row.category)}</td>
                      <td className="hide-narrow">{targetLabel(row.target)}</td>
                      <td>
                        <ResultBadge record={row.baseline} />
                      </td>
                      <td>
                        <ResultBadge record={row.guarded} />
                      </td>
                    </tr>
                    {open && (
                      <tr>
                        <td colSpan={6} className="detail-cell">
                          <div className="detail">
                            <p>
                              <strong>Prompt:</strong> {row.prompt}
                            </p>
                            <div className="run-details">
                              <RunDetail title="Baseline (guardrails off)" record={row.baseline} />
                              <RunDetail title="Guarded (guardrails on)" record={row.guarded} />
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
