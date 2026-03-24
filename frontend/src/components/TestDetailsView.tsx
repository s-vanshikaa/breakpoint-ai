import { useEffect, useState } from "react";
import { fetchRecords } from "../api";
import type { RunName, TestRecord } from "../types";

const CATEGORIES = [
  "direct_injection",
  "indirect_rag_injection",
  "secret_extraction",
  "tool_misuse",
  "benign",
];

export function TestDetailsView() {
  const [run, setRun] = useState<RunName>("guarded");
  const [category, setCategory] = useState<string>("");
  const [records, setRecords] = useState<TestRecord[]>([]);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetchRecords(run, category || undefined)
      .then(setRecords)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [run, category]);

  return (
    <div>
      <h2>Test Details</h2>
      <div className="controls">
        <label>
          Run:{" "}
          <select value={run} onChange={(e) => setRun(e.target.value as RunName)}>
            <option value="baseline">Baseline (guardrails off)</option>
            <option value="guarded">Guarded (guardrails on)</option>
          </select>
        </label>
        <label>
          Category:{" "}
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">All</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      </div>

      {loading && <p>Loading...</p>}
      {error && <p className="error">{error}</p>}

      <table>
        <thead>
          <tr>
            <th>Test ID</th>
            <th>Category</th>
            <th>Target</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <>
              <tr
                key={r.test_id}
                className="clickable-row"
                onClick={() =>
                  setExpandedId(expandedId === r.test_id ? null : r.test_id)
                }
              >
                <td>{r.test_id}</td>
                <td>{r.category}</td>
                <td>{r.target}</td>
                <td className={r.passed ? "pass" : "fail"}>
                  {r.passed ? "PASS" : "FAIL"}
                </td>
              </tr>
              {expandedId === r.test_id && (
                <tr>
                  <td colSpan={4}>
                    <div className="detail">
                      <p>
                        <strong>Prompt:</strong> {r.prompt}
                      </p>
                      <p>
                        <strong>Response:</strong> {r.response}
                      </p>
                      <p>
                        <strong>Reason:</strong> {r.reason}
                      </p>
                      <p>
                        <strong>Guardrails enabled:</strong>{" "}
                        {String(r.guardrails_enabled)}
                        {r.block_reason && (
                          <>
                            {" "}
                            — <strong>Blocked:</strong> {r.block_reason}
                          </>
                        )}
                      </p>
                      <p>
                        <strong>Latency:</strong> {r.latency_ms.toFixed(0)} ms
                      </p>

                      {r.retrieved_context && r.retrieved_context.length > 0 && (
                        <div>
                          <strong>Retrieved context:</strong>
                          <ul>
                            {r.retrieved_context.map((c, i) => (
                              <li key={i}>
                                <em>
                                  [{c.source}, score {c.score.toFixed(3)}]
                                </em>
                                <pre>{c.text}</pre>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {r.requested_tool_call && (
                        <p>
                          <strong>Requested tool call:</strong>{" "}
                          {r.requested_tool_call.tool}(
                          {JSON.stringify(r.requested_tool_call.arguments)})
                        </p>
                      )}
                      {r.executed_tool_call ? (
                        <p>
                          <strong>Executed tool call:</strong>{" "}
                          {r.executed_tool_call.tool} [
                          {r.executed_tool_call.permission}] →{" "}
                          {r.executed_tool_call.result}
                        </p>
                      ) : r.requested_tool_call ? (
                        <p>
                          <strong>Executed tool call:</strong> none (blocked or
                          failed)
                        </p>
                      ) : null}
                    </div>
                  </td>
                </tr>
              )}
            </>
          ))}
        </tbody>
      </table>
    </div>
  );
}
