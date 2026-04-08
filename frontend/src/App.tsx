import { useEffect, useState } from "react";
import { fetchSummary } from "./api";
import { CategoryView } from "./components/CategoryView";
import { SummaryView } from "./components/SummaryView";
import { TestDetailsView } from "./components/TestDetailsView";
import type { ComparisonReport } from "./types";
import "./App.css";

type Tab = "summary" | "categories" | "details";

function App() {
  const [tab, setTab] = useState<Tab>("summary");
  const [summary, setSummary] = useState<ComparisonReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSummary()
      .then(setSummary)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  return (
    <div className="dashboard">
      <h1>BreakPoint AI</h1>
      <p className="subtitle">
        Adversarial evaluation results for the RAG assistant and IT tool agent.
      </p>

      <nav className="tabs">
        <button
          className={tab === "summary" ? "active" : ""}
          onClick={() => setTab("summary")}
        >
          Overview
        </button>
        <button
          className={tab === "categories" ? "active" : ""}
          onClick={() => setTab("categories")}
        >
          Category Results
        </button>
        <button
          className={tab === "details" ? "active" : ""}
          onClick={() => setTab("details")}
        >
          Test Details
        </button>
      </nav>

      {error && (
        <p className="error">
          {error}
        </p>
      )}

      {!error && !summary && <p>Loading...</p>}

      {summary && tab === "summary" && <SummaryView summary={summary} />}
      {summary && tab === "categories" && <CategoryView summary={summary} />}
      {tab === "details" && <TestDetailsView />}
    </div>
  );
}

export default App;
