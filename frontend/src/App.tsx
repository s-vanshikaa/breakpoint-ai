import { useMemo, useState } from "react";
import { CategoryView } from "./components/CategoryView";
import { EmptyState, ErrorState, LoadingState, NoResultsHelp } from "./components/States";
import { SummaryView } from "./components/SummaryView";
import { TestExplorer } from "./components/TestExplorer";
import { buildRows, type OutcomeFilter } from "./rows";
import { useDashboardData } from "./useDashboardData";
import "./App.css";

type Tab = "summary" | "categories" | "tests";

const TABS: { id: Tab; label: string }[] = [
  { id: "summary", label: "Overview" },
  { id: "categories", label: "Categories" },
  { id: "tests", label: "Test explorer" },
];

function tabFromHash(): Tab {
  const id = window.location.hash.replace("#", "");
  return TABS.some((t) => t.id === id) ? (id as Tab) : "summary";
}

function App() {
  const [tab, setTabState] = useState<Tab>(tabFromHash);
  const [outcome, setOutcome] = useState<OutcomeFilter>("all");
  const [state, retry] = useDashboardData();

  const setTab = (next: Tab) => {
    window.history.replaceState(null, "", `#${next}`);
    setTabState(next);
  };

  const rows = useMemo(
    () => (state.status === "ready" ? buildRows(state.baseline, state.guarded) : []),
    [state]
  );

  const explore = (filter: OutcomeFilter) => {
    setOutcome(filter);
    setTab("tests");
  };

  return (
    <div className="dashboard">
      <header>
        <h1>BreakPoint AI</h1>
        <p className="subtitle">
          An adversarial evaluation framework for RAG systems and tool-using LLM agents. It
          measures how often prompt injection, secret extraction and tool misuse succeed, with and
          without guardrails.
        </p>
      </header>

      <nav className="tabs" aria-label="Dashboard sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={tab === t.id ? "active" : ""}
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {state.status === "loading" && <LoadingState />}

        {state.status === "error" &&
          (state.missingResults ? (
            <EmptyState title="No benchmark results yet">
              <NoResultsHelp />
              <button type="button" className="button" onClick={retry}>
                Reload results
              </button>
            </EmptyState>
          ) : (
            <ErrorState message={state.message} onRetry={retry} />
          ))}

        {state.status === "ready" && rows.length === 0 && (
          <EmptyState title="The stored results contain no tests">
            <NoResultsHelp />
          </EmptyState>
        )}

        {state.status === "ready" && rows.length > 0 && (
          <>
            {tab === "summary" && (
              <SummaryView
                summary={state.summary}
                baseline={state.baseline}
                guarded={state.guarded}
                onExplore={explore}
              />
            )}
            {tab === "categories" && (
              <CategoryView
                summary={state.summary}
                baseline={state.baseline}
                guarded={state.guarded}
              />
            )}
            {tab === "tests" && (
              <TestExplorer rows={rows} outcome={outcome} onOutcomeChange={setOutcome} />
            )}
          </>
        )}
      </main>

      <footer className="muted small">
        Results come from a local LLM and synthetic attacks; they illustrate the measurement
        method, not real-world security guarantees.
      </footer>
    </div>
  );
}

export default App;
