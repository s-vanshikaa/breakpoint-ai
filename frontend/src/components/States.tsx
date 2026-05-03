export function LoadingState() {
  return (
    <div className="state" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p>Loading benchmark results…</p>
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div className="state state-error" role="alert">
      <h2>Couldn't load results</h2>
      <p>{message}</p>
      <button type="button" className="button" onClick={onRetry}>
        Try again
      </button>
    </div>
  );
}

export function EmptyState({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="state">
      <h2>{title}</h2>
      {children}
    </div>
  );
}

export function NoResultsHelp() {
  return (
    <>
      <p>Run the benchmark to generate results, then reload this page:</p>
      <pre className="command">
        {"cd backend\npython -m runner all"}
      </pre>
    </>
  );
}
