import type { TestRecord } from "./types";

export const CATEGORY_LABELS: Record<string, string> = {
  direct_injection: "Direct prompt injection",
  indirect_rag_injection: "Indirect (RAG) injection",
  secret_extraction: "Secret extraction",
  tool_misuse: "Tool misuse",
  benign: "Benign tasks",
};

export const TARGET_LABELS: Record<string, string> = {
  rag_assistant: "RAG assistant",
  tool_agent: "Tool agent",
};

export const BENIGN = "benign";

export const categoryLabel = (c: string) => CATEGORY_LABELS[c] ?? c;
export const targetLabel = (t: string) => TARGET_LABELS[t] ?? t;
export const isBenign = (category: string) => category === BENIGN;

/** A passing malicious test means the attack was stopped; a passing benign test means the task worked. */
export function outcomeLabel(record: TestRecord): string {
  if (isBenign(record.category)) return record.passed ? "Task succeeded" : "Task failed";
  return record.passed ? "Attack blocked" : "Attack succeeded";
}
