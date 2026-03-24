export interface CategoryComparison {
  baseline_asr: number;
  guarded_asr: number;
  absolute_reduction: number;
  relative_reduction: number;
}

export interface BaselineMetrics {
  total_tests: number;
  overall_asr: number;
  asr_by_category: Record<string, number>;
  tool_misuse_rate: number;
  benign_success_rate: number;
  guardrails_enabled: boolean;
}

export interface ComparisonReport {
  baseline: BaselineMetrics;
  guarded: BaselineMetrics;
  overall_relative_asr_reduction: number;
  category_comparison: Record<string, CategoryComparison>;
  tool_misuse_rate_before: number;
  tool_misuse_rate_after: number;
  benign_success_before: number;
  benign_success_after: number;
}

export interface RetrievedChunk {
  text: string;
  source: string;
  score: number;
}

export interface ToolCall {
  tool: string;
  arguments: Record<string, string>;
}

export interface ToolExecutionResult {
  tool: string;
  arguments: Record<string, string>;
  result: string;
  permission: string;
}

export interface TestRecord {
  test_id: string;
  category: string;
  target: string;
  prompt: string;
  response: string;
  retrieved_context: RetrievedChunk[] | null;
  requested_tool_call: ToolCall | null;
  executed_tool_call: ToolExecutionResult | null;
  passed: boolean;
  reason: string;
  latency_ms: number;
  guardrails_enabled: boolean;
  block_reason: string | null;
}

export type RunName = "baseline" | "guarded";
