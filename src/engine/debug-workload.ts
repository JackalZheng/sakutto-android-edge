import type { PromptFormat } from './model/catalog';
import { InBrowserEngine } from './in-browser/engine';
import type { CompletionRequest, LlamaRuntime } from './in-browser/runtime';
import type { TranslateSegment } from './types';
import type { WllamaCompletionTimings } from './wllama/module';

/**
 * Title vs paragraph workload (debug harness, 1.3.0).
 *
 * On the POCO F5 Pro, WebGPU translated titles much faster than the CPU while a paragraph was
 * only a little faster. This runs the production pipeline (`InBrowserEngine`: planner, PLaMo line
 * packing, scheduler waves on the runtime's slots, prompt cache as in production) on a page-like
 * list of short titles and on one long paragraph, and reports the wall time plus every request
 * the engine sent (start, ttft, prompt / decode tokens and ms), so the CPU and WebGPU configs
 * (and their slot counts) can be compared on the phone.
 *
 * Two sets of each text: set A warms the runtime (shader compile, first allocations), set B is
 * the measured run. Different text, so the CPU's prompt cache cannot reuse the warm-up prompt.
 */

export type WorkloadKind = 'titles-packed' | 'titles-each' | 'paragraph';

export const WORKLOAD_SOURCE_LANGUAGE = 'en';
export const WORKLOAD_TARGET_LANGUAGE = 'zh-TW';

export const WORKLOAD_TITLES_A: readonly string[] = [
  'Local elections draw record turnout',
  'New bridge opens after six years of work',
  'How to keep houseplants alive in winter',
  'Central bank holds rates steady',
  'Five easy weeknight dinners',
  'Storm warning issued for the east coast',
  'Museum returns stolen paintings',
  'Why sleep matters more than you think',
  'City council approves bike lane plan',
  'Tech shares slip as earnings disappoint',
  'A guide to the best hiking trails',
  'Researchers map the ocean floor',
];

export const WORKLOAD_TITLES_B: readonly string[] = [
  'Train fares to rise next spring',
  'School lunch program expands statewide',
  'The quiet return of film cameras',
  'Wildfire smoke reaches the capital',
  'Ten tips for a cheaper holiday',
  'Football club names new coach',
  'Hospital opens a children’s wing',
  'What the new tax rules mean for you',
  'Coffee prices hit a ten-year high',
  'Volunteers clean up the river banks',
  'Startup unveils a foldable e-bike',
  'Scientists spot a comet near Mars',
];

export const WORKLOAD_PARAGRAPH_A =
  'The city library reopened on Monday after a two-year renovation that added a rooftop ' +
  'reading garden, a makerspace with 3D printers, and quiet study rooms that can be booked ' +
  'online. Officials said the project finished slightly under budget because volunteers ' +
  'helped move more than two hundred thousand books, and the old card catalog was kept as a ' +
  'display in the lobby. Opening hours have been extended until ten in the evening on weekdays.';

export const WORKLOAD_PARAGRAPH_B =
  'A small bakery on the corner of Fifth Street has become an unexpected tourist stop after a ' +
  'video of its owner shaping bread at dawn was shared millions of times. The owner, who opened ' +
  'the shop thirty years ago with a single oven, now hires four more bakers and still sells the ' +
  'morning loaves at the same price. Neighbours say the queue starts before six, and the shop ' +
  'closes as soon as the last tray is gone, usually before noon.';

export interface WorkloadRequest {
  /** ms from the start of the run until the engine sent this request to the runtime. */
  startMs: number;
  /** ms from sending to the first streamed token (null: nothing streamed). */
  ttftMs: number | null;
  totalMs: number;
  promptTokens: number | null;
  promptMs: number | null;
  predictedTokens: number | null;
  predictedMs: number | null;
  /** Length of the prompt sent (a packed PLaMo request carries several titles). */
  promptChars: number;
}

export interface WorkloadRun {
  kind: WorkloadKind;
  segments: number;
  wallMs: number;
  /** ms until the first segment was final (what the reader sees first). */
  firstFinalMs: number | null;
  requests: WorkloadRequest[];
  /** Most requests running in the runtime at once (slot use). */
  maxConcurrent: number;
  outputs: string[];
  error: string | null;
}

export interface WorkloadOptions {
  format: PromptFormat;
  /** n_ctx / n_parallel, as `LazyInBrowserEngine` gives the engine. */
  contextTokens: number;
  now?: () => number;
}

function segmentsFor(kind: WorkloadKind, set: 'A' | 'B'): TranslateSegment[] {
  if (kind === 'paragraph') {
    return [{ id: `p-${set}`, text: set === 'A' ? WORKLOAD_PARAGRAPH_A : WORKLOAD_PARAGRAPH_B }];
  }
  const titles = set === 'A' ? WORKLOAD_TITLES_A : WORKLOAD_TITLES_B;
  return titles.map((text, index) => ({ id: `t-${set}-${index}`, text }));
}

/** Wraps a runtime so every completion the engine sends is timed (llama.cpp timings too). */
export function instrumentRuntime(
  runtime: LlamaRuntime,
  started: number,
  sink: WorkloadRequest[],
  concurrency: { now: number; max: number },
  now: () => number,
): LlamaRuntime {
  return {
    backend: runtime.backend,
    threads: runtime.threads,
    parallel: runtime.parallel,
    dispose: () => runtime.dispose(),
    async complete(request: CompletionRequest): Promise<void> {
      const sent = now();
      let firstAt: number | null = null;
      let timings: WllamaCompletionTimings | null = null;
      concurrency.now += 1;
      concurrency.max = Math.max(concurrency.max, concurrency.now);
      try {
        await runtime.complete({
          ...request,
          onToken: (text) => {
            if (firstAt === null) firstAt = now();
            request.onToken(text);
          },
          onTimings: (value) => {
            timings = value;
            request.onTimings?.(value);
          },
        });
      } finally {
        concurrency.now -= 1;
        const t = timings as WllamaCompletionTimings | null;
        sink.push({
          startMs: Math.round(sent - started),
          ttftMs: firstAt === null ? null : Math.round(firstAt - sent),
          totalMs: Math.round(now() - sent),
          promptTokens: t?.prompt_n ?? null,
          promptMs: t?.prompt_ms ?? null,
          predictedTokens: t?.predicted_n ?? null,
          predictedMs: t?.predicted_ms ?? null,
          promptChars: request.prompt.length,
        });
      }
    },
  };
}

/**
 * One workload through the production engine. `titles-packed` sends the titles as one call, as
 * page blocks that are not headings (PLaMo packs up to 8 short lines per request).
 * `titles-each` sends one call per title at once, as headings (H1-H6) are sent by the page:
 * never packed, and up to 32 calls in flight on the GPU tier (`GPU_TRANSLATION_CONCURRENCY`)
 * but 2 on the CPU tiers (`TRANSLATION_CONCURRENCY`).
 */
export async function runWorkload(
  runtime: LlamaRuntime,
  kind: WorkloadKind,
  set: 'A' | 'B',
  options: WorkloadOptions,
): Promise<WorkloadRun> {
  const now = options.now ?? (() => performance.now());
  const segments = segmentsFor(kind, set);
  const requests: WorkloadRequest[] = [];
  const concurrency = { now: 0, max: 0 };
  const started = now();
  const engine = new InBrowserEngine(
    instrumentRuntime(runtime, started, requests, concurrency, now),
    { format: options.format, contextTokens: options.contextTokens },
  );
  const outputs = new Map<string, string>();
  let firstFinalMs: number | null = null;
  let error: string | null = null;
  const translateOptions = {
    src: WORKLOAD_SOURCE_LANGUAGE,
    tgt: WORKLOAD_TARGET_LANGUAGE,
    form: 'plain' as const,
    priority: 0,
    signal: new AbortController().signal,
  };
  const consume = async (batch: readonly TranslateSegment[], eager: boolean): Promise<void> => {
    for await (const chunk of engine.translate(batch, { ...translateOptions, eager })) {
      if (chunk.final === undefined) continue;
      outputs.set(chunk.id, chunk.final);
      if (firstFinalMs === null) firstFinalMs = Math.round(now() - started);
    }
  };
  try {
    if (kind === 'titles-each') {
      // The page's CPU tiers keep 2 calls in flight, the GPU tier up to 32.
      const inFlight = runtime.backend === 'webgpu' ? 32 : 2;
      let next = 0;
      const lane = async (): Promise<void> => {
        while (next < segments.length) {
          const segment = segments[next++]!;
          await consume([segment], false);
        }
      };
      await Promise.all(Array.from({ length: Math.min(inFlight, segments.length) }, lane));
    } else {
      // A page's first request of a run is eager (its first short block alone).
      await consume(segments, kind === 'titles-packed');
    }
  } catch (caught) {
    error = caught instanceof Error ? caught.message : String(caught);
  }
  requests.sort((a, b) => a.startMs - b.startMs);
  return {
    kind,
    segments: segments.length,
    wallMs: Math.round(now() - started),
    firstFinalMs,
    requests,
    maxConcurrent: concurrency.max,
    outputs: segments.map((segment) => outputs.get(segment.id) ?? ''),
    error,
  };
}

/** Six workload runs per config; a large model on a phone CPU needs more than the perf 5 min. */
export const WORKLOAD_CONFIG_TIMEOUT_MS = 600_000;

export const WORKLOAD_KINDS: readonly WorkloadKind[] = [
  'titles-packed',
  'titles-each',
  'paragraph',
];

/** Configs the comparison runs, one per page load (ids from `buildPerfConfigs`). */
export const WORKLOAD_CONFIG_IDS: readonly string[] = [
  'cpu',
  'cpu-2-slot',
  'gpu-r6-r4-1-slot',
  'gpu-r6-r4',
  'gpu-r6-r4-4-slot',
];

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((a, b) => a + b, 0) / values.length;
}

function rate(tokens: number, ms: number): string {
  return ms > 0 ? (tokens / (ms / 1000)).toFixed(1) : 'n/a';
}

export interface WorkloadSummary {
  requests: number;
  ttftAvgMs: number | null;
  ttftMaxMs: number | null;
  promptTokens: number;
  promptMs: number;
  predictedTokens: number;
  predictedMs: number;
}

export function summarizeWorkload(run: WorkloadRun): WorkloadSummary {
  const ttft = run.requests.map((r) => r.ttftMs).filter((v): v is number => v !== null);
  const sum = (pick: (r: WorkloadRequest) => number | null): number =>
    run.requests.reduce((total, r) => total + (pick(r) ?? 0), 0);
  const avg = mean(ttft);
  return {
    requests: run.requests.length,
    ttftAvgMs: avg === null ? null : Math.round(avg),
    ttftMaxMs: ttft.length === 0 ? null : Math.max(...ttft),
    promptTokens: sum((r) => r.promptTokens),
    promptMs: Math.round(sum((r) => r.promptMs)),
    predictedTokens: sum((r) => r.predictedTokens),
    predictedMs: Math.round(sum((r) => r.predictedMs)),
  };
}

/** Lines for one config's measured runs (set B). */
export function formatWorkloadRuns(runs: readonly WorkloadRun[]): string[] {
  const lines: string[] = [];
  for (const run of runs) {
    const s = summarizeWorkload(run);
    lines.push(
      `  ${run.kind} (${run.segments} segment${run.segments === 1 ? '' : 's'}): wall ${run.wallMs} ms, ` +
        `first final ${run.firstFinalMs ?? 'n/a'} ms, ${s.requests} requests, max ${run.maxConcurrent} at once, ` +
        `ttft avg ${s.ttftAvgMs ?? 'n/a'} / max ${s.ttftMaxMs ?? 'n/a'} ms, ` +
        `prompt ${s.promptTokens} tok ${s.promptMs} ms (${rate(s.promptTokens, s.promptMs)} t/s), ` +
        `decode ${s.predictedTokens} tok ${s.predictedMs} ms (${rate(s.predictedTokens, s.predictedMs)} t/s)` +
        (run.error ? `  ERROR: ${run.error}` : ''),
    );
    for (const r of run.requests) {
      lines.push(
        `    req @${r.startMs} ms: ttft ${r.ttftMs ?? 'n/a'} ms, total ${r.totalMs} ms, ` +
          `prompt ${r.promptTokens ?? 'n/a'} tok ${r.promptMs === null ? 'n/a' : Math.round(r.promptMs)} ms, ` +
          `decode ${r.predictedTokens ?? 'n/a'} tok ${r.predictedMs === null ? 'n/a' : Math.round(r.predictedMs)} ms`,
      );
    }
    const sample = run.outputs.find((o) => o.trim().length > 0);
    if (sample) lines.push(`    output: ${JSON.stringify(sample.slice(0, 60))}`);
  }
  return lines;
}

/** The comparison table: one row per config, wall ms per workload (set B). */
export function formatWorkloadTable(
  rows: readonly {
    configId: string;
    runs: readonly WorkloadRun[] | undefined;
    error?: string | null;
  }[],
): string[] {
  const header =
    '  config               ' +
    WORKLOAD_KINDS.map((kind) => `${kind} wall/reqs/ttft`.padStart(30)).join('');
  const lines = ['title vs paragraph (set B, warm; wall ms / requests / avg ttft ms):', header];
  for (const row of rows) {
    const cells = WORKLOAD_KINDS.map((kind) => {
      const run = row.runs?.find((r) => r.kind === kind);
      if (!run) return 'n/a'.padStart(30);
      const s = summarizeWorkload(run);
      return `${run.wallMs} / ${s.requests} / ${s.ttftAvgMs ?? 'n/a'}`.padStart(30);
    });
    lines.push(
      `  ${row.configId.padEnd(20)} ${cells.join('')}${row.error ? `  FAILED: ${row.error.split('\n')[0]!.slice(0, 100)}` : ''}`,
    );
  }
  return lines;
}
