import {
  applyMobileLimits,
  selectionForTier,
  type BackendProbe,
  type BackendSelection,
} from './in-browser/backend';
import { composeWebGpuDebugString, type WebGpuDebugProfile } from '@/shared/gpu-diagnostic';

/**
 * GPU perf breakdown and tuning sweep for the debug harness (pure part).
 *
 * Every config runs in a fresh page (fresh worker and WebGPU device): load the model, run the
 * harness prompt twice with the prompt cache off (run 1 includes shader/pipeline compilation,
 * run 2 is warm), and read llama.cpp's own timings plus ggml-webgpu's SAKUTTO_GPUSTAT /
 * SAKUTTO_GPUOPS lines (wllama patch 0021, enabled by the `perf` / `perfops` tuning tokens).
 * The page that drives it lives in `src/entrypoints/gpu-harness/main.ts`.
 */

export const PERF_STATE_KEY = 'sakuttoGpuPerfSweep';
/** Per config: load + two runs; the per-op run submits and waits after every op. */
export const PERF_CONFIG_TIMEOUT_MS = 300_000;
/** Suffix of the per-op variant of a config (adds the `perfops` token). */
export const PERF_OPS_SUFFIX = '#ops';

/** Repeating-layer counts of the harness models, for the half-offload config. */
export const HARNESS_MODEL_LAYERS: Readonly<Record<string, number>> = {
  'plamo-3-translate-0.8b-q4_k_m': 16,
  'qwen3-0.6b-q4_k_m': 28,
  'qwen3-1.7b-q4_k_m': 28,
  'gemma-3-1b-it-q4_k_m': 26,
};

export interface PerfConfig {
  id: string;
  label: string;
  selection: BackendSelection;
  debugProfile: Exclude<WebGpuDebugProfile, 'off'> | null;
  /** ggml-webgpu tuning tokens (patch 0021); `perf` is added to every WebGPU config. */
  tuning: readonly string[];
}

export function buildPerfConfigs(
  probe: BackendProbe,
  android: boolean,
  modelLayers: number | null,
): PerfConfig[] {
  const limit = (selection: BackendSelection): BackendSelection =>
    android ? applyMobileLimits(selection) : selection;
  const cpu = limit(selectionForTier('auto', { ...probe, webgpu: false }));
  const configs: PerfConfig[] = [
    {
      id: 'cpu',
      label: `CPU baseline (${cpu.nThreads} threads)`,
      selection: cpu,
      debugProfile: null,
      tuning: [],
    },
  ];
  if (!probe.webgpu) return configs;
  const gpu = limit(selectionForTier('g', probe));
  const mv = (
    id: string,
    label: string,
    selection: BackendSelection,
    tuning: readonly string[] = [],
  ): PerfConfig => ({
    id,
    label,
    selection,
    debugProfile: 'adreno-matvec',
    tuning: [...tuning, 'perf'],
  });
  configs.push(
    mv('gpu-mv', 'adreno-matvec (Adreno default: 16 per submit)', gpu),
    mv('gpu-mv-sub32', 'adreno-matvec, 32 kernels per submit', gpu, ['sub32']),
    mv('gpu-mv-sub64', 'adreno-matvec, 64 kernels per submit', gpu, ['sub64']),
    mv('gpu-mv-1-slot', 'adreno-matvec, 1 slot, n_ctx 1024', { ...gpu, nParallel: 1, nCtx: 1024 }),
    mv('gpu-mv-b512-ub256', 'adreno-matvec, n_batch 512 / n_ubatch 256', {
      ...gpu,
      nBatch: 512,
      debugLoad: { ...gpu.debugLoad, nUbatch: 256 },
    }),
    mv('gpu-mv-b512-ub512', 'adreno-matvec, n_batch 512 / n_ubatch 512', {
      ...gpu,
      nBatch: 512,
      debugLoad: { ...gpu.debugLoad, nUbatch: 512 },
    }),
    mv('gpu-mv-wg64', 'adreno-matvec, mat-vec workgroup 64', gpu, ['mvwg64']),
    mv('gpu-mv-wg128-o8', 'adreno-matvec, mat-vec workgroup 128, 8 outputs/WG', gpu, [
      'mvwg128',
      'mvo8',
    ]),
  );
  if (modelLayers !== null && modelLayers > 1) {
    const layers = Math.floor(modelLayers / 2) + 1;
    configs.push(
      mv(
        'gpu-mv-half-layers',
        `adreno-matvec, hybrid: n_gpu_layers ${layers} (output + last ${layers - 1} of ${modelLayers} layers on GPU)`,
        { ...gpu, nGpuLayers: layers },
      ),
    );
  }
  configs.push(
    mv(
      'gpu-mv-sub64-1-slot',
      'adreno-matvec, 64 per submit + 1 slot, n_ctx 1024',
      {
        ...gpu,
        nParallel: 1,
        nCtx: 1024,
      },
      ['sub64'],
    ),
  );
  return configs;
}

/** A config id, or its per-op variant (`<id>#ops`), resolved to the config to run. */
export function resolvePerfConfig(configs: readonly PerfConfig[], id: string): PerfConfig | null {
  const opsVariant = id.endsWith(PERF_OPS_SUFFIX);
  const base = configs.find(
    (c) => c.id === (opsVariant ? id.slice(0, -PERF_OPS_SUFFIX.length) : id),
  );
  if (!base) return null;
  if (!opsVariant) return base;
  if (base.selection.backend !== 'webgpu') return null;
  return {
    ...base,
    id,
    label: `${base.label} + per-op timing (submit+wait after every op; slow, for the op table only)`,
    tuning: [...base.tuning.filter((t) => t !== 'perf'), 'perfops'],
  };
}

/** The sweep: every config, then the per-op run of the Adreno default. */
export function perfSweepQueue(configs: readonly PerfConfig[]): string[] {
  const ids = configs.map((c) => c.id);
  if (ids.includes('gpu-mv')) ids.push(`gpu-mv${PERF_OPS_SUFFIX}`);
  return ids;
}

/** The breakdown: the chosen config, then its per-op variant when it runs on WebGPU. */
export function perfBreakdownQueue(configs: readonly PerfConfig[], id: string): string[] {
  const config = configs.find((c) => c.id === id);
  if (!config) return [];
  return config.selection.backend === 'webgpu' ? [id, `${id}${PERF_OPS_SUFFIX}`] : [id];
}

export function perfDebugString(config: PerfConfig): string | null {
  return composeWebGpuDebugString(config.debugProfile, config.tuning);
}

// --- ggml-webgpu stat lines (patch 0021) --------------------------------------------------

export interface GpuStatLine {
  graph: number;
  /** Largest MUL_MAT `ne[1]` in the graph: the tokens in the batch. */
  n: number;
  nodes: number;
  kernels: number;
  submits: number;
  encodeMs: number;
  /** Start of this graph minus start of the previous one. */
  cycleMs: number;
  /** Time spent waiting in buffer maps since the previous graph. */
  readbackMs: number;
  readbacks: number;
}

export interface GpuOpsLine {
  graph: number;
  n: number;
  opsMs: number;
  syncMs: number;
  ops: Array<{ key: string; ms: number }>;
}

function numberField(line: string, name: string): number | null {
  const match = new RegExp(`(?:^|\\s)${name}=(-?[0-9.]+)`, 'u').exec(line);
  return match ? Number(match[1]) : null;
}

export function parseGpuStatLine(line: string): GpuStatLine | null {
  const at = line.indexOf('SAKUTTO_GPUSTAT ');
  if (at < 0) return null;
  const rest = line.slice(at);
  const fields = [
    'g',
    'n',
    'nodes',
    'kernels',
    'submits',
    'encode_ms',
    'cycle_ms',
    'readback_ms',
    'readbacks',
  ].map((name) => numberField(rest, name));
  if (fields.some((v) => v === null || Number.isNaN(v))) return null;
  const [graph, n, nodes, kernels, submits, encodeMs, cycleMs, readbackMs, readbacks] = fields as [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  return { graph, n, nodes, kernels, submits, encodeMs, cycleMs, readbackMs, readbacks };
}

export function parseGpuOpsLine(line: string): GpuOpsLine | null {
  const at = line.indexOf('SAKUTTO_GPUOPS ');
  if (at < 0) return null;
  const rest = line.slice(at).trim();
  const graph = numberField(rest, 'g');
  const n = numberField(rest, 'n');
  const opsMs = numberField(rest, 'ops_ms');
  const syncMs = numberField(rest, 'sync_ms');
  if (graph === null || n === null || opsMs === null || syncMs === null) return null;
  const parts = rest.split(/\s+/u);
  const list = parts.length > 5 ? parts.slice(5).join(' ') : '';
  const ops: Array<{ key: string; ms: number }> = [];
  for (const item of list.split(',')) {
    const eq = item.lastIndexOf('=');
    if (eq <= 0) continue;
    const ms = Number(item.slice(eq + 1));
    if (!Number.isNaN(ms)) ops.push({ key: item.slice(0, eq), ms });
  }
  return { graph, n, opsMs, syncMs, ops };
}

/** Graphs with more tokens than mat-vec handles (n > 4) are prompt eval; the rest decode. */
export const DECODE_MAX_TOKENS = 4;

export interface GpuStatSummary {
  promptGraphs: number;
  promptEncodeMs: number;
  decodeGraphs: number;
  /** Per decode token (graphs whose previous graph was also decode). */
  decodeTokens: number;
  kernelsPerToken: number;
  submitsPerToken: number;
  readbacksPerToken: number;
  cycleMs: number;
  encodeMs: number;
  readbackWaitMs: number;
  /** cycle − encode − readback wait: sampling, CPU-side ops (e.g. an output layer over 128 MiB), JS. */
  otherMs: number;
}

function mean(values: readonly number[]): number {
  return values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length;
}

export function summarizeGpuStats(lines: readonly GpuStatLine[]): GpuStatSummary | null {
  if (lines.length === 0) return null;
  const prompt = lines.filter((l) => l.n > DECODE_MAX_TOKENS);
  const decode = lines.filter((l) => l.n <= DECODE_MAX_TOKENS);
  // cycle/readback of graph i belong to graph i-1: count pairs of consecutive decode graphs.
  const pairs: Array<{ prev: GpuStatLine; cur: GpuStatLine }> = [];
  for (let i = 1; i < lines.length; i += 1) {
    const prev = lines[i - 1]!;
    const cur = lines[i]!;
    if (prev.n <= DECODE_MAX_TOKENS && cur.n <= DECODE_MAX_TOKENS && cur.graph === prev.graph + 1) {
      pairs.push({ prev, cur });
    }
  }
  const cycleMs = mean(pairs.map((p) => p.cur.cycleMs));
  const encodeMs = mean(pairs.map((p) => p.prev.encodeMs));
  const readbackWaitMs = mean(pairs.map((p) => p.cur.readbackMs));
  return {
    promptGraphs: prompt.length,
    promptEncodeMs: prompt.reduce((a, l) => a + l.encodeMs, 0),
    decodeGraphs: decode.length,
    decodeTokens: pairs.length,
    kernelsPerToken: mean(decode.map((l) => l.kernels)),
    submitsPerToken: mean(decode.map((l) => l.submits)),
    readbacksPerToken: mean(pairs.map((p) => p.cur.readbacks)),
    cycleMs,
    encodeMs,
    readbackWaitMs,
    otherMs: Math.max(0, cycleMs - encodeMs - readbackWaitMs),
  };
}

export interface GpuOpsSummary {
  decodeGraphs: number;
  promptGraphs: number;
  /** Mean per decode token. */
  opsMs: number;
  syncMs: number;
  ops: Array<{ key: string; ms: number; share: number }>;
  promptOps: Array<{ key: string; ms: number; share: number }>;
}

function aggregateOps(
  lines: readonly GpuOpsLine[],
  divisor: number,
): Array<{ key: string; ms: number; share: number }> {
  const totals = new Map<string, number>();
  for (const line of lines) {
    for (const op of line.ops) totals.set(op.key, (totals.get(op.key) ?? 0) + op.ms);
  }
  const sum = [...totals.values()].reduce((a, b) => a + b, 0);
  return [...totals.entries()]
    .map(([key, ms]) => ({ key, ms: ms / Math.max(1, divisor), share: sum > 0 ? ms / sum : 0 }))
    .sort((a, b) => b.ms - a.ms);
}

export function summarizeGpuOps(lines: readonly GpuOpsLine[]): GpuOpsSummary | null {
  if (lines.length === 0) return null;
  const decode = lines.filter((l) => l.n <= DECODE_MAX_TOKENS);
  const prompt = lines.filter((l) => l.n > DECODE_MAX_TOKENS);
  return {
    decodeGraphs: decode.length,
    promptGraphs: prompt.length,
    opsMs: mean(decode.map((l) => l.opsMs)),
    syncMs: mean(decode.map((l) => l.syncMs)),
    ops: aggregateOps(decode, decode.length),
    promptOps: aggregateOps(prompt, prompt.length),
  };
}

// --- results and report -------------------------------------------------------------------

export interface PerfRunMetrics {
  ttftMs: number | null;
  totalMs: number;
  promptTokens: number | null;
  promptMs: number | null;
  predictedTokens: number | null;
  predictedMs: number | null;
  output: string;
}

export interface PerfResult {
  configId: string;
  modelId: string;
  label: string;
  selectionText: string;
  debugString: string | null;
  ok: boolean;
  error: string | null;
  loadMs: number | null;
  gpuEvidence: string | null;
  runs: PerfRunMetrics[];
  /** From the warm run (run 2) only. */
  gpuStats: GpuStatSummary | null;
  gpuOps: GpuOpsSummary | null;
  /** ggml-webgpu load lines worth seeing (device, tuning, adreno). */
  notes: string[];
}

export function describeSelection(selection: BackendSelection): string {
  return [
    `backend=${selection.backend}`,
    `n_gpu_layers=${selection.nGpuLayers}`,
    `threads=${selection.nThreads}`,
    `n_parallel=${selection.nParallel}`,
    `n_ctx=${selection.nCtx}`,
    `n_batch=${selection.nBatch ?? 'default'}`,
    `n_ubatch=${selection.debugLoad?.nUbatch ?? 'default'}`,
    `flash_attn=${selection.debugLoad?.flashAttn === false ? 'off' : 'auto'}`,
  ].join(' ');
}

function rate(tokens: number | null, ms: number | null): string {
  if (tokens === null || ms === null || ms <= 0) return 'n/a';
  return (tokens / (ms / 1000)).toFixed(2);
}

function fixed(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined || Number.isNaN(value)
    ? 'n/a'
    : value.toFixed(digits);
}

function runLine(label: string, run: PerfRunMetrics): string {
  return (
    `  ${label}: ttft ${run.ttftMs ?? 'n/a'} ms, total ${Math.round(run.totalMs)} ms, ` +
    `prompt ${run.promptTokens ?? 'n/a'} tok ${fixed(run.promptMs, 0)} ms (${rate(run.promptTokens, run.promptMs)} t/s), ` +
    `decode ${run.predictedTokens ?? 'n/a'} tok ${fixed(run.predictedMs, 0)} ms (${rate(run.predictedTokens, run.predictedMs)} t/s)`
  );
}

export function formatPerfResult(result: PerfResult): string[] {
  const lines = [`[${result.configId}] ${result.label}`];
  lines.push(`  ${result.selectionText} debug=${result.debugString ?? 'none'}`);
  if (result.gpuEvidence) lines.push(`  gpu evidence: ${result.gpuEvidence}`);
  for (const note of result.notes) lines.push(`  ${note}`);
  lines.push(`  load: ${result.loadMs === null ? 'n/a' : `${Math.round(result.loadMs)} ms`}`);
  const [cold, warm] = result.runs;
  if (cold) lines.push(runLine('run 1 (cold)', cold));
  if (warm) lines.push(runLine('run 2 (warm)', warm));
  if (cold && warm) {
    const ttft =
      cold.ttftMs !== null && warm.ttftMs !== null ? `${cold.ttftMs - warm.ttftMs} ms` : 'n/a';
    lines.push(
      `  first-run cost (shader/pipeline compile + warmup = run1 − run2): ttft ${ttft}, total ${Math.round(cold.totalMs - warm.totalMs)} ms`,
    );
  }
  const s = result.gpuStats;
  if (s) {
    lines.push(
      `  gpu per decode token (${s.decodeTokens} tokens): cycle ${fixed(s.cycleMs)} ms = ` +
        `encode+submit ${fixed(s.encodeMs)} + readback wait ${fixed(s.readbackWaitMs)} + other ${fixed(s.otherMs)}; ` +
        `kernels ${fixed(s.kernelsPerToken, 0)}, submits ${fixed(s.submitsPerToken, 0)}, readbacks ${fixed(s.readbacksPerToken, 1)}`,
    );
    lines.push(
      `  gpu prompt graphs: ${s.promptGraphs}, encode+submit ${fixed(s.promptEncodeMs)} ms total`,
    );
  }
  const o = result.gpuOps;
  if (o) {
    lines.push(
      `  per-op (decode, ${o.decodeGraphs} tokens; each op submitted alone, includes ~${fixed(o.syncMs, 2)} ms sync each): ` +
        `${fixed(o.opsMs)} ms/token`,
    );
    for (const op of o.ops.slice(0, 16)) {
      lines.push(
        `    ${op.key.padEnd(28)} ${fixed(op.ms, 2).padStart(9)} ms/token ${(op.share * 100).toFixed(1).padStart(5)}%`,
      );
    }
    if (o.promptOps.length > 0) {
      lines.push(`  per-op (prompt, ${o.promptGraphs} graphs, ms per graph):`);
      for (const op of o.promptOps.slice(0, 8)) {
        lines.push(
          `    ${op.key.padEnd(28)} ${fixed(op.ms, 2).padStart(9)} ms ${(op.share * 100).toFixed(1).padStart(5)}%`,
        );
      }
    }
  }
  if (warm ?? cold) lines.push(`  output: ${JSON.stringify((warm ?? cold)!.output.slice(0, 80))}`);
  if (result.error) lines.push(`  ERROR: ${result.error}`);
  return lines;
}

export function formatPerfReport(
  header: readonly string[],
  results: readonly PerfResult[],
): string {
  const lines = ['=== Sakutto GPU perf report ===', ...header, ''];
  lines.push('summary (run 2 = warm; t/s from llama.cpp timings):');
  lines.push('  config               load_ms  ttft_ms  total_ms  prompt_t/s  decode_t/s  ms/token');
  for (const r of results) {
    const warm = r.runs[1] ?? r.runs[0];
    const msPerToken =
      warm?.predictedTokens && warm.predictedMs
        ? (warm.predictedMs / warm.predictedTokens).toFixed(0)
        : 'n/a';
    lines.push(
      `  ${r.configId.padEnd(20)} ${String(r.loadMs === null ? 'n/a' : Math.round(r.loadMs)).padStart(7)}  ` +
        `${String(warm?.ttftMs ?? 'n/a').padStart(7)}  ${String(warm ? Math.round(warm.totalMs) : 'n/a').padStart(8)}  ` +
        `${rate(warm?.promptTokens ?? null, warm?.promptMs ?? null).padStart(10)}  ` +
        `${rate(warm?.predictedTokens ?? null, warm?.predictedMs ?? null).padStart(10)}  ${msPerToken.padStart(8)}` +
        `${r.ok ? '' : `  FAILED: ${(r.error ?? '').split('\n')[0]!.slice(0, 120)}`}`,
    );
  }
  lines.push('');
  for (const r of results) lines.push(...formatPerfResult(r), '');
  return lines.join('\n');
}

// --- isolated sweep state (sessionStorage) -------------------------------------------------

export interface PerfSweepState {
  modelId: string;
  queue: string[];
  /** Set just before a config starts; still set on the next load means the page died. */
  running: string | null;
  results: PerfResult[];
}

export function emptyPerfState(modelId: string): PerfSweepState {
  return { modelId, queue: [], running: null, results: [] };
}

export function parsePerfState(raw: string | null): PerfSweepState | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<PerfSweepState>;
    if (
      typeof value.modelId !== 'string' ||
      !Array.isArray(value.queue) ||
      !Array.isArray(value.results)
    ) {
      return null;
    }
    return {
      modelId: value.modelId,
      queue: value.queue.filter((id): id is string => typeof id === 'string'),
      running: typeof value.running === 'string' ? value.running : null,
      results: value.results as PerfResult[],
    };
  } catch {
    return null;
  }
}

/** A result for a config whose page died (crash, OOM kill, or a timeout reload). */
export function crashedPerfResult(configId: string, modelId: string, reason: string): PerfResult {
  return {
    configId,
    modelId,
    label: configId,
    selectionText: '',
    debugString: null,
    ok: false,
    error: reason,
    loadMs: null,
    gpuEvidence: null,
    runs: [],
    gpuStats: null,
    gpuOps: null,
    notes: [],
  };
}

export function recordPerfResult(state: PerfSweepState, result: PerfResult): PerfSweepState {
  return {
    ...state,
    running: null,
    queue: state.queue.filter((id) => id !== result.configId),
    results: [...state.results.filter((r) => r.configId !== result.configId), result],
  };
}

export class PerfTimeoutError extends Error {
  constructor(ms: number) {
    super(`config did not finish within ${Math.round(ms / 1000)} s`);
    this.name = 'PerfTimeoutError';
  }
}

export function withPerfTimeout<T>(task: Promise<T>, ms = PERF_CONFIG_TIMEOUT_MS): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new PerfTimeoutError(ms)), ms);
  });
  return Promise.race([task, timeout]).finally(() => clearTimeout(timer));
}
