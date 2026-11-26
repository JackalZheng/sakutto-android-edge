import {
  ADRENO_R6_TUNING,
  applyAdrenoProfile,
  applyMobileLimits,
  selectionForTier,
  type BackendProbe,
  type BackendSelection,
} from './in-browser/backend';
import { composeWebGpuDebugString, type WebGpuDebugProfile } from '@/shared/gpu-diagnostic';
import { ADRENO_PROFILE_SPECS } from '@/shared/adreno-profiles';
import { MODEL_LIST } from './model/catalog';
import {
  formatWorkloadRuns,
  formatWorkloadTable,
  WORKLOAD_CONFIG_IDS,
  type WorkloadRun,
} from './debug-workload';

function withoutTuning(selection: BackendSelection): BackendSelection {
  const rest = { ...selection };
  delete rest.webgpuTuning;
  return rest;
}

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

/** Repeating-layer counts of the harness models (the catalog), for the half-offload config. */
export const HARNESS_MODEL_LAYERS: Readonly<Record<string, number>> = Object.fromEntries(
  MODEL_LIST.map((model) => [model.id, model.layers]),
);

export interface PerfConfig {
  id: string;
  label: string;
  selection: BackendSelection;
  debugProfile: Exclude<WebGpuDebugProfile, 'off'> | null;
  /** ggml-webgpu tuning tokens (patch 0021); `perf` is added to every WebGPU config. */
  tuning: readonly string[];
  /** false: only runs when picked for a breakdown, never in the sweep (the hybrid configs). */
  sweep?: boolean;
}

/** The Adreno default since 1.3.0 (settings "Optimized", round 6 with 4 rows per group). */
export const PERF_DEFAULT_ID = 'gpu-r6-r4';

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
    {
      // Title vs paragraph comparison only: the CPU with two sequences (each its full compute).
      id: 'cpu-2-slot',
      label: `CPU (${cpu.nThreads} threads) with 2 slots (n_ctx 2048), comparison only`,
      selection: { ...cpu, nParallel: 2, nCtx: 2048 },
      debugProfile: null,
      tuning: [],
      sweep: false,
    },
  ];
  if (!probe.webgpu) return configs;
  // Adreno: the production slot count (1 slot, n_ctx 2048 since 1.3.0, `applyAdrenoProfile`);
  // before 1.3.0 the harness ran the plain WebGPU width (4 slots, n_ctx 4096) here. The tuning
  // comes from each config.
  const gpuBase = selectionForTier('g', probe);
  const gpu = limit(probe.adreno === true ? withoutTuning(applyAdrenoProfile(gpuBase)) : gpuBase);
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
  // Round 5 (patch 0023) is the Adreno default (`gpu-r5`): prompt MUL_MATs with 4 < n <= 64 run
  // through the mat-vec shader in batches of 4 columns, 128-wide SOFT_MAX / fused RMS_NORM
  // workgroups. Round 6 (patch 0025, debug tokens only): `q4v2*` the q4_K one-column mat-vec v2
  // (16-byte weight and activation loads; r<N> rows per thread group, t<N> threads per row),
  // `mvglu` the fused gate+up mat-vec + SWIGLU, `fna` the fused RMS_NORM+MUL+ADD[+RMS_NORM+MUL]
  // residual chain; `gpu-r6` combines them (the round-6 candidate). Box (lavapipe): kernels per
  // decode token 355 -> 277 for PLaMo 3, output identical to the CPU.
  const half = modelLayers !== null && modelLayers > 1 ? Math.floor(modelLayers / 2) + 1 : null;
  const r6 = ['q4v2r2', 'mvglu', 'fna'];
  const r6r4 = [...ADRENO_R6_TUNING];
  configs.push(
    mv(
      'gpu-r6-r4',
      `Adreno default since 1.3.0 ("Optimized", round 6: ${r6r4.join('+')}, 1 slot)`,
      gpu,
      r6r4,
    ),
    mv(
      'gpu-r5',
      'round 5 ("Compatible" profile, 1.3.0 default: batched mat-vec for n <= 64, 128-wide row reductions)',
      gpu,
    ),
    mv('gpu-q4v2', 'round 6: q4_K mat-vec v2 only (16 threads x 1 row per group)', gpu, ['q4v2']),
    mv('gpu-q4v2-r2', 'round 6: q4_K mat-vec v2 only, 2 rows per group', gpu, ['q4v2r2']),
    mv('gpu-q4v2-r4', 'round 6: q4_K mat-vec v2 only, 4 rows per group', gpu, ['q4v2r4']),
    mv('gpu-q4v2-t32', 'round 6: q4_K mat-vec v2 only, 32 threads x 2 rows per group', gpu, [
      'q4v2t32',
      'q4v2r2',
    ]),
    mv('gpu-mvglu', 'round 6: fused gate+up mat-vec + SWIGLU only', gpu, ['mvglu']),
    mv('gpu-fna', 'round 6: fused RMS_NORM+MUL+ADD(+RMS_NORM+MUL) only', gpu, ['fna']),
    mv('gpu-r6', 'round 6 with 2 rows per group: q4_K mat-vec v2 + mvglu + fna', gpu, r6),
    // The other settings profiles (1.3.0), so each one the options page offers can be timed.
    mv(
      'gpu-r6-r4-1-slot',
      '"Low memory" profile: 1.3.0 default with n_ctx 1024 (1 slot)',
      { ...gpu, nParallel: 1, nCtx: 1024 },
      r6r4,
    ),
    mv(
      'gpu-r4',
      `"Round 4" profile: no batched prompt mat-vec, 256-wide row reductions (${ADRENO_PROFILE_SPECS.r4.tuning.join('+')})`,
      gpu,
      ADRENO_PROFILE_SPECS.r4.tuning,
    ),
  );
  if (half !== null) {
    // Breakdown only: slower than full GPU in every phone sweep, and gpu-r6-half once stalled
    // for 41 s; the settings no longer offer a hybrid profile (1.3.1).
    const hybrid = `hybrid: n_gpu_layers ${half} (output + last ${half - 1} of ${modelLayers} layers on GPU)`;
    configs.push(
      { ...mv('gpu-r5-half', `round 5, ${hybrid}`, { ...gpu, nGpuLayers: half }), sweep: false },
      {
        ...mv('gpu-r6-r4-half', `1.3.0 default, ${hybrid}`, { ...gpu, nGpuLayers: half }, r6r4),
        sweep: false,
      },
    );
  }
  configs.push(mv('gpu-r6-r4-repeat', '1.3.0 default again (thermal drift check)', gpu, r6r4));
  // Title vs paragraph comparison only (`WORKLOAD_CONFIG_IDS`): the slot counts the GPU and the
  // CPU could use. Not in the sweep; nothing in the settings uses them (1.3.0).
  configs.push({
    ...mv(
      'gpu-r6-r4-2-slot',
      '1.3.0 default with 2 slots (n_ctx 2048; the Adreno default before 1.3.0), comparison only',
      { ...gpu, nParallel: 2, nCtx: 2048 },
      r6r4,
    ),
    sweep: false,
  });
  configs.push({
    ...mv(
      'gpu-r6-r4-4-slot',
      '1.3.0 default with 4 slots (n_ctx 4096), comparison only',
      { ...gpu, nParallel: 4, nCtx: 4096 },
      r6r4,
    ),
    sweep: false,
  });
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

/**
 * The sweep: every config except the breakdown-only ones. No per-op runs: `perfops` submits and
 * waits after every op, and on the POCO F5 Pro (1.3.1-dev) that run reloaded the page.
 */
export function perfSweepQueue(configs: readonly PerfConfig[]): string[] {
  return configs.filter((c) => c.sweep !== false).map((c) => c.id);
}

/** The title vs paragraph comparison: the configs of `WORKLOAD_CONFIG_IDS` this device has. */
export function workloadQueue(configs: readonly PerfConfig[]): string[] {
  return WORKLOAD_CONFIG_IDS.filter((id) => configs.some((c) => c.id === id));
}

/** The breakdown: the chosen config, then its per-op variant only when asked for (WebGPU only). */
export function perfBreakdownQueue(
  configs: readonly PerfConfig[],
  id: string,
  withOps = false,
): string[] {
  const config = configs.find((c) => c.id === id);
  if (!config) return [];
  return withOps && config.selection.backend === 'webgpu' ? [id, `${id}${PERF_OPS_SUFFIX}`] : [id];
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
  /** Title vs paragraph comparison (`debug-workload.ts`): the measured runs (set B). */
  workload?: WorkloadRun[];
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
  result.runs.forEach((run, index) => {
    if (run.output.trim() === '') {
      lines.push(
        `  WARNING: run ${index + 1} produced no text (${run.predictedTokens ?? 'n/a'} tokens): an end-of-generation token first, ` +
          'or a GPU fault that left the logits empty (see runtime lines)',
      );
    }
  });
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
  if (result.workload) lines.push(...formatWorkloadRuns(result.workload));
  if (warm ?? cold) lines.push(`  output: ${JSON.stringify((warm ?? cold)!.output.slice(0, 80))}`);
  if (result.error) lines.push(`  ERROR: ${result.error}`);
  return lines;
}

export function formatPerfReport(
  header: readonly string[],
  results: readonly PerfResult[],
): string {
  const lines = ['=== Sakutto GPU perf report ===', ...header, ''];
  if (results.some((r) => r.workload !== undefined)) {
    lines.push(
      ...formatWorkloadTable(
        results.map((r) => ({ configId: r.configId, runs: r.workload, error: r.error })),
      ),
      '',
    );
    for (const r of results) lines.push(...formatPerfResult(r), '');
    return lines.join('\n');
  }
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
  /** `workload`: the title vs paragraph comparison instead of the fixed-prompt timing. */
  mode?: 'perf' | 'workload';
  queue: string[];
  /** Set just before a config starts; still set on the next load means the page died. */
  running: string | null;
  /** What the running config got to (load, runs), for a report when its page dies. */
  partial?: PerfResult | null;
  results: PerfResult[];
}

export function emptyPerfState(modelId: string): PerfSweepState {
  return { modelId, mode: 'perf', queue: [], running: null, partial: null, results: [] };
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
      mode: value.mode === 'workload' ? 'workload' : 'perf',
      queue: value.queue.filter((id): id is string => typeof id === 'string'),
      running: typeof value.running === 'string' ? value.running : null,
      partial:
        value.partial && typeof value.partial === 'object' ? (value.partial as PerfResult) : null,
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
    partial: null,
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
