/**
 * Op-level WebGPU vs CPU comparison for the debug GPU harness.
 *
 * The patched wllama links llama.cpp's test-backend-ops (patch 0016) with an
 * op suite of the harness models (flag `--sakutto-plamo3`; PLaMo 3 Translate
 * 0.8B from patch 0015, Qwen3 0.6B and Gemma 3 1B from patch 0020): the weight
 * matmul shapes, F16 attention matmuls, norms, RoPE, softmax, flash attention,
 * GLU etc. the models actually run, each evaluated on WebGPU and
 * on the CPU reference. Every compared node prints a `SAKUTTO_OPSTAT` line and
 * every case a `SAKUTTO_OPRESULT` line; this module turns those lines into
 * rows. Thresholds are test-backend-ops' own per-op NMSE limits, and the
 * status also covers its NaN / Inf / sentinel (out-of-bounds write) checks.
 *
 * Each group runs in its own throwaway worker (a fresh WebGPU device), so an
 * op that loses the device only takes its own group down.
 */

export interface OpGroup {
  id: string;
  label: string;
  /** test-backend-ops `-o` filter (op name of the case's output node). */
  op: string;
  /** Optional test-backend-ops `-p` regex on the case parameters. */
  params?: string;
}

export const PLAMO3_OP_GROUPS: readonly OpGroup[] = [
  { id: 'mul_mat-q6_k', label: 'MUL_MAT Q6_K weights', op: 'MUL_MAT', params: 'type_a=q6_K' },
  { id: 'mul_mat-q4_k', label: 'MUL_MAT Q4_K weights', op: 'MUL_MAT', params: 'type_a=q4_K' },
  {
    id: 'mul_mat-f16',
    label: 'MUL_MAT F16 (attention KQ / KQV)',
    op: 'MUL_MAT',
    params: 'type_a=f16',
  },
  {
    id: 'mul_mat-q5_0',
    label: 'MUL_MAT Q5_0 weights (Gemma 3)',
    op: 'MUL_MAT',
    params: 'type_a=q5_0',
  },
  {
    id: 'mul_mat-q8_0',
    label: 'MUL_MAT Q8_0 weights (Gemma 3)',
    op: 'MUL_MAT',
    params: 'type_a=q8_0',
  },
  { id: 'flash_attn', label: 'FLASH_ATTN_EXT (head dim 128 / 256)', op: 'FLASH_ATTN_EXT' },
  { id: 'soft_max', label: 'SOFT_MAX (masked, scaled, sinks)', op: 'SOFT_MAX' },
  { id: 'rope', label: 'ROPE (NEOX, YaRN / plain)', op: 'ROPE' },
  { id: 'rms_norm', label: 'RMS_NORM', op: 'RMS_NORM' },
  { id: 'add', label: 'ADD (incl. fused RMS_NORM+MUL+ADD)', op: 'ADD' },
  { id: 'mul', label: 'MUL (norm weight broadcast)', op: 'MUL' },
  { id: 'swiglu', label: 'SWIGLU (GLU; packed and split)', op: 'SWIGLU' },
  { id: 'geglu', label: 'GEGLU (GLU, Gemma 3)', op: 'GEGLU' },
  { id: 'get_rows', label: 'GET_ROWS', op: 'GET_ROWS' },
  { id: 'set_rows', label: 'SET_ROWS (KV cache write)', op: 'SET_ROWS' },
  { id: 'cpy', label: 'CPY F32->F16', op: 'CPY' },
  { id: 'cont', label: 'CONT (permuted)', op: 'CONT' },
  { id: 'scale', label: 'SCALE', op: 'SCALE' },
];

/**
 * `backend` limits the run to one ggml device by name. Unset tests every non-CPU device
 * (only WebGPU in this build); 'CPU' compares the optimized CPU kernels with the CPU
 * reference, a self-check of the suite that works without a usable GPU.
 */
export function opGroupArgs(group: OpGroup, backend?: string): string[] {
  const args = ['test', '--sakutto-plamo3', '-o', group.op];
  if (group.params) args.push('-p', group.params);
  if (backend) args.push('-b', backend);
  return args;
}

/**
 * test-backend-ops lines worth keeping in the harness log: which devices it found and
 * skipped, and why a device could not be used.
 */
export function isOpDiagnosticContextLine(line: string): boolean {
  return /Backend \d+\/\d+|Skipping|Device lost|device ready|lacks shader-f16|no adapter|giving up|error:|abort/iu.test(
    line,
  );
}

export type OpCaseStatus = 'OK' | 'FAIL' | 'NOT_SUPPORTED' | 'INCOMPLETE';

export interface OpNodeStat {
  node: string;
  n: number;
  nmse: number;
  threshold: number;
  maxAbs: number;
  nanGpu: number;
  nanCpu: number;
  infGpu: number;
  infCpu: number;
}

export interface OpCaseRow {
  op: string;
  vars: string;
  status: OpCaseStatus;
  /** Worst compared node of the case (largest NMSE relative to its threshold). */
  worst: OpNodeStat | null;
  nodes: OpNodeStat[];
}

const STAT_TAG = 'SAKUTTO_OPSTAT\t';
const RESULT_TAG = 'SAKUTTO_OPRESULT\t';

function field(parts: readonly string[], name: string): number {
  const prefix = `${name}=`;
  const raw = parts.find((part) => part.startsWith(prefix));
  if (raw === undefined) return Number.NaN;
  const value = raw.slice(prefix.length).trim().toLowerCase();
  if (value === 'inf' || value === '+inf') return Number.POSITIVE_INFINITY;
  if (value === '-inf') return Number.NEGATIVE_INFINITY;
  return Number(value);
}

function severity(stat: OpNodeStat): number {
  if (stat.nanGpu + stat.infGpu > stat.nanCpu + stat.infCpu) return Number.POSITIVE_INFINITY;
  if (!Number.isFinite(stat.nmse)) return Number.POSITIVE_INFINITY;
  return stat.threshold > 0 ? stat.nmse / stat.threshold : stat.nmse;
}

/**
 * Parse test-backend-ops output lines (in order). Stat lines belong to the next
 * result line; stats left without a result (the device was lost or the module
 * aborted mid-case) become an INCOMPLETE row.
 */
export function parseOpDiagnosticLines(lines: readonly string[]): OpCaseRow[] {
  const rows: OpCaseRow[] = [];
  let pending: { op: string; vars: string; nodes: OpNodeStat[] } | null = null;

  const flushIncomplete = (): void => {
    if (!pending) return;
    rows.push(finish(pending.op, pending.vars, 'INCOMPLETE', pending.nodes));
    pending = null;
  };

  for (const line of lines) {
    const statAt = line.indexOf(STAT_TAG);
    if (statAt >= 0) {
      const parts = line.slice(statAt + STAT_TAG.length).split('\t');
      const [op = '', node = '', vars = ''] = parts;
      const stat: OpNodeStat = {
        node,
        n: field(parts, 'n'),
        nmse: field(parts, 'nmse'),
        threshold: field(parts, 'thr'),
        maxAbs: field(parts, 'maxabs'),
        nanGpu: field(parts, 'nan_gpu'),
        nanCpu: field(parts, 'nan_cpu'),
        infGpu: field(parts, 'inf_gpu'),
        infCpu: field(parts, 'inf_cpu'),
      };
      if (pending && (pending.op !== op || pending.vars !== vars)) flushIncomplete();
      pending ??= { op, vars, nodes: [] };
      pending.nodes.push(stat);
      continue;
    }
    const resultAt = line.indexOf(RESULT_TAG);
    if (resultAt >= 0) {
      const [status = '', op = '', vars = ''] = line
        .slice(resultAt + RESULT_TAG.length)
        .split('\t')
        .map((part) => part.trim());
      const nodes = pending && pending.op === op && pending.vars === vars ? pending.nodes : [];
      if (pending && nodes.length === 0) flushIncomplete();
      pending = null;
      rows.push(finish(op, vars, toStatus(status), nodes));
    }
  }
  flushIncomplete();
  return rows;
}

function toStatus(raw: string): OpCaseStatus {
  return raw === 'OK' || raw === 'FAIL' || raw === 'NOT_SUPPORTED' ? raw : 'INCOMPLETE';
}

function finish(op: string, vars: string, status: OpCaseStatus, nodes: OpNodeStat[]): OpCaseRow {
  let worst: OpNodeStat | null = null;
  for (const node of nodes) {
    if (!worst || severity(node) > severity(worst)) worst = node;
  }
  return { op, vars, status, worst, nodes };
}

export interface OpGroupOutcome {
  group: OpGroup;
  rows: OpCaseRow[];
  /** testBackendOps rejected (device lost, abort), timed out, or the module is missing. */
  error: string | null;
  retcode: number | null;
  /** ggml-webgpu debug profile the group ran under; absent for the default. */
  profile?: string;
}

/**
 * A group whose worker aborts (device lost, wasm abort) may never answer: wllama's
 * testBackendOps then never settles. Each group gets this long before the runner records it
 * as a crash and moves on to the next group, so one bad group cannot end the run.
 */
export const OP_GROUP_TIMEOUT_MS = 180_000;

export class OpGroupTimeoutError extends Error {
  override readonly name = 'OpGroupTimeoutError';
}

/** Settle with `promise`, or reject with OpGroupTimeoutError after `ms`. */
export function withOpGroupTimeout<T>(
  promise: Promise<T>,
  ms: number = OP_GROUP_TIMEOUT_MS,
  timers: {
    set: (fn: () => void, ms: number) => unknown;
    clear: (id: unknown) => void;
  } = {
    set: (fn, delay) => setTimeout(fn, delay),
    clear: (id) => clearTimeout(id as ReturnType<typeof setTimeout>),
  },
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = timers.set(() => {
      reject(
        new OpGroupTimeoutError(
          `no answer after ${Math.round(ms / 1000)} s (worker hung: device lost or wasm abort)`,
        ),
      );
    }, ms);
    promise.then(
      (value) => {
        timers.clear(id);
        resolve(value);
      },
      (error: unknown) => {
        timers.clear(id);
        reject(error instanceof Error ? error : new Error(String(error)));
      },
    );
  });
}

/** The MUL_MAT groups, run under every Adreno profile by "Run MUL_MAT × Adreno profiles". */
export const MUL_MAT_GROUP_IDS: readonly string[] = [
  'mul_mat-q6_k',
  'mul_mat-q4_k',
  'mul_mat-f16',
  'mul_mat-q5_0',
  'mul_mat-q8_0',
];

/**
 * Profiles for the Adreno MUL_MAT matrix run (patch 0017): the default (subgroups off, mat-vec
 * on), upstream behaviour, subgroups on with mat-vec routed to the tiled matmul, and the
 * previous default (no-matvec on Adreno: subgroups off, mat-vec routed to the tiled matmul).
 */
export const ADRENO_OP_PROFILES: ReadonlyArray<string | null> = [
  null,
  'adreno-subgroups',
  'adreno-sg-tiled',
  'no-matvec',
];

/**
 * Profiles with their own "Run all ops" button: the Adreno default spelled out (every op
 * group the models use, with subgroups off), and subgroups on with mat-vec routed to the
 * tiled matmul, to find which other subgroup shader garbles Gemma 3 on Adreno.
 */
export const ALL_OPS_PROFILES: readonly string[] = ['adreno-matvec', 'adreno-sg-tiled'];

export interface OpGroupSummary {
  ok: number;
  fail: number;
  unsupported: number;
  incomplete: number;
  verdict: 'PASS' | 'FAIL' | 'CRASH' | 'NO-DATA';
}

export function summarizeOpGroup(outcome: OpGroupOutcome): OpGroupSummary {
  const count = (status: OpCaseStatus): number =>
    outcome.rows.filter((row) => row.status === status).length;
  const ok = count('OK');
  const fail = count('FAIL');
  const unsupported = count('NOT_SUPPORTED');
  const incomplete = count('INCOMPLETE');
  let verdict: OpGroupSummary['verdict'];
  if (fail > 0) verdict = 'FAIL';
  else if (outcome.error !== null || incomplete > 0) verdict = 'CRASH';
  else if (ok === 0) verdict = 'NO-DATA';
  else verdict = 'PASS';
  return { ok, fail, unsupported, incomplete, verdict };
}

export function formatNumber(value: number): string {
  if (Number.isNaN(value)) return 'n/a';
  if (!Number.isFinite(value)) return value > 0 ? 'inf' : '-inf';
  if (value === 0) return '0';
  return Math.abs(value) >= 1e-3 && Math.abs(value) < 1e4
    ? value.toPrecision(4)
    : value.toExponential(2);
}

export function formatOpRow(row: OpCaseRow): string {
  const w = row.worst;
  const stats = w
    ? ` nmse=${formatNumber(w.nmse)} (thr ${formatNumber(w.threshold)}) maxabs=${formatNumber(w.maxAbs)}` +
      ` nan gpu/cpu=${w.nanGpu}/${w.nanCpu} inf gpu/cpu=${w.infGpu}/${w.infCpu}`
    : '';
  return `${row.status.padEnd(13)} ${row.op}(${row.vars})${stats}`;
}
