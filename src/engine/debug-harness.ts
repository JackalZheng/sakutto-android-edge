import {
  applyMobileLimits,
  selectionForTier,
  type BackendProbe,
  type BackendSelection,
} from './in-browser/backend';
import { looksLikeBrokenTranslation } from './text/garbage';
import type { GpuDiagnostic, WebGpuDebugProfile } from '@/shared/gpu-diagnostic';

/**
 * Stage 1 debug harness for the Android WebGPU garbage-output investigation.
 *
 * One fixed prompt is run through the CPU backend (the reference) and through
 * several WebGPU configurations. The phone's job is to produce the numbers
 * that decide whether the llama.cpp bump fixed the Adreno/Vulkan corruption
 * and, if not, which knob (slots, context size, shader-f16, subgroups,
 * offloaded layers) changes it.
 *
 * Stage 2 (the bump did not fix it — the device run showed CPU correct and
 * three WebGPU configs the identical deterministic garble from the first
 * token) adds: a no-subgroups profile (patch 0013), partial-offload
 * comparisons (n_gpu_layers 1 and 8), and a CPU fallback for the no-f16
 * config so it reports an outcome instead of an error.
 *
 * This module is the pure part: configuration building and output comparison.
 * The page that drives the real runtimes lives in
 * `src/entrypoints/gpu-harness/main.ts`.
 */

/** The sample page's first paragraph; long enough to trip the garbage check. */
export const HARNESS_SOURCE_TEXT =
  'The morning train leaves the quiet station while the baker opens a small shop on the corner.';
export const HARNESS_SOURCE_LANGUAGE = 'en';
export const HARNESS_TARGET_LANGUAGE = 'zh-hant';
export const HARNESS_MAX_TOKENS = 128;

/**
 * PLaMo 3 Translate 0.8B layer layout (GGUF `plamo3.block_count` 16,
 * `attention.sliding_window` 2048, `sliding_window_pattern` 8): every layer is
 * attention + SwiGLU FFN (no Mamba/SSM blocks); layers 7 and 15 use full
 * attention (RoPE base 1e6 + YaRN), the other 14 use sliding-window attention
 * (RoPE base 1e4). llama.cpp offloads from the END: `n_gpu_layers` N puts the
 * output layer plus repeating layers 16-(N-1)..15 on the GPU, so layers-2 is
 * layer 15 alone (full attention) and layers-3 adds layer 14 (SWA).
 */
export const PLAMO3_LAYER_COUNT = 16;
export const PLAMO3_FULL_ATTENTION_LAYERS: readonly number[] = [7, 15];

export function plamo3SwaLayers(): number[] {
  return Array.from({ length: PLAMO3_LAYER_COUNT }, (_, i) => i).filter(
    (i) => !PLAMO3_FULL_ATTENTION_LAYERS.includes(i),
  );
}

/** A llama.cpp `-ot` rule that keeps every tensor of the given layers on the CPU. */
export function layersOnCpuOverride(layers: readonly number[]): string {
  return `blk\\.(${layers.join('|')})\\.=CPU`;
}

/** Repeating layers that land on the GPU for a given n_gpu_layers (output layer excluded). */
export function offloadedRepeatingLayers(
  nGpuLayers: number,
  layerCount = PLAMO3_LAYER_COUNT,
): number[] {
  const repeating = Math.max(0, Math.min(layerCount, nGpuLayers - 1));
  return Array.from({ length: repeating }, (_, i) => layerCount - repeating + i);
}

export interface HarnessConfig {
  id: string;
  label: string;
  selection: BackendSelection;
  /**
   * ggml debug profile to set on the harness page before this load, or null.
   * 'no-f16': the patched ggml treats the adapter as lacking shader-f16,
   * refuses the device, and the load degrades to CPU inside the worker.
   * 'no-subgroups' (patch 0013): the patched ggml requests a device without
   * the Subgroups feature, so the workgroup-reduction shaders run.
   */
  debugProfile: Exclude<WebGpuDebugProfile, 'off'> | null;
  /**
   * When the WebGPU load refuses the device (EngineError 'gpu-unavailable'),
   * rerun this config on the CPU selection and report the fallback outcome
   * instead of a load error. Only the no-f16 config sets this: its point is
   * to show the model still translates without shader-f16.
   */
  fallbackToCpu?: boolean;
  /**
   * The config is built from the PLaMo 3 0.8B layer layout (tensor overrides
   * by layer index); the model switcher hides it for other models.
   */
  plamoOnly?: true;
}

export function buildHarnessConfigs(probe: BackendProbe, android: boolean): HarnessConfig[] {
  const limit = (selection: BackendSelection): BackendSelection =>
    android ? applyMobileLimits(selection) : selection;

  const cpu = limit(selectionForTier('auto', { ...probe, webgpu: false }));
  const configs: HarnessConfig[] = [
    { id: 'cpu', label: 'CPU (reference)', selection: cpu, debugProfile: null },
  ];

  if (!probe.webgpu) return configs;

  const gpu = limit(selectionForTier('g', probe));
  configs.push(
    {
      id: 'gpu-default',
      label:
        'WebGPU (default; Adreno: subgroups off, mat-vec WG 64, new soft_max, 16 kernels per submit)',
      selection: gpu,
      debugProfile: null,
    },
    {
      id: 'gpu-1-slot',
      label: 'WebGPU (1 slot, n_ctx 1024)',
      selection: { ...gpu, nParallel: 1, nCtx: 1024 },
      debugProfile: null,
    },
    {
      id: 'gpu-nctx-2048',
      label: 'WebGPU (n_ctx 2048)',
      selection: { ...gpu, nCtx: 2048 },
      debugProfile: null,
    },
    {
      id: 'gpu-no-f16',
      label: 'WebGPU (no shader-f16 profile)',
      selection: gpu,
      debugProfile: 'no-f16',
      fallbackToCpu: true,
    },
    {
      id: 'gpu-no-subgroups',
      label: 'WebGPU (no subgroups profile)',
      selection: gpu,
      debugProfile: 'no-subgroups',
    },
    // Patch 0017 profiles. On Adreno the default is subgroups off, mat-vec on and 16 kernels
    // per queue submit; these turn each part back (no effect on other GPUs).
    {
      id: 'gpu-adreno-upstream',
      label: 'WebGPU (Adreno: upstream subgroups + mat-vec; expected to garble)',
      selection: gpu,
      debugProfile: 'adreno-subgroups',
    },
    {
      id: 'gpu-adreno-matvec',
      label: 'WebGPU (Adreno: subgroups off, mat-vec on; = Adreno default)',
      selection: gpu,
      debugProfile: 'adreno-matvec',
    },
    {
      id: 'gpu-adreno-sg-tiled',
      label: 'WebGPU (Adreno: subgroups on, mat-vec routed to tiled matmul)',
      selection: gpu,
      debugProfile: 'adreno-sg-tiled',
    },
    {
      id: 'gpu-adreno-tiled',
      label: 'WebGPU (Adreno: previous default, mat-vec routed to tiled matmul; 16 per submit)',
      selection: gpu,
      debugProfile: 'no-matvec',
    },
    {
      id: 'gpu-adreno-submit-64',
      label: 'WebGPU (Adreno default with 64 kernels per submit, as upstream)',
      selection: gpu,
      debugProfile: 'adreno-submit-64',
    },
    // Device-lost bisection on Adreno (PLaMo 3 / Qwen3), all on the mat-vec default.
    {
      id: 'gpu-matvec-1-slot',
      label: 'WebGPU (Adreno mat-vec; 1 slot, n_ctx 1024)',
      selection: { ...gpu, nParallel: 1, nCtx: 1024 },
      debugProfile: 'adreno-matvec',
    },
    {
      id: 'gpu-matvec-no-fa',
      label: 'WebGPU (Adreno mat-vec; flash attention off)',
      selection: { ...gpu, debugLoad: { flashAttn: false } },
      debugProfile: 'adreno-matvec',
    },
    ...[64, 32].map((nBatch): HarnessConfig => ({
      id: `gpu-matvec-small-batch-${nBatch}`,
      label: `WebGPU (Adreno mat-vec; n_batch ${nBatch})`,
      selection: { ...gpu, nBatch },
      debugProfile: 'adreno-matvec',
    })),
    {
      id: 'gpu-layers-1',
      label: 'WebGPU (1 GPU layer)',
      selection: { ...gpu, nGpuLayers: 1 },
      debugProfile: null,
    },
    ...[2, 3, 4].map((layers): HarnessConfig => ({
      id: `gpu-layers-${layers}`,
      label: `WebGPU (${layers} GPU layers: output + last ${layers - 1}; PLaMo 3: layer ${offloadedRepeatingLayers(layers).join(',')})`,
      selection: { ...gpu, nGpuLayers: layers },
      debugProfile: null,
    })),
    {
      id: 'gpu-layers-8',
      label: 'WebGPU (8 GPU layers)',
      selection: { ...gpu, nGpuLayers: 8 },
      debugProfile: null,
    },
    {
      id: 'gpu-no-flash-attn',
      label: 'WebGPU (flash attention off)',
      selection: { ...gpu, debugLoad: { flashAttn: false } },
      debugProfile: null,
    },
    {
      id: 'gpu-kv-on-cpu',
      label: 'WebGPU (weights on GPU, KV cache + attention on CPU)',
      selection: { ...gpu, debugLoad: { noKvOffload: true } },
      debugProfile: null,
    },
    {
      id: 'gpu-swa-layers-only',
      label: `WebGPU (only SWA layers' weights on GPU; layers ${PLAMO3_FULL_ATTENTION_LAYERS.join(',')} on CPU)`,
      selection: {
        ...gpu,
        debugLoad: { tensorOverrides: layersOnCpuOverride(PLAMO3_FULL_ATTENTION_LAYERS) },
      },
      debugProfile: null,
      plamoOnly: true,
    },
    {
      id: 'gpu-full-attn-layers-only',
      label: `WebGPU (only full-attention layers ${PLAMO3_FULL_ATTENTION_LAYERS.join(',')} weights on GPU)`,
      selection: { ...gpu, debugLoad: { tensorOverrides: layersOnCpuOverride(plamo3SwaLayers()) } },
      debugProfile: null,
      plamoOnly: true,
    },
  );
  return configs;
}

export interface HarnessRunResult {
  configId: string;
  /** Harness model the run used (model switcher); absent in results saved before it. */
  modelId?: string;
  ok: boolean;
  output: string;
  /** First streamed chunk; the closest thing to a first-token comparison the runtime exposes. */
  firstToken: string;
  ttftMs: number | null;
  totalMs: number | null;
  error: string | null;
  /** From the load log after a GPU load: did a GPU device actually engage? */
  gpuEvidence: 'engaged' | 'missing' | 'unknown' | null;
  /** ggml's own correctness-probe verdict from the load log, when present. */
  correctnessProbe: string | null;
  /** Subgroup size the correctness probe measured on the device, when reported. */
  correctnessProbeSubgroupSize: number | null;
  /** Correctness-probe duration from the load log, when reported. */
  correctnessProbeMs: number | null;
  /** True when the run fell back to the CPU after WebGPU refused the device. */
  fellBackToCpu: boolean;
}

export function firstStreamedToken(chunks: readonly string[]): string {
  for (const chunk of chunks) {
    if (chunk.length > 0) return chunk;
  }
  return '';
}

export interface RunComparison {
  firstTokenMatches: boolean;
  outputsIdentical: boolean;
  cpuLooksBroken: boolean;
  gpuLooksBroken: boolean;
}

export function compareRuns(cpu: HarnessRunResult, gpu: HarnessRunResult): RunComparison {
  return {
    firstTokenMatches: cpu.firstToken === gpu.firstToken,
    outputsIdentical: cpu.output === gpu.output,
    cpuLooksBroken: looksLikeBrokenTranslation(
      HARNESS_SOURCE_TEXT,
      cpu.output,
      HARNESS_TARGET_LANGUAGE,
    ),
    gpuLooksBroken: looksLikeBrokenTranslation(
      HARNESS_SOURCE_TEXT,
      gpu.output,
      HARNESS_TARGET_LANGUAGE,
    ),
  };
}

export interface RunVerdict {
  pass: boolean;
  reasons: string[];
}

/**
 * The harness verdict for one GPU config. Deliberately based ONLY on the
 * real output compared with the CPU reference — never on ggml's built-in
 * correctness probe: on the owner's POCO F5 Pro that probe reports "ok"
 * (42.4 ms, subgroup_size 128) while real translation output is garbled,
 * because the probe's add/subgroup kernels do not cover the failing
 * Q4_K dequant / mul_mat path. Probe fields are printed for the record.
 */
export function verdictForRun(cpu: HarnessRunResult, gpu: HarnessRunResult): RunVerdict {
  const reasons: string[] = [];
  if (!gpu.ok) reasons.push(`run failed: ${gpu.error ?? 'unknown error'}`);
  const comparison = compareRuns(cpu, gpu);
  if (!comparison.firstTokenMatches) reasons.push('first token differs from the CPU reference');
  if (comparison.gpuLooksBroken) {
    reasons.push('output flagged as broken by the mixed-script/length check');
  }
  return { pass: reasons.length === 0, reasons };
}

/**
 * The owner's POCO F5 Pro WebGPU diagnostic (Sakutto settings diagnostic,
 * 2026-11-21), kept as the reference the harness compares a run against
 * field by field, so a phone run can be diffed against the known device
 * line by line. Note the subgroup width: 64–128 on this Adreno, where
 * NVIDIA reports 32 — ggml's subgroup reductions are sized from these
 * numbers, and a wrong width is a prime garbage-output suspect.
 */
export const POCO_F5_PRO_REFERENCE = {
  vendor: 'qualcomm',
  architecture: 'adreno-7xx',
  fallbackAdapter: false,
  shaderF16: true,
  subgroups: true,
  subgroupMinSize: 64,
  subgroupMaxSize: 128,
  maxBufferSize: 1073741824,
  maxStorageBufferBindingSize: 134217728,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 1024,
  adapterAttempt: 'high-performance',
} as const;

export interface AdapterFieldComparison {
  field: string;
  reference: string;
  actual: string;
  matches: boolean;
}

export function compareAdapterToReference(
  diagnostic: GpuDiagnostic,
  reference: typeof POCO_F5_PRO_REFERENCE = POCO_F5_PRO_REFERENCE,
): AdapterFieldComparison[] {
  const rows: Array<readonly [string, unknown, unknown]> = [
    ['vendor', reference.vendor, diagnostic.vendor],
    ['architecture', reference.architecture, diagnostic.architecture],
    ['fallbackAdapter', reference.fallbackAdapter, diagnostic.fallbackAdapter],
    ['shaderF16', reference.shaderF16, diagnostic.shaderF16],
    ['subgroups', reference.subgroups, diagnostic.subgroups],
    ['subgroupMinSize', reference.subgroupMinSize, diagnostic.subgroupMinSize],
    ['subgroupMaxSize', reference.subgroupMaxSize, diagnostic.subgroupMaxSize],
    ['maxBufferSize', reference.maxBufferSize, diagnostic.maxBufferSize],
    [
      'maxStorageBufferBindingSize',
      reference.maxStorageBufferBindingSize,
      diagnostic.maxStorageBufferBindingSize,
    ],
    [
      'maxComputeWorkgroupStorageSize',
      reference.maxComputeWorkgroupStorageSize,
      diagnostic.maxComputeWorkgroupStorageSize,
    ],
    [
      'maxComputeInvocationsPerWorkgroup',
      reference.maxComputeInvocationsPerWorkgroup,
      diagnostic.maxComputeInvocationsPerWorkgroup,
    ],
    ['adapterAttempts', reference.adapterAttempt, diagnostic.adapterAttempts.join(', ')],
  ];
  return rows.map(([field, ref, actual]) => ({
    field,
    reference: String(ref),
    actual: actual === null || actual === undefined ? 'n/a' : String(actual),
    // Attempt entries carry outcome text; the reference is the attempt name.
    matches: field === 'adapterAttempts' ? String(actual).includes(String(ref)) : ref === actual,
  }));
}

/**
 * "Run isolated": one config per page load. The page stores the queue and the
 * results in sessionStorage, runs the head of the queue in a fresh page (new
 * JS realm, new wllama worker, new WebGPU device), saves the result and
 * reloads, so nothing a previous config did to the device or the module can
 * leak into the next one.
 */
export const ISOLATED_STATE_KEY = 'sakuttoGpuHarnessIsolated';
/** Pause before the next isolated config so a lost device can be torn down. */
export const ISOLATED_RELOAD_DELAY_MS = 1500;
const ISOLATED_LOG_LIMIT = 200_000;

export interface IsolatedRunState {
  version: 1;
  queue: string[];
  results: HarnessRunResult[];
  log: string;
  /** Model the queued configs run with (model switcher). */
  modelId?: string;
}

export function emptyIsolatedState(): IsolatedRunState {
  return { version: 1, queue: [], results: [], log: '' };
}

/**
 * Queue for an isolated run of `requested` config ids: unknown ids are dropped,
 * duplicates removed, and the CPU reference is put first unless a CPU result
 * is already stored (every verdict needs it).
 */
export function planIsolatedQueue(
  configIds: readonly string[],
  requested: readonly string[],
  state: IsolatedRunState,
  modelId?: string,
): string[] {
  const known = new Set(configIds);
  const queue: string[] = [];
  for (const id of requested) {
    if (known.has(id) && !queue.includes(id)) queue.push(id);
  }
  const haveCpu = state.results.some((r) => r.configId === 'cpu' && r.ok && r.modelId === modelId);
  const withoutCpu = queue.filter((id) => id !== 'cpu');
  return haveCpu && !requested.includes('cpu') ? withoutCpu : ['cpu', ...withoutCpu];
}

export function parseIsolatedState(raw: string | null): IsolatedRunState | null {
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<IsolatedRunState> | null;
    if (
      !value ||
      value.version !== 1 ||
      !Array.isArray(value.queue) ||
      !Array.isArray(value.results) ||
      typeof value.log !== 'string'
    ) {
      return null;
    }
    return {
      version: 1,
      queue: value.queue.filter((id): id is string => typeof id === 'string'),
      results: value.results,
      log: value.log,
      ...(typeof value.modelId === 'string' ? { modelId: value.modelId } : {}),
    };
  } catch {
    return null;
  }
}

/** Store a finished config: drop it from the queue, replace any older result for it. */
export function recordIsolatedResult(
  state: IsolatedRunState,
  result: HarnessRunResult,
  logChunk: string,
): IsolatedRunState {
  const log = `${state.log}${logChunk}`;
  return {
    ...state,
    version: 1,
    queue: state.queue.filter((id, index) => !(index === 0 && id === result.configId)),
    results: [...state.results.filter((r) => !sameRun(r, result.configId, result.modelId)), result],
    log: log.length > ISOLATED_LOG_LIMIT ? log.slice(log.length - ISOLATED_LOG_LIMIT) : log,
  };
}

/** Whether `result` is the run of `configId` with `modelId`. */
export function sameRun(result: HarnessRunResult, configId: string, modelId?: string): boolean {
  return result.configId === configId && result.modelId === modelId;
}

/** Error text including the cause chain (the wasm abort message lives in the cause). */
export function describeErrorChain(error: unknown): string {
  const parts: string[] = [];
  let current: unknown = error;
  for (let depth = 0; depth < 4 && current !== undefined && current !== null; depth += 1) {
    if (current instanceof Error) {
      const detail = (current as { detail?: unknown }).detail;
      parts.push(
        `${current.name}: ${current.message}${typeof detail === 'string' && detail ? ` [${detail}]` : ''}`,
      );
      current = (current as { cause?: unknown }).cause;
    } else {
      parts.push(String(current));
      break;
    }
  }
  return parts.join(' <- ');
}
