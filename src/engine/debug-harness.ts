import {
  applyMobileLimits,
  selectionForTier,
  type BackendProbe,
  type BackendSelection,
} from './in-browser/backend';
import { looksLikeBrokenTranslation } from './text/garbage';
import type { GpuDiagnostic } from '@/shared/gpu-diagnostic';

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
  debugProfile: 'no-f16' | 'no-subgroups' | null;
  /**
   * When the WebGPU load refuses the device (EngineError 'gpu-unavailable'),
   * rerun this config on the CPU selection and report the fallback outcome
   * instead of a load error. Only the no-f16 config sets this: its point is
   * to show the model still translates without shader-f16.
   */
  fallbackToCpu?: boolean;
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
    { id: 'gpu-default', label: 'WebGPU (default)', selection: gpu, debugProfile: null },
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
    {
      id: 'gpu-layers-1',
      label: 'WebGPU (1 GPU layer)',
      selection: { ...gpu, nGpuLayers: 1 },
      debugProfile: null,
    },
    {
      id: 'gpu-layers-8',
      label: 'WebGPU (8 GPU layers)',
      selection: { ...gpu, nGpuLayers: 8 },
      debugProfile: null,
    },
  );
  return configs;
}

export interface HarnessRunResult {
  configId: string;
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
