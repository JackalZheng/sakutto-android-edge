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
 * and, if not, which knob (slots, context size, shader-f16) changes it.
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
   * Only 'no-f16' is used: the patched ggml treats the adapter as lacking
   * shader-f16, refuses the device, and the load degrades to CPU inside the
   * worker — the harness reports that through the load log's GPU evidence.
   */
  debugProfile: 'no-f16' | null;
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
