import { MOBILE_MAX_BATCH, MOBILE_MAX_THREADS } from '@/shared/mobile';
import type { AdrenoGpuProfile } from '@/shared/settings';

export type ComputeTier = 'g' | 'c8' | 'c4' | 'c1';

export type BackendKind = 'webgpu' | 'cpu-wasm';

export const CONTEXT_TOKENS = 2048;

export const WEBGPU_ALL_LAYERS = 99999;

/**
 * A WebGPU decode step costs about the same for one token as for many, so sequences advance
 * together. The scheduler starts them in waves, because a prompt that joins mid-wave makes
 * every step of the sequences already generating as slow as a prefill.
 *
 * The width depends on ggml's integer shaders, which only some GPUs get. Without them, up to
 * four rows still use the vector shader and a fifth falls to the tiled one. Long article on
 * an RTX 4090 with the integer shaders forced off: 1 slot 8.1 s, 4 slots 3.7 s, 16 slots 5.2 s.
 * Sixteen slots are kept to the GPUs they have been measured on; see `supportsFastBatches`.
 */
const WEBGPU_SLOTS_FAST = 16;
const WEBGPU_SLOTS_PLAIN = 4;
/** One sequence: smaller KV, used after a wide WebGPU load failed. */
const WEBGPU_SLOTS_CONSERVATIVE = 1;
// llama.cpp divides n_ctx across parallel slots. A piece of text fits well inside 1024.
const WEBGPU_CONTEXT_PER_SLOT = 1024;

export interface BackendProbe {
  webgpu: boolean;
  hardwareConcurrency: number;
  /** The GPU has been measured at the full width. Any other GPU gets the narrower one. */
  fastBatches?: boolean;
  /**
   * SharedArrayBuffer / wasm threads. Firefox extension pages are never
   * `crossOriginIsolated`, so this is false there and CPU is one thread.
   */
  sharedMemory?: boolean;
  /** The WebGPU adapter is a Qualcomm Adreno (vendor `qualcomm` or architecture `adreno-*`). */
  adreno?: boolean;
}

/**
 * The WebGPU setup on an Adreno GPU (settings: "Adreno GPU profile"). ggml-webgpu applies the
 * Adreno defaults by itself (subgroups off, mat-vec workgroup 64, the new SOFT_MAX, 128-wide row
 * reductions, 16 kernels per submit, batched mat-vec for 5-64-token prompts); the profile only
 * picks what differs:
 * - 'optimized': those defaults;
 * - 'compat': the round-4 setup measured on the phone (no batched mat-vec, `bmv0`);
 * - 'hybrid': the defaults with about half the layers on the GPU, the rest on the CPU.
 */
export type { AdrenoGpuProfile };

/**
 * Adreno: two slots of 1024 tokens (n_ctx 2048). The phone sweep showed no speed difference
 * between n_ctx 1024-4096 and 1-4 slots, so the smaller KV cache wins.
 */
const WEBGPU_SLOTS_ADRENO = 2;

export interface DebugLoadOverrides {
  /** false turns flash attention off; unset keeps llama.cpp's auto choice. */
  flashAttn?: boolean;
  /** Keep the KV cache (and so attention) on the CPU while the weights are on the GPU. */
  noKvOffload?: boolean;
  /** llama.cpp `-ot` rules, e.g. `blk\.(7|15)\.=CPU`. */
  tensorOverrides?: string;
  /** llama.cpp `n_ubatch` (physical batch); unset keeps llama.cpp's default (min(n_batch, 512)). */
  nUbatch?: number;
}

export interface BackendSelection {
  tier: ComputeTier;
  backend: BackendKind;
  nGpuLayers: number;
  nThreads: number;
  nParallel: number;
  nCtx: number;
  /** Prompt batch size passed to llama.cpp; unset keeps the loader default. */
  nBatch?: number;
  sharedMemory?: boolean;
  /** Debug harness only: extra llama.cpp load options. Production selections never set it. */
  debugLoad?: DebugLoadOverrides;
  /**
   * ggml-webgpu tuning tokens for this load (see `composeWebGpuDebugString`). Set on Adreno
   * WebGPU selections (an empty list clears an earlier profile); unset leaves them alone.
   */
  webgpuTuning?: readonly string[];
  /** Hybrid: put about half of the model's layers on the GPU (resolved per model at load). */
  gpuLayersHalf?: boolean;
}

function clampThreads(requested: number, hardwareConcurrency: number): number {
  const cores = Number.isFinite(hardwareConcurrency) ? Math.floor(hardwareConcurrency) : 4;
  return Math.max(1, Math.min(requested, Math.max(1, cores)));
}

/**
 * Half the logical cores, which is the physical core count on SMT CPUs, between 8 and 16.
 * On a 32-thread desktop, 16 threads decoded 60 tokens/s where 8 decoded 43 (SIMD build,
 * Q4_K_M), so a large CPU is not held to eight. A machine with 16 or fewer threads keeps 8.
 */
function largeCpuThreads(hardwareConcurrency: number): number {
  const cores = Number.isFinite(hardwareConcurrency) ? Math.floor(hardwareConcurrency) : 4;
  return clampThreads(Math.min(16, Math.max(8, Math.floor(cores / 2))), cores);
}

function withMemory(selection: BackendSelection, probe: BackendProbe): BackendSelection {
  return probe.sharedMemory === false ? { ...selection, sharedMemory: false } : selection;
}

function singleThreadCpu(probe: BackendProbe): BackendSelection {
  return withMemory(
    {
      tier: 'c1',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: 1,
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    },
    probe,
  );
}

/** The CPU tiers keep one sequence: there, each sequence costs its full compute. */
export function selectBackend(probe: BackendProbe): BackendSelection {
  if (probe.sharedMemory === false && !probe.webgpu) {
    return singleThreadCpu(probe);
  }

  if (probe.webgpu) {
    const slots = probe.fastBatches === true ? WEBGPU_SLOTS_FAST : WEBGPU_SLOTS_PLAIN;
    return withMemory(
      {
        tier: 'g',
        backend: 'webgpu',
        nGpuLayers: WEBGPU_ALL_LAYERS,
        nThreads: probe.sharedMemory === false ? 1 : clampThreads(8, probe.hardwareConcurrency),
        nParallel: slots,
        nCtx: WEBGPU_CONTEXT_PER_SLOT * slots,
      },
      probe,
    );
  }

  if (probe.hardwareConcurrency >= 8) {
    return withMemory(
      {
        tier: 'c8',
        backend: 'cpu-wasm',
        nGpuLayers: 0,
        nThreads: largeCpuThreads(probe.hardwareConcurrency),
        nParallel: 1,
        nCtx: CONTEXT_TOKENS,
      },
      probe,
    );
  }

  return withMemory(
    {
      tier: 'c4',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: clampThreads(4, probe.hardwareConcurrency),
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    },
    probe,
  );
}

export type TierPreference = 'auto' | ComputeTier;

/** Resolves the settings choice. `auto` keeps the probe; the others force that tier. */
export function selectionForTier(
  preference: TierPreference,
  probe: BackendProbe,
): BackendSelection {
  if (preference === 'auto') return selectBackend(probe);
  if (preference === 'g') {
    return selectBackend({ ...probe, webgpu: true });
  }
  if (preference === 'c8') {
    if (probe.sharedMemory === false) return singleThreadCpu(probe);
    return withMemory(
      {
        tier: 'c8',
        backend: 'cpu-wasm',
        nGpuLayers: 0,
        nThreads: largeCpuThreads(probe.hardwareConcurrency),
        nParallel: 1,
        nCtx: CONTEXT_TOKENS,
      },
      probe,
    );
  }

  if (preference === 'c1' || probe.sharedMemory === false) {
    return singleThreadCpu(probe);
  }

  return withMemory(
    {
      tier: 'c4',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: clampThreads(4, probe.hardwareConcurrency),
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    },
    probe,
  );
}

export function cpuFallback(selection: BackendSelection): BackendSelection | null {
  if (selection.backend === 'cpu-wasm') return null;
  return selectBackend({
    webgpu: false,
    hardwareConcurrency: selection.nThreads,
    ...(selection.sharedMemory === undefined ? {} : { sharedMemory: selection.sharedMemory }),
  });
}

/**
 * The phone caps for an Edge for Android runtime: a phone's small cores and shared
 * memory budget cannot use the desktop thread count, and a smaller prompt batch keeps
 * first inference responsive. The parallel width is untouched — a WebGPU phone still
 * translates on all its slots.
 */
export function applyMobileLimits(selection: BackendSelection): BackendSelection {
  return {
    ...selection,
    nThreads: Math.min(selection.nThreads, MOBILE_MAX_THREADS),
    nBatch: MOBILE_MAX_BATCH,
  };
}

export interface RuntimeContext {
  /** Edge for Android: phone caps, and WebGPU only on an explicit choice. */
  android: boolean;
  /** WebGPU produced broken output on this device before; see `gpu-output.ts`. */
  gpuOutputBroken?: boolean;
  /** The WebGPU adapter is an Adreno: "auto" stays on the CPU, and `gpuProfile` applies. */
  adreno?: boolean;
  /** Settings: the Adreno WebGPU profile (default 'optimized'). */
  gpuProfile?: AdrenoGpuProfile;
}

/**
 * The Adreno WebGPU selection for a profile: two slots, the profile's tuning tokens, and for
 * 'hybrid' half the layers. Other selections are returned unchanged.
 */
export function applyAdrenoProfile(
  selection: BackendSelection,
  profile: AdrenoGpuProfile = 'optimized',
): BackendSelection {
  if (selection.backend !== 'webgpu') return selection;
  const slots = Math.min(selection.nParallel, WEBGPU_SLOTS_ADRENO);
  const next: BackendSelection = {
    ...selection,
    nParallel: slots,
    nCtx: WEBGPU_CONTEXT_PER_SLOT * slots,
    webgpuTuning: profile === 'compat' ? ['bmv0'] : [],
  };
  if (profile === 'hybrid') next.gpuLayersHalf = true;
  else delete next.gpuLayersHalf;
  return next;
}

/** `n_gpu_layers` for a hybrid selection: the output layer plus the last half of the blocks. */
export function resolveGpuLayers(
  selection: BackendSelection,
  modelLayers: number | null,
): BackendSelection {
  if (selection.backend !== 'webgpu' || selection.gpuLayersHalf !== true) return selection;
  if (modelLayers === null || modelLayers < 2) return selection;
  return { ...selection, nGpuLayers: Math.floor(modelLayers / 2) + 1 };
}

/**
 * The selection for a settings choice on this runtime.
 *
 * On Edge for Android, "auto" never picks WebGPU. llama.cpp's WebGPU backend returned
 * garbage on an Adreno 730 (POCO F5 Pro) while the CPU path translated correctly, and a
 * phone cannot tell a wrong GPU from a right one before it has translated something. WebGPU
 * stays available as an explicit choice, unless it already produced broken output here.
 *
 * On an Adreno GPU (phone or Windows on Snapdragon), "auto" also stays on the CPU: on the POCO F5
 * Pro the fixed WebGPU path translated correctly but the CPU was still faster (PLaMo 3 0.8B,
 * total 2.52 s CPU, 3.77 s full GPU, 3.28 s hybrid). An explicit WebGPU choice runs the Adreno
 * profile from the settings (`applyAdrenoProfile`).
 */
export function resolveSelection(
  preference: TierPreference,
  probe: BackendProbe,
  context: RuntimeContext,
): BackendSelection {
  const adreno = context.adreno === true || probe.adreno === true;
  const adrenoProfile = (selection: BackendSelection): BackendSelection =>
    adreno ? applyAdrenoProfile(selection, context.gpuProfile) : selection;
  if (!context.android) {
    // Auto stays on the CPU on Adreno (faster on the phones measured so far), and so does an
    // explicit GPU choice once its output came out garbled (cleared by choosing GPU again).
    if (adreno && (preference === 'auto' || (preference === 'g' && context.gpuOutputBroken))) {
      return selectionForTier('auto', { ...probe, webgpu: false });
    }
    return adrenoProfile(selectionForTier(preference, probe));
  }
  const cpuOnly = preference === 'auto' || (preference === 'g' && context.gpuOutputBroken === true);
  const selection = cpuOnly
    ? selectionForTier('auto', { ...probe, webgpu: false })
    : adrenoProfile(selectionForTier(preference, probe));
  return applyMobileLimits(selection);
}

/** Where to run after WebGPU produced broken output: the CPU, with the phone caps. */
export function cpuAfterBrokenOutput(
  selection: BackendSelection,
  context: RuntimeContext,
): BackendSelection {
  const cpu = cpuFallback(selection) ?? selection;
  return context.android ? applyMobileLimits(cpu) : cpu;
}

/**
 * True when a WebGPU load ended on the CPU. A manual CPU choice is the same backend on
 * both sides, so it must not be reported as "WebGPU could not load the model".
 */
export function isGpuLoadFallback(requested: BackendSelection, loaded: BackendSelection): boolean {
  return requested.backend === 'webgpu' && loaded.backend === 'cpu-wasm';
}

/**
 * The next lighter place to run after `selection` failed: four slots, then one
 * conservative WebGPU slot, then the CPU.
 */
export function lighterSelection(selection: BackendSelection): BackendSelection | null {
  if (selection.backend === 'webgpu' && selection.nParallel > WEBGPU_SLOTS_PLAIN) {
    return {
      ...selection,
      nParallel: WEBGPU_SLOTS_PLAIN,
      nCtx: WEBGPU_CONTEXT_PER_SLOT * WEBGPU_SLOTS_PLAIN,
    };
  }
  if (selection.backend === 'webgpu' && selection.nParallel > WEBGPU_SLOTS_CONSERVATIVE) {
    return {
      ...selection,
      nParallel: WEBGPU_SLOTS_CONSERVATIVE,
      nCtx: WEBGPU_CONTEXT_PER_SLOT * WEBGPU_SLOTS_CONSERVATIVE,
    };
  }
  return cpuFallback(selection);
}

/** Losses further apart than this are unrelated, such as a driver update or a resume from sleep. */
export const GPU_LOSS_WINDOW_MS = 120_000;

/** A lone GPU loss reloads the same way. A second soon after means this load is too heavy. */
export function selectionAfterGpuLoss(
  selection: BackendSelection,
  sinceLastLossMs: number,
): BackendSelection {
  if (sinceLastLossMs >= GPU_LOSS_WINDOW_MS) return selection;
  return lighterSelection(selection) ?? selection;
}
