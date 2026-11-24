import { MOBILE_MAX_BATCH, MOBILE_MAX_THREADS } from '@/shared/mobile';
import {
  ADRENO_PROFILE_SPECS,
  ADRENO_R6_TUNING,
  DEFAULT_ADRENO_GPU_PROFILE,
  type AdrenoGpuProfile,
} from '@/shared/adreno-profiles';

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
 * reductions, 16 kernels per submit, batched mat-vec for 5-64-token prompts: round 5); a profile
 * adds tuning tokens on top and may change the slot count (`shared/adreno-profiles.ts`):
 * - 'optimized' (default): round 6 (`ADRENO_R6_TUNING`), the q4_K mat-vec v2 with 4 rows per
 *   thread group + fused gate/up SWIGLU + fused norm chain. POCO F5 Pro (Adreno 730), PLaMo 3
 *   0.8B, warm: total 2.25 s, decode 22.2 t/s vs CPU 2.91 s / 17.3 t/s;
 * - 'q4v2': only the q4_K mat-vec v2 (4 rows), no fusions (2.29 s);
 * - 'r6': round 6 with 2 rows per group (2.54 s, lighter register pressure);
 * - 'low-memory': 'optimized' with one slot (n_ctx 1024), for the > 1 GB models or low-RAM phones;
 * - 'compat': round 5, the 1.3.0-dev default (2.69 s);
 * - 'r4': round 4, no batched prompt mat-vec and 256-wide row reductions.
 */
export type { AdrenoGpuProfile };
export { ADRENO_R6_TUNING };

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
  /** The WebGPU adapter is an Adreno: "auto" uses WebGPU with `gpuProfile`. */
  adreno?: boolean;
  /** Settings: the Adreno WebGPU profile (default 'optimized'). */
  gpuProfile?: AdrenoGpuProfile;
  /**
   * The active model drifts from the CPU result on Adreno WebGPU (Qwen3, Gemma 3; see
   * `ModelEntry.gpuExperimental`): "auto" keeps it on the CPU there.
   */
  gpuExperimental?: boolean;
}

/**
 * The Adreno WebGPU selection for a profile: its slot count (two by default) and its tuning
 * tokens. Other selections are returned unchanged.
 */
export function applyAdrenoProfile(
  selection: BackendSelection,
  profile: AdrenoGpuProfile = DEFAULT_ADRENO_GPU_PROFILE,
): BackendSelection {
  if (selection.backend !== 'webgpu') return selection;
  const spec = ADRENO_PROFILE_SPECS[profile] ?? ADRENO_PROFILE_SPECS[DEFAULT_ADRENO_GPU_PROFILE];
  const slots = Math.max(1, Math.min(selection.nParallel, spec.slots ?? WEBGPU_SLOTS_ADRENO));
  return {
    ...selection,
    nParallel: slots,
    nCtx: WEBGPU_CONTEXT_PER_SLOT * slots,
    webgpuTuning: [...spec.tuning],
  };
}

/**
 * The selection for a settings choice on this runtime.
 *
 * On Edge for Android without an Adreno GPU, "auto" never picks WebGPU. A phone cannot tell a
 * wrong GPU from a right one before it has translated something, and only Adreno has been
 * fixed and measured. WebGPU stays available as an explicit choice.
 *
 * On an Adreno GPU (phone or Windows on Snapdragon), "auto" picks WebGPU with the settings'
 * Adreno profile (`applyAdrenoProfile`): with the round-6 kernels the POCO F5 Pro translated
 * correctly and faster than its CPU (PLaMo 3 0.8B: total 2.25 s vs 2.91 s). The output guard
 * still watches it, and once WebGPU produced broken output here (`gpuOutputBroken`), "auto"
 * and an explicit GPU choice both run on the CPU until GPU is chosen again in the settings
 * (or the extension is updated, see `background.ts`).
 *
 * 1.3.0: with a GPU-experimental model (`gpuExperimental`: Qwen3, Gemma 3, which drift from the
 * CPU result on Adreno), "auto" on Adreno runs on the CPU; an explicit WebGPU choice still runs
 * the Adreno profile so the model can be tested there.
 */
export function resolveSelection(
  preference: TierPreference,
  probe: BackendProbe,
  context: RuntimeContext,
): BackendSelection {
  const adreno = context.adreno === true || probe.adreno === true;
  const broken = context.gpuOutputBroken === true;
  const cpu = (): BackendSelection => selectionForTier('auto', { ...probe, webgpu: false });
  let selection: BackendSelection;
  if (adreno) {
    if (broken && (preference === 'auto' || preference === 'g')) selection = cpu();
    else if (preference === 'auto' && context.gpuExperimental === true) selection = cpu();
    else selection = applyAdrenoProfile(selectionForTier(preference, probe), context.gpuProfile);
  } else if (context.android && (preference === 'auto' || (preference === 'g' && broken))) {
    selection = cpu();
  } else {
    selection = selectionForTier(preference, probe);
  }
  return context.android ? applyMobileLimits(selection) : selection;
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
