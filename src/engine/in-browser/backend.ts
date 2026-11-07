export type ComputeTier = 'g' | 'c8' | 'c4';

export type BackendKind = 'webgpu' | 'cpu-wasm';

export const CONTEXT_TOKENS = 2048;

export const WEBGPU_ALL_LAYERS = 99999;

/**
 * Two sequences on WebGPU.
 *
 * Two slots used to look slower than one, but the cost was wllama's `kv_unified: true`
 * default, not the second slot: with one shared KV cache, the second sequence made the
 * first take 4.3× as long, and the request after a pair was poisoned as well (a pair that
 * took 1.2 s was followed by a 4.2 s request). With a per-sequence KV cache the same pair
 * finishes in 0.92 s against 1.6 s one after the other, and the next request stays at
 * 0.65 s. `runtime.ts` therefore turns `kv_unified` off whenever `nParallel` is above one.
 */
const WEBGPU_SLOTS = 2;
// llama.cpp divides n_ctx across parallel slots, so the GPU tier sizes it for two.
const WEBGPU_CONTEXT_TOKENS = CONTEXT_TOKENS * WEBGPU_SLOTS;

export interface BackendProbe {
  webgpu: boolean;
  hardwareConcurrency: number;
}

export interface BackendSelection {
  tier: ComputeTier;
  backend: BackendKind;
  nGpuLayers: number;
  nThreads: number;
  nParallel: number;
  nCtx: number;
}

function clampThreads(requested: number, hardwareConcurrency: number): number {
  const cores = Number.isFinite(hardwareConcurrency) ? Math.floor(hardwareConcurrency) : 4;
  return Math.max(1, Math.min(requested, Math.max(1, cores)));
}

/**
 * Every CPU tier keeps one sequence, so the paragraph on screen gets the whole model instead
 * of sharing each decode step with the next one. WebGPU takes two, which only pays off with a
 * per-sequence KV cache; see `WEBGPU_SLOTS`.
 */
export function selectBackend(probe: BackendProbe): BackendSelection {
  if (probe.webgpu) {
    return {
      tier: 'g',
      backend: 'webgpu',
      nGpuLayers: WEBGPU_ALL_LAYERS,
      nThreads: clampThreads(8, probe.hardwareConcurrency),
      nParallel: WEBGPU_SLOTS,
      nCtx: WEBGPU_CONTEXT_TOKENS,
    };
  }

  if (probe.hardwareConcurrency >= 8) {
    return {
      tier: 'c8',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: clampThreads(8, probe.hardwareConcurrency),
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    };
  }

  return {
    tier: 'c4',
    backend: 'cpu-wasm',
    nGpuLayers: 0,
    nThreads: clampThreads(4, probe.hardwareConcurrency),
    nParallel: 1,
    nCtx: CONTEXT_TOKENS,
  };
}

export type TierPreference = 'auto' | ComputeTier;

/** Resolves the settings choice. `auto` keeps the probe; the others force that tier. */
export function selectionForTier(
  preference: TierPreference,
  probe: BackendProbe,
): BackendSelection {
  if (preference === 'auto') return selectBackend(probe);
  if (preference === 'g') {
    return selectBackend({ webgpu: true, hardwareConcurrency: probe.hardwareConcurrency });
  }
  if (preference === 'c8') {
    return {
      tier: 'c8',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: clampThreads(8, probe.hardwareConcurrency),
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    };
  }

  return {
    tier: 'c4',
    backend: 'cpu-wasm',
    nGpuLayers: 0,
    nThreads: clampThreads(4, probe.hardwareConcurrency),
    nParallel: 1,
    nCtx: CONTEXT_TOKENS,
  };
}

export function cpuFallback(selection: BackendSelection): BackendSelection | null {
  if (selection.backend === 'cpu-wasm') return null;
  return selectBackend({
    webgpu: false,
    hardwareConcurrency: selection.nThreads,
  });
}
