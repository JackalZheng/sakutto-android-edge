/**
 * The Adreno WebGPU profiles offered in the settings (1.3.0). Each is a set of ggml-webgpu tuning
 * tokens (patches 0021-0025, see `WEBGPU_TUNING_TOKEN_PATTERN`) on top of the Adreno defaults
 * ggml-webgpu applies by itself (subgroups off, mat-vec workgroup 64, the new SOFT_MAX, 16 kernels
 * per submit, batched mat-vec for 5-64-token prompts and 128-wide row reductions: round 5), plus
 * an optional context override. All of them come from the GPU harness sweeps on the POCO
 * F5 Pro (Adreno 730, PLaMo 3 0.8B, warm run) and translated correctly there.
 *
 * Left out on purpose: the subgroup profiles (`adreno-upstream`, `adreno-subgroups`,
 * `adreno-sg-tiled`) garble every translation on Adreno; 64 kernels per submit (`sub64`) lost the
 * device in perf runs; the hybrid half-offload was slower and once stalled for 41 s.
 */
export const ADRENO_GPU_PROFILES = [
  'optimized',
  'q4v2',
  'r6',
  'low-memory',
  'compat',
  'r4',
] as const;

export type AdrenoGpuProfile = (typeof ADRENO_GPU_PROFILES)[number];

export const DEFAULT_ADRENO_GPU_PROFILE: AdrenoGpuProfile = 'optimized';

/** Round 6 with 4 rows per thread group: q4_K mat-vec v2 + fused gate/up SWIGLU + fused norm. */
export const ADRENO_R6_TUNING: readonly string[] = ['q4v2r4', 'mvglu', 'fna'];

export interface AdrenoProfileSpec {
  id: AdrenoGpuProfile;
  /** ggml-webgpu tuning tokens (empty: the round-5 defaults). */
  tuning: readonly string[];
  /**
   * n_ctx override; unset keeps the Adreno default (one slot, n_ctx 2048). Since 1.3.0 every
   * profile runs one slot: on the POCO F5 Pro two slots ran one after the other and each title
   * lost the shared prompt-prefix cache (Gemma 3 1B titles, one per request: 14.7 s on 1 slot,
   * 31.8 s on 2, 28.0 s on 4).
   */
  nCtx?: number;
  /** The GPU harness config this profile was measured as. */
  harnessConfig: string;
}

export const ADRENO_PROFILE_SPECS: Readonly<Record<AdrenoGpuProfile, AdrenoProfileSpec>> =
  Object.freeze({
    optimized: { id: 'optimized', tuning: ADRENO_R6_TUNING, harnessConfig: 'gpu-r6-r4' },
    q4v2: { id: 'q4v2', tuning: ['q4v2r4'], harnessConfig: 'gpu-q4v2-r4' },
    r6: { id: 'r6', tuning: ['q4v2r2', 'mvglu', 'fna'], harnessConfig: 'gpu-r6' },
    'low-memory': {
      id: 'low-memory',
      tuning: ADRENO_R6_TUNING,
      nCtx: 1024,
      harnessConfig: 'gpu-r6-r4-1-slot (n_ctx 1024)',
    },
    compat: { id: 'compat', tuning: [], harnessConfig: 'gpu-r5' },
    // Round 4: no batched mat-vec for prompts (bmv0), 256-wide row reductions (rwg256).
    r4: { id: 'r4', tuning: ['bmv0', 'rwg256'], harnessConfig: 'gpu-r4' },
  });

export function isAdrenoGpuProfile(value: unknown): value is AdrenoGpuProfile {
  return typeof value === 'string' && (ADRENO_GPU_PROFILES as readonly string[]).includes(value);
}

/**
 * Stored values from older builds: 1.3.0-dev's 'hybrid' (removed in 1.3.1) becomes the default.
 */
export function normalizeAdrenoGpuProfile(value: unknown): AdrenoGpuProfile {
  return isAdrenoGpuProfile(value) ? value : DEFAULT_ADRENO_GPU_PROFILE;
}
