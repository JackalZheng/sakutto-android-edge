import { describe, expect, it } from 'vitest';
import {
  applyMobileLimits,
  CONTEXT_TOKENS,
  cpuFallback,
  GPU_LOSS_WINDOW_MS,
  isGpuLoadFallback,
  lighterSelection,
  selectionAfterGpuLoss,
  selectBackend,
  selectionForTier,
  WEBGPU_ALL_LAYERS,
  ADRENO_R6_TUNING,
  applyAdrenoProfile,
  resolveSelection,
} from '@/engine/in-browser/backend';
import { DEFAULT_MODEL_ID, getModel, isModelId, MODEL_CATALOG } from '@/engine/model/catalog';

describe('backend selection', () => {
  it('runs four slots on a GPU without the fast integer shaders', () => {
    for (const probe of [{ fastBatches: false }, {}]) {
      expect(selectBackend({ webgpu: true, hardwareConcurrency: 8, ...probe })).toMatchObject({
        nParallel: 4,
        nCtx: 1024 * 4,
      });
    }
  });

  it('keeps the fast shader width when the GPU tier is forced', () => {
    const probe = { webgpu: false, hardwareConcurrency: 8, fastBatches: true };
    expect(selectionForTier('g', probe).nParallel).toBe(16);
  });

  it('uses every GPU layer and sixteen slots when WebGPU has the fast shaders', () => {
    expect(selectBackend({ webgpu: true, hardwareConcurrency: 4, fastBatches: true })).toEqual({
      tier: 'g',
      backend: 'webgpu',
      nGpuLayers: WEBGPU_ALL_LAYERS,
      nThreads: 4,
      nParallel: 16,
      nCtx: 1024 * 16,
    });
  });

  it('gives each GPU sequence 1024 tokens of context', () => {
    const gpu = selectBackend({ webgpu: true, hardwareConcurrency: 16 });
    expect(gpu.nCtx).toBe(gpu.nParallel * 1024);
  });

  it('runs eight threads sequentially on a CPU tier with enough cores', () => {
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 12 })).toEqual({
      tier: 'c8',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: 8,
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    });
  });

  it('gives a large CPU half its logical cores, up to sixteen threads', () => {
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 16 }).nThreads).toBe(8);
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 24 }).nThreads).toBe(12);
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 32 }).nThreads).toBe(16);
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 64 }).nThreads).toBe(16);
    expect(selectionForTier('c8', { webgpu: true, hardwareConcurrency: 32 }).nThreads).toBe(16);
    expect(selectBackend({ webgpu: true, hardwareConcurrency: 32 }).nThreads).toBe(8);
  });

  it('uses one CPU thread when the page is not cross-origin isolated', () => {
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 32, sharedMemory: false })).toEqual({
      tier: 'c1',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: 1,
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
      sharedMemory: false,
    });
    expect(
      selectBackend({ webgpu: true, hardwareConcurrency: 16, sharedMemory: false }).nThreads,
    ).toBe(1);
    expect(
      selectionForTier('c8', { webgpu: true, hardwareConcurrency: 32, sharedMemory: false }),
    ).toMatchObject({ tier: 'c1', nThreads: 1 });
  });

  it('caps a small CPU at four threads', () => {
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 4 })).toMatchObject({
      tier: 'c4',
      backend: 'cpu-wasm',
      nThreads: 4,
      nParallel: 1,
    });
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 1 }).nThreads).toBe(1);
    expect(selectBackend({ webgpu: false, hardwareConcurrency: 0 }).nThreads).toBe(1);
  });

  it('falls back from WebGPU to a matching CPU tier', () => {
    expect(cpuFallback(selectBackend({ webgpu: true, hardwareConcurrency: 12 }))).toMatchObject({
      tier: 'c8',
      backend: 'cpu-wasm',
      nGpuLayers: 0,
      nThreads: 8,
      nParallel: 1,
      nCtx: CONTEXT_TOKENS,
    });
    expect(cpuFallback(selectBackend({ webgpu: true, hardwareConcurrency: 4 }))).toMatchObject({
      tier: 'c4',
      backend: 'cpu-wasm',
      nThreads: 4,
    });
    expect(cpuFallback(selectBackend({ webgpu: false, hardwareConcurrency: 8 }))).toBeNull();
    expect(
      cpuFallback(selectBackend({ webgpu: true, hardwareConcurrency: 12, sharedMemory: false })),
    ).toMatchObject({
      tier: 'c1',
      nThreads: 1,
    });
  });

  it('lets a saved preference force the CPU tier on a GPU machine', () => {
    const probe = { webgpu: true, hardwareConcurrency: 12 };
    expect(selectionForTier('auto', probe).tier).toBe('g');
    expect(selectionForTier('c4', probe)).toMatchObject({
      tier: 'c4',
      backend: 'cpu-wasm',
      nThreads: 4,
      nParallel: 1,
    });
    expect(selectionForTier('c8', { webgpu: false, hardwareConcurrency: 4 })).toMatchObject({
      tier: 'c8',
      nThreads: 4,
    });
  });
  it('caps threads and batch size for a phone without narrowing the GPU', () => {
    const gpu = applyMobileLimits(selectBackend({ webgpu: true, hardwareConcurrency: 8 }));
    expect(gpu).toMatchObject({ nThreads: 4, nBatch: 256, nParallel: 4 });

    const cpu = applyMobileLimits(selectBackend({ webgpu: false, hardwareConcurrency: 12 }));
    expect(cpu).toMatchObject({ nThreads: 4, nBatch: 256, nParallel: 1 });

    const small = applyMobileLimits(selectBackend({ webgpu: false, hardwareConcurrency: 2 }));
    expect(small).toMatchObject({ nThreads: 2, nBatch: 256 });
  });
});

describe('model catalog', () => {
  it('pins the default model commit, size and digest', () => {
    const entry = getModel(DEFAULT_MODEL_ID);

    expect(entry.repository).toBe('pfnet/plamo-3-translate-0.8b-GGUF');
    expect(entry.file).toBe('plamo-3-translate-0.8b-Q4_K_M.gguf');
    expect(entry.commit).toBe('dbeb0856cb0b004b484f190bb71dd65876297013');
    expect(entry.size).toBe(573_123_200);
    expect(entry.sha256).toBe('923b2f28a9856de1b8b3b610a7e59ded0b14d007466fd5dd526e6a9ecaadafdd');
    expect(entry.url).toBe(
      'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/dbeb0856cb0b004b484f190bb71dd65876297013/plamo-3-translate-0.8b-Q4_K_M.gguf',
    );
  });

  it('pins the optional 2B model with the same verification data', () => {
    const entry = MODEL_CATALOG['plamo-3-translate-2b-q4_k_m'];

    expect(entry.repository).toBe('pfnet/plamo-3-translate-2b-GGUF');
    expect(entry.commit).toBe('232fbd7916519edecddfdae0beae7e76fe70453f');
    expect(entry.size).toBe(1_807_199_392);
    expect(entry.sha256).toBe('f36d0813dd95573a852ab5d76e374c543632d9e292929e922bcdbf7e53b494e0');
  });

  it('validates model identifiers', () => {
    expect(isModelId(DEFAULT_MODEL_ID)).toBe(true);
    expect(isModelId('plamo-3-translate-2b-q4_k_m')).toBe(true);
    expect(isModelId('plamo-3-translate-8b-q4_k_m')).toBe(false);
    expect(isModelId(null)).toBe(false);
  });
});

describe('stepping down after a GPU failure', () => {
  const wide = selectBackend({ webgpu: true, hardwareConcurrency: 8, fastBatches: true });
  const plain = selectBackend({ webgpu: true, hardwareConcurrency: 8 });
  const cpu = selectBackend({ webgpu: false, hardwareConcurrency: 8 });

  it('walks from sixteen slots to four to one to the CPU', () => {
    expect(lighterSelection(wide)).toEqual(plain);
    const conservative = lighterSelection(plain);
    expect(conservative).toMatchObject({ backend: 'webgpu', nParallel: 1, nCtx: 1024 });
    expect(lighterSelection(conservative!)).toMatchObject({ backend: 'cpu-wasm', nParallel: 1 });
    expect(lighterSelection(cpu)).toBeNull();
  });

  it('keeps the selection after a lone loss', () => {
    expect(selectionAfterGpuLoss(wide, GPU_LOSS_WINDOW_MS)).toBe(wide);
    expect(selectionAfterGpuLoss(wide, Number.POSITIVE_INFINITY)).toBe(wide);
  });

  it('steps down when the GPU is lost again soon after', () => {
    expect(selectionAfterGpuLoss(wide, GPU_LOSS_WINDOW_MS - 1)).toEqual(plain);
    expect(selectionAfterGpuLoss(plain, 1000)).toMatchObject({ backend: 'webgpu', nParallel: 1 });
  });

  it('does not treat a manual CPU choice as a GPU failure', () => {
    expect(isGpuLoadFallback(wide, cpu)).toBe(true);
    expect(isGpuLoadFallback(cpu, cpu)).toBe(false);
    expect(isGpuLoadFallback(wide, plain)).toBe(false);
  });
});

describe('Adreno policy (v1.3.1)', () => {
  const gpu = { webgpu: true, hardwareConcurrency: 8, fastBatches: false, adreno: true };

  it('runs Automatic on WebGPU with the round-6 profile on Adreno, phone or desktop', () => {
    for (const android of [true, false]) {
      const selection = resolveSelection('auto', gpu, { android, adreno: true });
      expect(selection).toMatchObject({ backend: 'webgpu', nParallel: 2, nCtx: 2048 });
      expect(selection.webgpuTuning).toEqual(['q4v2r4', 'mvglu', 'fna']);
      expect(selection.webgpuTuning).toBe(ADRENO_R6_TUNING);
    }
  });

  it('applies the phone caps to Automatic on an Adreno phone', () => {
    const selection = resolveSelection('auto', gpu, { android: true, adreno: true });
    expect(selection.nThreads).toBeLessThanOrEqual(4);
    expect(selection.nBatch).toBeDefined();
  });

  it('keeps Automatic on the CPU on Adreno without a WebGPU adapter', () => {
    const selection = resolveSelection(
      'auto',
      { ...gpu, webgpu: false },
      {
        android: true,
        adreno: true,
      },
    );
    expect(selection.backend).toBe('cpu-wasm');
    expect(selection.webgpuTuning).toBeUndefined();
  });

  it('keeps Automatic on the CPU on a non-Adreno phone', () => {
    expect(
      resolveSelection('auto', { ...gpu, adreno: false }, { android: true, adreno: false }).backend,
    ).toBe('cpu-wasm');
  });

  it('leaves non-Adreno GPUs on the upstream defaults', () => {
    const selection = resolveSelection('auto', { ...gpu, adreno: false }, { android: false });
    expect(selection).toMatchObject({ backend: 'webgpu', nParallel: 4, nCtx: 4096 });
    expect(selection.webgpuTuning).toBeUndefined();
  });

  it('runs an explicit GPU choice with two slots, 2048 tokens of context and round 6', () => {
    const selection = resolveSelection('g', gpu, {
      android: true,
      adreno: true,
      gpuProfile: 'optimized',
    });
    expect(selection).toMatchObject({ backend: 'webgpu', nParallel: 2, nCtx: 2048 });
    expect(selection.webgpuTuning).toEqual(['q4v2r4', 'mvglu', 'fna']);
    expect(selection.nGpuLayers).toBe(WEBGPU_ALL_LAYERS);
  });

  it('runs the round-5 defaults (no tokens) for the compatible profile', () => {
    for (const preference of ['auto', 'g'] as const) {
      const selection = resolveSelection(preference, gpu, {
        android: false,
        adreno: true,
        gpuProfile: 'compat',
      });
      expect(selection.backend).toBe('webgpu');
      expect(selection.webgpuTuning).toEqual([]);
    }
  });

  it('falls back to the CPU after garbled GPU output, for Automatic and GPU', () => {
    for (const android of [true, false]) {
      for (const preference of ['auto', 'g'] as const) {
        expect(
          resolveSelection(preference, gpu, { android, adreno: true, gpuOutputBroken: true })
            .backend,
        ).toBe('cpu-wasm');
      }
    }
  });

  it('keeps an explicit CPU tier on the CPU on Adreno', () => {
    const selection = resolveSelection('c4', gpu, { android: true, adreno: true });
    expect(selection.backend).toBe('cpu-wasm');
    expect(selection.webgpuTuning).toBeUndefined();
  });

  it('never applies the profile to a CPU selection', () => {
    const cpu = selectionForTier('c4', gpu);
    expect(applyAdrenoProfile(cpu, 'compat')).toBe(cpu);
    expect(applyAdrenoProfile(cpu, 'optimized')).toBe(cpu);
  });

  it('gives every catalog model its block count', () => {
    expect(Object.values(MODEL_CATALOG).map((model) => model.layers)).toEqual([16, 24]);
  });
});
