import { describe, expect, it } from 'vitest';
import {
  cpuAfterBrokenOutput,
  fitSelectionToModel,
  isLargeModelOnPhone,
  LARGE_MODEL_PHONE_LIMITS,
  lighterSelection,
  phoneKeepsOffGpu,
  resolveSelection,
  type BackendProbe,
} from '@/engine/in-browser/backend';
import { getModel, LARGE_MODEL_BYTES, MODEL_LIST } from '@/engine/model/catalog';
import { isMobileModelAllowed } from '@/shared/mobile';

// Issue (1.3.0, POCO F5 Pro): Hy-MT2 1.8B and PLaMo 3 2B closed Edge for Android while loading
// on WebGPU; both ran fine on the CPU there ("hy_mt2跟plamo 2b，cpu可執行").

const ADRENO: BackendProbe = {
  webgpu: true,
  hardwareConcurrency: 8,
  sharedMemory: true,
  adreno: true,
};

const HY = getModel('hy-mt2-1.8b-q4_k_m').size;
const SMALL = getModel('plamo-3-translate-0.8b-q4_k_m').size;
const PLAMO_2B = getModel('plamo-3-translate-2b-q4_k_m').size;
const QWEN_17 = getModel('qwen3-1.7b-q4_k_m').size;

describe('large models on a phone', () => {
  it('loads a model over 1 GB with one slot, n_ctx 1024 and a 128-token batch', () => {
    for (const tier of ['auto', 'g', 'c4', 'c8'] as const) {
      const selection = resolveSelection(tier, ADRENO, {
        android: true,
        adreno: true,
        modelBytes: HY,
      });
      expect(selection.nParallel).toBe(LARGE_MODEL_PHONE_LIMITS.nParallel);
      expect(selection.nCtx).toBe(LARGE_MODEL_PHONE_LIMITS.nCtx);
      expect(selection.nBatch).toBe(LARGE_MODEL_PHONE_LIMITS.nBatch);
      expect(selection.nThreads).toBeLessThanOrEqual(4);
    }
  });

  it('runs every model over 1 GB on the CPU under auto and an explicit WebGPU choice', () => {
    for (const modelBytes of [HY, PLAMO_2B, QWEN_17]) {
      expect(modelBytes).toBeGreaterThan(LARGE_MODEL_BYTES);
      for (const tier of ['auto', 'g'] as const) {
        const selection = resolveSelection(tier, ADRENO, {
          android: true,
          adreno: true,
          gpuProfile: 'optimized',
          modelBytes,
        });
        expect(selection.backend).toBe('cpu-wasm');
        expect(selection.nGpuLayers).toBe(0);
        expect(selection.webgpuTuning).toBeUndefined();
      }
      // The CPU tiers stay as chosen.
      expect(
        resolveSelection('c4', ADRENO, { android: true, adreno: true, modelBytes }).nThreads,
      ).toBe(4);
    }
    // Same without an Adreno (explicit WebGPU on another phone GPU).
    expect(
      resolveSelection('g', { ...ADRENO, adreno: false }, { android: true, modelBytes: HY })
        .backend,
    ).toBe('cpu-wasm');
  });

  it('keeps the smaller models on WebGPU, and a desktop large model too', () => {
    for (const model of MODEL_LIST.filter((m) => m.size <= LARGE_MODEL_BYTES)) {
      expect(isLargeModelOnPhone({ android: true, modelBytes: model.size })).toBe(false);
      expect(
        resolveSelection('g', ADRENO, { android: true, adreno: true, modelBytes: model.size })
          .backend,
      ).toBe('webgpu');
    }
    expect(
      resolveSelection('g', ADRENO, { android: false, adreno: true, modelBytes: PLAMO_2B }).backend,
    ).toBe('webgpu');
  });

  it('runs a model that crashed in a WebGPU load on the CPU until WebGPU is chosen again', () => {
    const context = { android: true, adreno: true, modelBytes: SMALL, gpuCrashed: true };
    expect(phoneKeepsOffGpu(context)).toBe(true);
    expect(resolveSelection('auto', ADRENO, context).backend).toBe('cpu-wasm');
    expect(resolveSelection('g', ADRENO, context).backend).toBe('cpu-wasm');
    expect(resolveSelection('g', ADRENO, { ...context, gpuCrashed: false }).backend).toBe('webgpu');
    // Desktop never gets the flag's effect.
    expect(phoneKeepsOffGpu({ ...context, android: false })).toBe(false);
  });

  it('keeps the measured PLaMo 0.8B setup (2 slots on the Adreno GPU)', () => {
    const selection = resolveSelection('auto', ADRENO, {
      android: true,
      adreno: true,
      modelBytes: SMALL,
    });
    expect(selection.backend).toBe('webgpu');
    expect(selection.nParallel).toBe(2);
    expect(selection.nCtx).toBe(2048);
    expect(selection.nBatch).toBe(256);
  });

  it('applies the limits to the CPU fallbacks too', () => {
    const gpu = resolveSelection('g', ADRENO, { android: true, adreno: true, modelBytes: HY });
    const cpu = lighterSelection(gpu) ?? gpu;
    // The fallback itself is derived without the phone caps; the host fits every attempt.
    const fitted = fitSelectionToModel(cpu, { android: true, modelBytes: HY });
    expect(fitted.nCtx).toBe(1024);
    expect(fitted.nBatch).toBe(128);
    const broken = cpuAfterBrokenOutput(gpu, { android: true, modelBytes: HY });
    expect(broken.backend).toBe('cpu-wasm');
    expect(broken.nCtx).toBe(1024);
    expect(broken.nBatch).toBe(128);
  });

  it('leaves desktop selections alone', () => {
    const desktop = resolveSelection(
      'g',
      { ...ADRENO, adreno: false },
      { android: false, modelBytes: HY },
    );
    expect(fitSelectionToModel(desktop, { android: false, modelBytes: HY })).toEqual(desktop);
    expect(desktop.nBatch).toBeUndefined();
  });

  it('offers every catalog model on a phone again, PLaMo 3 2B included', () => {
    for (const model of MODEL_LIST) expect(isMobileModelAllowed(model.id)).toBe(true);
    expect(isMobileModelAllowed('plamo-3-translate-2b-q4_k_m')).toBe(true);
  });

  it('wires the guard and the limits into the engine host', async () => {
    const { readFileSync } = await import('node:fs');
    const host = readFileSync(new URL('../src/engine/offscreen/host.ts', import.meta.url), 'utf8');
    expect(host).toContain('fitSelectionToModel(selection, runtime)');
    expect(host).toContain('if (android) await markModelLoad(modelId, fitted.backend);');
    expect(host).toContain('if (android) await markModelLoad(null);');
    expect(host).toContain('requestCrashedModelLoad()');
    expect(host).toContain('runtime.modelBytes = getModel(modelId).size;');
    expect(host).toContain('runtime.gpuCrashed = gpuCrashed.has(modelId);');
  });
});
