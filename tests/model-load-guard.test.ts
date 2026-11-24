import { beforeEach, describe, expect, it, vi } from 'vitest';

// Issue (1.3.0, POCO F5 Pro): "hy-mt2 2b 跟 plamo2b 會當掉". Edge closed the engine page while
// the large model loaded, and the remembered model was loaded again by the next engine page.

const state = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  settings: { modelId: 'hy-mt2-1.8b-q4_k_m' as string },
}));

vi.mock('wxt/browser', () => ({
  browser: {
    storage: {
      local: {
        get: async (keys: string | string[]) =>
          Object.fromEntries(
            (Array.isArray(keys) ? keys : [keys]).map((key) => [key, state.values.get(key)]),
          ),
        set: async (values: Record<string, unknown>) => {
          for (const [key, value] of Object.entries(values)) state.values.set(key, value);
        },
        remove: async (key: string) => {
          state.values.delete(key);
        },
      },
    },
  },
}));

vi.mock('@/shared/storage', () => ({
  loadSettings: async () => ({ ...state.settings }),
  saveSettings: async (patch: { modelId?: string }) => {
    if (patch.modelId) state.settings.modelId = patch.modelId;
  },
}));

const guard = await import('@/shared/model-load-guard');

beforeEach(() => {
  state.values.clear();
  state.settings.modelId = 'hy-mt2-1.8b-q4_k_m';
});

describe('model load-crash guard', () => {
  it('blames a recent mark with its backend; the default model only on WebGPU', () => {
    const now = 1_000_000_000;
    expect(
      guard.crashedLoadFromMark(
        { modelId: 'hy-mt2-1.8b-q4_k_m', at: now - 5000, backend: 'cpu-wasm' },
        now,
      ),
    ).toEqual({ modelId: 'hy-mt2-1.8b-q4_k_m', backend: 'cpu-wasm' });
    // A dcdcf54 mark has no backend: that build loaded these models on WebGPU.
    expect(guard.crashedLoadFromMark({ modelId: 'hy-mt2-1.8b-q4_k_m', at: now }, now)).toEqual({
      modelId: 'hy-mt2-1.8b-q4_k_m',
      backend: 'webgpu',
    });
    expect(
      guard.crashedLoadFromMark(
        { modelId: 'plamo-3-translate-0.8b-q4_k_m', at: now, backend: 'cpu-wasm' },
        now,
      ),
    ).toBeNull();
    expect(
      guard.crashedLoadFromMark(
        { modelId: 'plamo-3-translate-0.8b-q4_k_m', at: now, backend: 'webgpu' },
        now,
      ),
    ).toEqual({ modelId: 'plamo-3-translate-0.8b-q4_k_m', backend: 'webgpu' });
    expect(
      guard.crashedLoadFromMark(
        { modelId: 'hy-mt2-1.8b-q4_k_m', at: now - guard.MODEL_LOAD_CRASH_WINDOW_MS - 1 },
        now,
      ),
    ).toBeNull();
    expect(guard.crashedLoadFromMark({ modelId: 'nope', at: now }, now)).toBeNull();
    expect(guard.crashedLoadFromMark(undefined, now)).toBeNull();
  });

  it('a crash on WebGPU keeps the model and reports it for the CPU', async () => {
    state.settings.modelId = 'plamo-3-translate-2b-q4_k_m';
    await guard.writeModelLoadMark('plamo-3-translate-2b-q4_k_m', 'webgpu');

    expect(await guard.recoverFromCrashedModelLoad()).toEqual({
      crashed: { modelId: 'plamo-3-translate-2b-q4_k_m', backend: 'webgpu' },
      gpuCrashed: ['plamo-3-translate-2b-q4_k_m'],
    });
    expect(state.settings.modelId).toBe('plamo-3-translate-2b-q4_k_m');
    // Later engine pages still get it, so it stays on the CPU.
    expect(await guard.requestCrashedModelLoad()).toEqual({
      crashed: null,
      gpuCrashed: ['plamo-3-translate-2b-q4_k_m'],
    });
    // Choosing the model again keeps the GPU record; choosing WebGPU again clears it.
    await guard.clearCrashedModelLoad('plamo-3-translate-2b-q4_k_m');
    expect(Object.keys(await guard.loadCrashedModelLoads())).toEqual([
      'plamo-3-translate-2b-q4_k_m',
    ]);
    await guard.clearGpuCrashedModelLoads();
    expect(await guard.loadCrashedModelLoads()).toEqual({});
  });

  it('a crash on the CPU moves the settings back to the default and records the model', async () => {
    await guard.writeModelLoadMark('hy-mt2-1.8b-q4_k_m', 'cpu-wasm');

    expect(await guard.recoverFromCrashedModelLoad()).toEqual({
      crashed: { modelId: 'hy-mt2-1.8b-q4_k_m', backend: 'cpu-wasm' },
      gpuCrashed: [],
    });
    expect(state.settings.modelId).toBe('plamo-3-translate-0.8b-q4_k_m');
    expect(state.values.has(guard.MODEL_LOAD_PENDING_KEY)).toBe(false);
    expect(Object.keys(await guard.loadCrashedModelLoads())).toEqual(['hy-mt2-1.8b-q4_k_m']);

    // Asked again by the same host: nothing new.
    expect((await guard.recoverFromCrashedModelLoad()).crashed).toBeNull();
  });

  it('a finished load leaves nothing to blame', async () => {
    await guard.markModelLoad('hy-mt2-1.8b-q4_k_m', 'webgpu');
    await guard.markModelLoad(null);

    expect(await guard.requestCrashedModelLoad()).toEqual({ crashed: null, gpuCrashed: [] });
    expect(state.settings.modelId).toBe('hy-mt2-1.8b-q4_k_m');
  });

  it('choosing the model again clears its CPU crash record', async () => {
    await guard.writeModelLoadMark('qwen3-1.7b-q4_k_m', 'cpu-wasm');
    await guard.takeCrashedModelLoad();
    await guard.clearCrashedModelLoad('qwen3-1.7b-q4_k_m');
    expect(await guard.loadCrashedModelLoads()).toEqual({});
  });

  it('reads a dcdcf54 crash record (a bare timestamp) as a WebGPU crash', () => {
    expect(guard.normalizeCrashedModelLoads({ 'hy-mt2-1.8b-q4_k_m': 5, junk: 1 })).toEqual({
      'hy-mt2-1.8b-q4_k_m': { at: 5, backend: 'webgpu' },
    });
  });

  it('accepts only well-formed guard messages', () => {
    expect(guard.isModelLoadMarkUpdate({ type: 'engine:model-load', modelId: null })).toBe(true);
    expect(
      guard.isModelLoadMarkUpdate({
        type: 'engine:model-load',
        modelId: 'hy-mt2-1.8b-q4_k_m',
        backend: 'webgpu',
      }),
    ).toBe(true);
    expect(
      guard.isModelLoadMarkUpdate({
        type: 'engine:model-load',
        modelId: 'hy-mt2-1.8b-q4_k_m',
        backend: 'metal',
      }),
    ).toBe(false);
    expect(guard.isModelLoadMarkUpdate({ type: 'engine:model-load', modelId: 'x' })).toBe(false);
    expect(guard.isModelLoadCrashQuery({ type: 'engine:model-load-crash' })).toBe(true);
    expect(guard.isModelLoadCrashReport({ crashed: null, gpuCrashed: [] })).toBe(true);
    expect(
      guard.isModelLoadCrashReport({
        crashed: { modelId: 'hy-mt2-1.8b-q4_k_m', backend: 'webgpu' },
        gpuCrashed: ['hy-mt2-1.8b-q4_k_m'],
      }),
    ).toBe(true);
    expect(guard.isModelLoadCrashReport('hy-mt2-1.8b-q4_k_m')).toBe(false);
  });
});
