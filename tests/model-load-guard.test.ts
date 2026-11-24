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
  it('blames a recent mark, never the default model or a stale mark', () => {
    const now = 1_000_000_000;
    expect(guard.crashedModelFromMark({ modelId: 'hy-mt2-1.8b-q4_k_m', at: now - 5000 }, now)).toBe(
      'hy-mt2-1.8b-q4_k_m',
    );
    expect(
      guard.crashedModelFromMark({ modelId: 'plamo-3-translate-0.8b-q4_k_m', at: now }, now),
    ).toBeNull();
    expect(
      guard.crashedModelFromMark(
        { modelId: 'hy-mt2-1.8b-q4_k_m', at: now - guard.MODEL_LOAD_CRASH_WINDOW_MS - 1 },
        now,
      ),
    ).toBeNull();
    expect(guard.crashedModelFromMark({ modelId: 'nope', at: now }, now)).toBeNull();
    expect(guard.crashedModelFromMark(undefined, now)).toBeNull();
  });

  it('a mark left behind moves the settings back to the default and records the model', async () => {
    await guard.writeModelLoadMark('hy-mt2-1.8b-q4_k_m');

    expect(await guard.recoverFromCrashedModelLoad()).toBe('hy-mt2-1.8b-q4_k_m');
    expect(state.settings.modelId).toBe('plamo-3-translate-0.8b-q4_k_m');
    expect(state.values.has(guard.MODEL_LOAD_PENDING_KEY)).toBe(false);
    expect(Object.keys(await guard.loadCrashedModelLoads())).toEqual(['hy-mt2-1.8b-q4_k_m']);

    // Asked again by the same host: nothing new.
    expect(await guard.recoverFromCrashedModelLoad()).toBeNull();
  });

  it('a finished load leaves nothing to blame', async () => {
    await guard.markModelLoad('hy-mt2-1.8b-q4_k_m');
    await guard.markModelLoad(null);

    expect(await guard.requestCrashedModelLoad()).toBeNull();
    expect(state.settings.modelId).toBe('hy-mt2-1.8b-q4_k_m');
  });

  it('choosing the model again clears its crash record', async () => {
    await guard.writeModelLoadMark('qwen3-1.7b-q4_k_m');
    await guard.takeCrashedModelLoad();
    await guard.clearCrashedModelLoad('qwen3-1.7b-q4_k_m');
    expect(await guard.loadCrashedModelLoads()).toEqual({});
  });

  it('accepts only well-formed guard messages', () => {
    expect(guard.isModelLoadMarkUpdate({ type: 'engine:model-load', modelId: null })).toBe(true);
    expect(
      guard.isModelLoadMarkUpdate({ type: 'engine:model-load', modelId: 'hy-mt2-1.8b-q4_k_m' }),
    ).toBe(true);
    expect(guard.isModelLoadMarkUpdate({ type: 'engine:model-load', modelId: 'x' })).toBe(false);
    expect(guard.isModelLoadCrashQuery({ type: 'engine:model-load-crash' })).toBe(true);
  });
});
