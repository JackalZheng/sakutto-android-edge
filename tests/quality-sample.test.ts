import { afterEach, describe, expect, it, vi } from 'vitest';

const loadQualitySample = vi.fn();
const saveQualitySample = vi.fn();

vi.mock('@/shared/quality', () => ({
  loadQualitySample: () => loadQualitySample(),
  saveQualitySample: (sample: unknown) => saveQualitySample(sample),
}));

import type { EngineClient } from '@/background/engine-client';
import { measureQuality } from '@/background/quality';

const EDGE_ANDROID_UA =
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36 EdgA/140.0.0.0';
const DESKTOP_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36';

function client(): EngineClient & { benchmark: ReturnType<typeof vi.fn> } {
  return {
    status: vi.fn(async () => ({
      model: { id: 'plamo-3-translate-0.8b-q4_k_m', state: 'loaded' },
    })),
    warmModel: vi.fn(),
    benchmark: vi.fn(async () => 1234),
  } as unknown as EngineClient & { benchmark: ReturnType<typeof vi.fn> };
}

afterEach(() => {
  vi.unstubAllGlobals();
  loadQualitySample.mockReset();
  saveQualitySample.mockReset();
});

describe('measureQuality', () => {
  it('never runs the 512-token benchmark on Edge for Android', async () => {
    vi.stubGlobal('navigator', { userAgent: EDGE_ANDROID_UA });
    loadQualitySample.mockResolvedValue(null);
    const engine = client();

    await expect(measureQuality(engine)).resolves.toBeNull();
    expect(engine.benchmark).not.toHaveBeenCalled();
    expect(saveQualitySample).not.toHaveBeenCalled();
  });

  it('still measures once on desktop', async () => {
    vi.stubGlobal('navigator', { userAgent: DESKTOP_UA });
    loadQualitySample.mockResolvedValue(null);
    const engine = client();

    const sample = await measureQuality(engine);
    expect(engine.benchmark).toHaveBeenCalledTimes(1);
    expect(sample?.milliseconds).toBe(1234);
    expect(saveQualitySample).toHaveBeenCalledTimes(1);
  });
});
