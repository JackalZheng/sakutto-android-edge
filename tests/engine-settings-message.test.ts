import { describe, expect, it } from 'vitest';
import {
  isEngineSettingsQuery,
  isEngineSettingsSnapshot,
  isGpuOutputBrokenUpdate,
} from '@/shared/messages';

describe('engine settings bridge (offscreen has no storage)', () => {
  it('recognizes the query and the flag update', () => {
    expect(isEngineSettingsQuery({ type: 'engine:settings' })).toBe(true);
    expect(isEngineSettingsQuery({ type: 'engine:tier' })).toBe(false);
    expect(isGpuOutputBrokenUpdate({ type: 'engine:gpu-output-broken', broken: true })).toBe(true);
    expect(isGpuOutputBrokenUpdate({ type: 'engine:gpu-output-broken', broken: 'yes' })).toBe(
      false,
    );
  });

  it('validates the snapshot the background returns', () => {
    expect(
      isEngineSettingsSnapshot({ tier: 'g', gpuProfile: 'compat', gpuOutputBroken: false }),
    ).toBe(true);
    // 1.3.1 removed the hybrid profile.
    expect(
      isEngineSettingsSnapshot({ tier: 'g', gpuProfile: 'hybrid', gpuOutputBroken: false }),
    ).toBe(false);
    expect(
      isEngineSettingsSnapshot({ tier: 'g', gpuProfile: 'turbo', gpuOutputBroken: false }),
    ).toBe(false);
    expect(isEngineSettingsSnapshot(null)).toBe(false);
  });

  it('carries every 1.3.0 profile and the stored model', () => {
    for (const gpuProfile of ['optimized', 'q4v2', 'r6', 'low-memory', 'compat', 'r4']) {
      expect(isEngineSettingsSnapshot({ tier: 'auto', gpuProfile, gpuOutputBroken: false })).toBe(
        true,
      );
    }
    expect(
      isEngineSettingsSnapshot({
        tier: 'auto',
        gpuProfile: 'optimized',
        gpuOutputBroken: false,
        modelId: 'hy-mt2-1.8b-q4_k_m',
      }),
    ).toBe(true);
    expect(
      isEngineSettingsSnapshot({
        tier: 'auto',
        gpuProfile: 'optimized',
        gpuOutputBroken: false,
        modelId: 'gpt-5',
      }),
    ).toBe(false);
  });
});
