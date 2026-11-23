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
      isEngineSettingsSnapshot({ tier: 'g', gpuProfile: 'hybrid', gpuOutputBroken: false }),
    ).toBe(true);
    expect(
      isEngineSettingsSnapshot({ tier: 'g', gpuProfile: 'turbo', gpuOutputBroken: false }),
    ).toBe(false);
    expect(isEngineSettingsSnapshot(null)).toBe(false);
  });
});
