import { describe, expect, it } from 'vitest';
import { gpuOutputBrokenForVersion } from '@/shared/gpu-output';

describe('broken-GPU-output flag', () => {
  it('only counts for the extension version that stored it', () => {
    expect(gpuOutputBrokenForVersion('1.3.1', '1.3.1')).toBe(true);
    expect(gpuOutputBrokenForVersion('1.3.1', '1.3.2')).toBe(false);
  });

  it('ignores the flag 1.3.0 and earlier stored as true', () => {
    expect(gpuOutputBrokenForVersion(true, '1.3.1')).toBe(false);
  });

  it('is off without a flag or a version', () => {
    expect(gpuOutputBrokenForVersion(undefined, '1.3.1')).toBe(false);
    expect(gpuOutputBrokenForVersion('1.3.1', null)).toBe(false);
  });
});
