import { describe, expect, it } from 'vitest';
import { gpuOutputBrokenForVersion } from '@/shared/gpu-output';

describe('broken-GPU-output flag', () => {
  it('only counts for the extension version that stored it', () => {
    expect(gpuOutputBrokenForVersion('1.3.0', '1.3.0')).toBe(true);
    expect(gpuOutputBrokenForVersion('1.3.0', '1.3.1')).toBe(false);
  });

  it('ignores the boolean flag 1.2.x and the 1.3.0 development build stored', () => {
    expect(gpuOutputBrokenForVersion(true, '1.3.0')).toBe(false);
  });

  it('ignores a flag a 1.3.1 development build stored', () => {
    expect(gpuOutputBrokenForVersion('1.3.1', '1.3.0')).toBe(false);
  });

  it('is off without a flag or a version', () => {
    expect(gpuOutputBrokenForVersion(undefined, '1.3.0')).toBe(false);
    expect(gpuOutputBrokenForVersion('1.3.0', null)).toBe(false);
  });
});
