import { describe, expect, it } from 'vitest';
import { adapterDisplayName, isAdrenoAdapter } from '@/shared/capabilities';

describe('Adreno adapter detection', () => {
  it('matches the Qualcomm vendor or an adreno architecture', () => {
    expect(isAdrenoAdapter('qualcomm', 'adreno-7xx')).toBe(true);
    expect(isAdrenoAdapter('', 'adreno-6xx')).toBe(true);
    expect(isAdrenoAdapter('Qualcomm Technologies', null)).toBe(true);
    expect(isAdrenoAdapter('nvidia', 'ampere')).toBe(false);
    expect(isAdrenoAdapter(null, null)).toBe(false);
  });

  it('names the adapter for the settings page', () => {
    expect(
      adapterDisplayName({
        vendor: 'qualcomm',
        architecture: 'adreno-7xx',
        description: 'Adreno (TM) 740',
      }),
    ).toBe('qualcomm adreno-7xx (Adreno (TM) 740)');
    expect(adapterDisplayName({ vendor: 'intel', architecture: '', description: '' })).toBe(
      'intel',
    );
    expect(adapterDisplayName({ vendor: null, architecture: null, description: null })).toBeNull();
  });
});
