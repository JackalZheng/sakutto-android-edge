import { describe, expect, it } from 'vitest';
import { shouldInjectOnLoad } from '@/background/page-actions';
import { DEFAULT_SETTINGS, normalizeSettings } from '@/shared/settings';

const CONSENT = {
  license: 'plamo-community',
  version: '2026-11-05',
  notice: '2026-11-06',
  acceptedAt: '2026-11-20T00:00:00.000Z',
  ageNoticeShown: true,
} as const;

describe('floating button on every site', () => {
  it('is on by default, and an older stored record without the field turns it on', () => {
    expect(DEFAULT_SETTINGS.fabEverywhere).toBe(true);
    const stored = { ...DEFAULT_SETTINGS } as Record<string, unknown>;
    delete stored.fabEverywhere;
    expect(normalizeSettings(stored, 'en').fabEverywhere).toBe(true);
    expect(normalizeSettings({ ...stored, fabEverywhere: false }, 'en').fabEverywhere).toBe(false);
  });

  it('injects on every page load on a touch runtime once the license is accepted', () => {
    const settings = { fabEverywhere: true, consent: CONSENT };
    expect(shouldInjectOnLoad({ rule: { auto: 'ask' }, settings, touch: true })).toBe(true);
  });

  it('respects the switch, a site set to never, consent, and desktop', () => {
    const settings = { fabEverywhere: true, consent: CONSENT };
    expect(
      shouldInjectOnLoad({
        rule: { auto: 'ask' },
        settings: { ...settings, fabEverywhere: false },
        touch: true,
      }),
    ).toBe(false);
    expect(shouldInjectOnLoad({ rule: { auto: 'never' }, settings, touch: true })).toBe(false);
    expect(
      shouldInjectOnLoad({
        rule: { auto: 'ask' },
        settings: { ...settings, consent: null },
        touch: true,
      }),
    ).toBe(false);
    expect(shouldInjectOnLoad({ rule: { auto: 'ask' }, settings, touch: false })).toBe(false);
  });

  it('still injects auto-translate sites everywhere', () => {
    expect(shouldInjectOnLoad({ rule: { auto: 'always' }, settings: null, touch: false })).toBe(
      true,
    );
  });
});
