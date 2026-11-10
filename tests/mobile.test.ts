import { describe, expect, it } from 'vitest';
import {
  MOBILE_MODEL_ID,
  isAndroidEdgeUserAgent,
  isEdgeUserAgent,
  isMobileModelAllowed,
} from '@/shared/mobile';
import { edgeCommands } from '@/shared/manifest';

const EDGE_ANDROID_UA =
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36 EdgA/140.0.0.0';
const DESKTOP_EDGE_UA =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0';
const CHROME_ANDROID_UA =
  'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Mobile Safari/537.36';

describe('Edge for Android mobile target', () => {
  it('detects Edge on Android from the EdgA token', () => {
    expect(isAndroidEdgeUserAgent(EDGE_ANDROID_UA)).toBe(true);
    expect(isAndroidEdgeUserAgent(DESKTOP_EDGE_UA)).toBe(false);
    expect(isAndroidEdgeUserAgent(CHROME_ANDROID_UA)).toBe(false);
    expect(isEdgeUserAgent(DESKTOP_EDGE_UA)).toBe(true);
  });

  it('only allows the 0.8B model on mobile', () => {
    expect(MOBILE_MODEL_ID).toBe('plamo-3-translate-0.8b-q4_k_m');
    expect(isMobileModelAllowed('plamo-3-translate-0.8b-q4_k_m')).toBe(true);
    expect(isMobileModelAllowed('plamo-3-translate-2b-q4_k_m')).toBe(false);
  });

  it('declares Edge commands with no keyboard shortcut', () => {
    const commands = edgeCommands();
    expect(Object.keys(commands).sort()).toEqual([
      'toggle-bilingual',
      'translate-compose',
      'translate-page',
      'translate-selection',
    ]);
    for (const command of Object.values(commands)) {
      expect(command).not.toHaveProperty('suggested_key');
      expect(typeof command.description).toBe('string');
    }
  });
});
