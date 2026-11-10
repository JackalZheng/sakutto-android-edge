/**
 * Mobile / Android Edge helpers.
 *
 * Edge for Android runs Chromium MV3 extensions, but it is not desktop
 * Chrome: there is no physical keyboard (so `commands` shortcuts are
 * useless), the built-in Chrome Translator API does not exist, the
 * service worker / offscreen document is killed more aggressively, and
 * most Android GPUs fall back to the WASM CPU path. The port therefore
 * treats Edge on Android as a touch-first, PLaMo-only, 0.8B-only target.
 */

import { getModel, type ModelId } from '@/engine/model/catalog';

/** The only model offered on Edge for Android: 573 MB, not the 1.8 GB 2B. */
export const MOBILE_MODEL_ID: ModelId = 'plamo-3-translate-0.8b-q4_k_m';

/** Conservative engine limits for a phone: fewer threads, smaller batch. */
export const MOBILE_MAX_THREADS = 4;
export const MOBILE_MAX_BATCH = 256;

export function isAndroidUserAgent(userAgent: string): boolean {
  return /Android/i.test(userAgent);
}

export function isEdgeUserAgent(userAgent: string): boolean {
  return /EdgA?\//i.test(userAgent);
}

export function isAndroidEdgeUserAgent(userAgent: string): boolean {
  return isAndroidUserAgent(userAgent) && isEdgeUserAgent(userAgent);
}

/** True at runtime inside Edge for Android (service worker, popup, page). */
export function isAndroidEdgeRuntime(): boolean {
  if (typeof navigator === 'undefined') return false;
  return isAndroidEdgeUserAgent(navigator.userAgent);
}

export function isTouchRuntime(): boolean {
  if (typeof window === 'undefined') return isAndroidEdgeRuntime();
  return window.matchMedia?.('(pointer: coarse)').matches === true || navigator.maxTouchPoints > 0;
}

export function isMobileModelAllowed(modelId: ModelId): boolean {
  return modelId === MOBILE_MODEL_ID;
}

export function mobileModelSizeBytes(): number {
  return getModel(MOBILE_MODEL_ID).size;
}
