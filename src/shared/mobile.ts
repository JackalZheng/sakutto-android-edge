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

/**
 * True for a UA that belongs to an Android browser, with or without the
 * EdgA token — Quetta's UA has none, so it still counts as Android.
 * `userAgentData.platform` covers a UA that hides the OS token.
 */
export function isAndroidRuntimeUserAgent(
  userAgent: string,
  userAgentData?: { platform?: string } | null,
): boolean {
  return isAndroidUserAgent(userAgent) || userAgentData?.platform === 'Android';
}

type NavigatorWithUserAgentData = Navigator & {
  userAgentData?: { platform?: string; mobile?: boolean };
};

/** True at runtime inside an Android browser running the extension. */
export function isAndroidRuntime(): boolean {
  if (typeof navigator === 'undefined') return false;
  return isAndroidRuntimeUserAgent(
    navigator.userAgent,
    (navigator as NavigatorWithUserAgentData).userAgentData ?? null,
  );
}

/**
 * True at runtime inside Edge for Android — and, since the same phone caps,
 * model restriction, and Auto→CPU guard apply to any Android browser running
 * the extension (Quetta has no EdgA token), inside an Android browser
 * generally. The strict EdgA check stays in `isAndroidEdgeUserAgent` for
 * UA classification; this one answers "is this an Android runtime".
 */
export function isAndroidEdgeRuntime(): boolean {
  return isAndroidRuntime();
}

export function isTouchRuntime(): boolean {
  if (typeof window === 'undefined') return isAndroidEdgeRuntime();
  return window.matchMedia?.('(pointer: coarse)').matches === true || navigator.maxTouchPoints > 0;
}

/**
 * Whether a page command warms the engine first. On a touch runtime the warm-up
 * generation would take the inference slot ahead of the translation the user just
 * asked for, so the command goes straight to translating; the translate request
 * loads the model itself when it has to.
 */
export function warmBeforePageCommand(): boolean {
  return !isTouchRuntime();
}

export function isMobileModelAllowed(modelId: ModelId): boolean {
  return modelId === MOBILE_MODEL_ID;
}

export function mobileModelSizeBytes(): number {
  return getModel(MOBILE_MODEL_ID).size;
}
