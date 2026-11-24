/**
 * Mobile / Android Edge helpers.
 *
 * Edge for Android runs Chromium MV3 extensions, but it is not desktop
 * Chrome: there is no physical keyboard (so `commands` shortcuts are
 * useless), the built-in Chrome Translator API does not exist, the
 * service worker / offscreen document is killed more aggressively, and
 * most Android GPUs fall back to the WASM CPU path. The port therefore
 * treats Edge on Android as a touch-first target. Until 1.2.x only PLaMo 3 0.8B was offered;
 * 1.3.0 offers every catalog model (the settings warn about the > 1 GB ones on a phone).
 */

import { getModel, isModelId, type ModelId } from '@/engine/model/catalog';

/** The default model on Edge for Android: 573 MB, the one the onboarding downloads. */
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

/**
 * 1.3.0: every catalog model may be downloaded and used on a phone, PLaMo 3 2B included.
 *
 * Byte math (llama.cpp buffers, Q4_K_M, measured with the same build on the box; on a phone
 * the Adreno "GPU memory" is the same RAM as the tab's WASM heap):
 * - PLaMo 3 0.8B (works on both): weights 544 MiB (CPU 218: token_embd + the 129 MiB output
 *   matrix, which is over the 128 MiB storage-binding limit; GPU 326), KV 16, compute 133.
 * - Hy-MT2 1.8B, 2 slots / n_ctx 2048 / n_batch 256: weights 1076 MiB (CPU 194 tied
 *   token_embd, GPU 882), KV 128 (32 full-attention layers), compute 148: ~1.32 GiB.
 * - PLaMo 3 2B, same defaults: weights 1721 MiB (CPU 363, GPU 1358), KV 96, compute ~150.
 * On a POCO F5 Pro both closed the engine page in a WebGPU load and both ran fine on the CPU,
 * where the weights stay in the WASM heap and no GPU buffers are allocated on top. So a model
 * over 1 GB runs on the CPU on a phone (`phoneKeepsOffGpu` in engine/in-browser/backend.ts),
 * with the large-model limits (1 slot, n_ctx 1024, n_batch 128), and the load-crash guard
 * (`shared/model-load-guard.ts`) still catches a load that closes the page.
 */
export function isMobileModelAllowed(modelId: ModelId): boolean {
  return isModelId(modelId);
}

export function mobileModelSizeBytes(): number {
  return getModel(MOBILE_MODEL_ID).size;
}
