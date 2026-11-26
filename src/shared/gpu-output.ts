import { browser } from 'wxt/browser';

/**
 * Set when WebGPU produced broken translations on this device and the engine moved to the
 * CPU. While it is set, "auto" and an explicit WebGPU choice run on the CPU, so a phone whose
 * GPU computes wrong results does not retry it on every page. Choosing WebGPU again in the
 * options clears it.
 *
 * Since the 1.3.0 store release the stored value is the extension version that saw the broken
 * output, and the flag only counts for that version: an update ships new GPU kernels (1.3.0
 * made the round-6 kernels the Adreno default and "auto" use them), so a verdict about the old
 * kernels must not keep the new ones off for good. 1.2.x and the 1.3.0 development build stored
 * `true`, which therefore no longer counts; a 1.3.1 development build stored "1.3.1", which
 * does not match "1.3.0" either. If the new kernels are broken too, the output guard catches it
 * once and sets the flag again for this version. GPU-experimental models never set it (host.ts).
 */
export const GPU_OUTPUT_BROKEN_KEY = 'webgpuOutputBroken';

function extensionVersion(): string | null {
  try {
    const version = browser.runtime?.getManifest?.().version;
    return typeof version === 'string' && version.length > 0 ? version : null;
  } catch {
    return null;
  }
}

/** True when the stored flag belongs to this extension version (see above). */
export function gpuOutputBrokenForVersion(stored: unknown, version: string | null): boolean {
  return version !== null && stored === version;
}

export async function loadGpuOutputBroken(): Promise<boolean> {
  // The offscreen document has no storage API: the host reads the flag from the
  // background's settings snapshot instead (`requestEngineSettings`).
  if (!browser.storage?.local) return false;
  try {
    const stored = await browser.storage.local.get(GPU_OUTPUT_BROKEN_KEY);
    return gpuOutputBrokenForVersion(stored[GPU_OUTPUT_BROKEN_KEY], extensionVersion());
  } catch {
    return false;
  }
}

export async function saveGpuOutputBroken(broken: boolean): Promise<void> {
  if (!browser.storage?.local) {
    // Offscreen document: the background owns storage.
    try {
      await browser.runtime.sendMessage({ type: 'engine:gpu-output-broken', broken });
    } catch {
      // Best effort, as below.
    }
    return;
  }
  try {
    const version = extensionVersion();
    if (broken && version !== null) {
      await browser.storage.local.set({ [GPU_OUTPUT_BROKEN_KEY]: version });
    } else {
      await browser.storage.local.remove(GPU_OUTPUT_BROKEN_KEY);
    }
  } catch {
    // Best effort: without the flag the guard simply catches the GPU again next time.
  }
}
