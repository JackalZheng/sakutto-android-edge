import { browser } from 'wxt/browser';

/**
 * Set when WebGPU produced broken translations on this device and the engine moved to the
 * CPU. While it is set, an explicit WebGPU choice still runs on the CPU, so a phone whose
 * GPU computes wrong results does not retry it on every page. Choosing WebGPU again in the
 * options clears it.
 */
export const GPU_OUTPUT_BROKEN_KEY = 'webgpuOutputBroken';

export async function loadGpuOutputBroken(): Promise<boolean> {
  // The offscreen document has no storage API: the host reads the flag from the
  // background's settings snapshot instead (`requestEngineSettings`).
  if (!browser.storage?.local) return false;
  try {
    const stored = await browser.storage.local.get(GPU_OUTPUT_BROKEN_KEY);
    return stored[GPU_OUTPUT_BROKEN_KEY] === true;
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
    if (broken) await browser.storage.local.set({ [GPU_OUTPUT_BROKEN_KEY]: true });
    else await browser.storage.local.remove(GPU_OUTPUT_BROKEN_KEY);
  } catch {
    // Best effort: without the flag the guard simply catches the GPU again next time.
  }
}
