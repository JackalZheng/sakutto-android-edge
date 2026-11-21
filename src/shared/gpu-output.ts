import { browser } from 'wxt/browser';

/**
 * Set when WebGPU produced broken translations on this device and the engine moved to the
 * CPU. While it is set, an explicit WebGPU choice still runs on the CPU, so a phone whose
 * GPU computes wrong results does not retry it on every page. Choosing WebGPU again in the
 * options clears it.
 */
export const GPU_OUTPUT_BROKEN_KEY = 'webgpuOutputBroken';

export async function loadGpuOutputBroken(): Promise<boolean> {
  if (!browser.storage?.local) return false;
  try {
    const stored = await browser.storage.local.get(GPU_OUTPUT_BROKEN_KEY);
    return stored[GPU_OUTPUT_BROKEN_KEY] === true;
  } catch {
    return false;
  }
}

export async function saveGpuOutputBroken(broken: boolean): Promise<void> {
  if (!browser.storage?.local) return;
  try {
    if (broken) await browser.storage.local.set({ [GPU_OUTPUT_BROKEN_KEY]: true });
    else await browser.storage.local.remove(GPU_OUTPUT_BROKEN_KEY);
  } catch {
    // Best effort: without the flag the guard simply catches the GPU again next time.
  }
}
