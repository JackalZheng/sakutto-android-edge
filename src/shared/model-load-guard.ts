import { browser } from 'wxt/browser';
import { DEFAULT_MODEL_ID, isModelId, type ModelId } from '@/engine/model/catalog';
import { loadSettings, saveSettings } from './storage';

/**
 * Load-crash guard (1.3.0, Edge for Android).
 *
 * A model too large for the phone does not fail its load: Edge closes the engine page (and
 * sometimes the tab) while llama.cpp allocates it. 1.3.0 remembers the chosen model, so the
 * next engine page loaded the same model and was closed again. The engine host now marks a
 * load before it starts and clears the mark when it ends either way. A new engine page that
 * finds a mark left behind knows the previous one died inside a load: that model is recorded
 * as crashed, the settings go back to PLaMo 3 0.8B, and the model list says why.
 */
export const MODEL_LOAD_PENDING_KEY = 'modelLoadPending';
export const MODEL_LOAD_CRASHED_KEY = 'modelLoadCrashed';

/** A mark older than this is not a crash in a load; a load takes well under a minute. */
export const MODEL_LOAD_CRASH_WINDOW_MS = 10 * 60_000;

export interface PendingModelLoad {
  modelId: ModelId;
  at: number;
}

export type CrashedModelLoads = Partial<Record<ModelId, number>>;

export interface ModelLoadMarkUpdate {
  type: 'engine:model-load';
  /** The model whose load starts, or null when it ended. */
  modelId: ModelId | null;
}

export interface ModelLoadCrashQuery {
  type: 'engine:model-load-crash';
}

export function isModelLoadMarkUpdate(value: unknown): value is ModelLoadMarkUpdate {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    record.type === 'engine:model-load' && (record.modelId === null || isModelId(record.modelId))
  );
}

export function isModelLoadCrashQuery(value: unknown): value is ModelLoadCrashQuery {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as Record<string, unknown>).type === 'engine:model-load-crash'
  );
}

function normalizePending(value: unknown): PendingModelLoad | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  if (!isModelId(record.modelId) || typeof record.at !== 'number') return null;
  return { modelId: record.modelId, at: record.at };
}

export function normalizeCrashedModelLoads(value: unknown): CrashedModelLoads {
  const crashed: CrashedModelLoads = {};
  if (typeof value !== 'object' || value === null) return crashed;
  for (const [key, at] of Object.entries(value as Record<string, unknown>)) {
    if (isModelId(key) && typeof at === 'number') crashed[key] = at;
  }
  return crashed;
}

/**
 * The model a mark left behind says crashed, if any. The default model is never blamed: it
 * is the one to fall back to, and it was measured on the phone.
 */
export function crashedModelFromMark(pending: unknown, now: number): ModelId | null {
  const mark = normalizePending(pending);
  if (!mark || mark.modelId === DEFAULT_MODEL_ID) return null;
  if (now - mark.at < 0 || now - mark.at > MODEL_LOAD_CRASH_WINDOW_MS) return null;
  return mark.modelId;
}

// --- storage side (background, options; the offscreen host goes through messages) ---

export async function writeModelLoadMark(modelId: ModelId | null, now = Date.now()): Promise<void> {
  if (modelId === null) {
    await browser.storage.local.remove(MODEL_LOAD_PENDING_KEY);
    return;
  }
  await browser.storage.local.set({
    [MODEL_LOAD_PENDING_KEY]: { modelId, at: now } satisfies PendingModelLoad,
  });
}

/** Reads and clears a mark left behind; records the model it blames. */
export async function takeCrashedModelLoad(now = Date.now()): Promise<ModelId | null> {
  const stored = await browser.storage.local.get([MODEL_LOAD_PENDING_KEY, MODEL_LOAD_CRASHED_KEY]);
  const crashed = crashedModelFromMark(stored[MODEL_LOAD_PENDING_KEY], now);
  if (stored[MODEL_LOAD_PENDING_KEY] !== undefined) {
    await browser.storage.local.remove(MODEL_LOAD_PENDING_KEY);
  }
  if (!crashed) return null;
  const record = normalizeCrashedModelLoads(stored[MODEL_LOAD_CRASHED_KEY]);
  record[crashed] = now;
  await browser.storage.local.set({ [MODEL_LOAD_CRASHED_KEY]: record });
  return crashed;
}

/**
 * `takeCrashedModelLoad`, and the settings go back to the default model when the crashed one
 * was the chosen one, so the next engine page does not load it again.
 */
export async function recoverFromCrashedModelLoad(now = Date.now()): Promise<ModelId | null> {
  const crashed = await takeCrashedModelLoad(now);
  if (!crashed) return null;
  const settings = await loadSettings().catch(() => null);
  if (settings?.modelId === crashed) {
    await saveSettings({ modelId: DEFAULT_MODEL_ID }).catch(() => undefined);
  }
  return crashed;
}

export async function loadCrashedModelLoads(): Promise<CrashedModelLoads> {
  try {
    const stored = await browser.storage.local.get(MODEL_LOAD_CRASHED_KEY);
    return normalizeCrashedModelLoads(stored[MODEL_LOAD_CRASHED_KEY]);
  } catch {
    return {};
  }
}

/** A model that has since loaded and finished its load is no longer flagged. */
export async function clearCrashedModelLoad(modelId: ModelId): Promise<void> {
  const record = await loadCrashedModelLoads();
  if (record[modelId] === undefined) return;
  delete record[modelId];
  await browser.storage.local.set({ [MODEL_LOAD_CRASHED_KEY]: record });
}

// --- engine host side (no storage API in the offscreen document) ---

/** Best effort: a missing mark only turns the guard off for this load. */
export async function markModelLoad(modelId: ModelId | null): Promise<void> {
  try {
    if (browser.storage?.local) {
      await writeModelLoadMark(modelId);
      return;
    }
    await browser.runtime.sendMessage({
      type: 'engine:model-load',
      modelId,
    } satisfies ModelLoadMarkUpdate);
  } catch {
    // The guard is a safety net; translation goes on without it.
  }
}

/** Asked once when an engine host starts, before it loads anything. */
export async function requestCrashedModelLoad(): Promise<ModelId | null> {
  try {
    if (browser.storage?.local) return await recoverFromCrashedModelLoad();
    const response: unknown = await browser.runtime.sendMessage({
      type: 'engine:model-load-crash',
    } satisfies ModelLoadCrashQuery);
    return isModelId(response) ? response : null;
  } catch {
    return null;
  }
}
