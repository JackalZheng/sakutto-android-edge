import { browser } from 'wxt/browser';
import { DEFAULT_MODEL_ID, isModelId, type ModelId } from '@/engine/model/catalog';
import { loadSettings, saveSettings } from './storage';

/**
 * Load-crash guard (1.3.0, Edge for Android).
 *
 * A model too large for the phone does not fail its load: Edge closes the engine page (and
 * sometimes the tab) while llama.cpp allocates it. 1.3.0 remembers the chosen model, so the
 * next engine page loaded the same model and was closed again. The engine host now marks every
 * load attempt (model and backend) before it starts and clears the mark when the load ends
 * either way. A new engine page that finds a mark left behind knows the previous one died
 * inside that load, and the fallback depends on where it died:
 *
 * - on WebGPU: the model stays chosen and runs on the CPU from then on. On the POCO F5 Pro the
 *   models over 1 GB crashed only on WebGPU (GPU buffers come out of the same memory as the
 *   page) and ran fine on the CPU. Choosing WebGPU again in the settings retries it.
 * - on the CPU: the settings go back to PLaMo 3 0.8B, and the model list says why.
 */
export const MODEL_LOAD_PENDING_KEY = 'modelLoadPending';
export const MODEL_LOAD_CRASHED_KEY = 'modelLoadCrashed';

/** A mark older than this is not a crash in a load; a load takes well under a minute. */
export const MODEL_LOAD_CRASH_WINDOW_MS = 10 * 60_000;

/** Same values as `BackendKind` (engine/in-browser/backend.ts), kept here to stay dependency-free. */
export type LoadBackend = 'webgpu' | 'cpu-wasm';

export interface PendingModelLoad {
  modelId: ModelId;
  at: number;
  /** Missing on marks from dcdcf54: that build loaded large models on WebGPU, see below. */
  backend?: LoadBackend;
}

export interface CrashedModelLoad {
  at: number;
  backend: LoadBackend;
}

export type CrashedModelLoads = Partial<Record<ModelId, CrashedModelLoad>>;

export interface ModelLoadCrash {
  modelId: ModelId;
  backend: LoadBackend;
}

/** What a new engine host learns before its first load. */
export interface ModelLoadCrashReport {
  /** The load the previous engine page died in, if any. */
  crashed: ModelLoadCrash | null;
  /** Models that crashed on WebGPU here: they run on the CPU until WebGPU is chosen again. */
  gpuCrashed: ModelId[];
}

export interface ModelLoadMarkUpdate {
  type: 'engine:model-load';
  /** The model whose load starts, or null when it ended. */
  modelId: ModelId | null;
  backend?: LoadBackend;
}

export interface ModelLoadCrashQuery {
  type: 'engine:model-load-crash';
}

function isLoadBackend(value: unknown): value is LoadBackend {
  return value === 'webgpu' || value === 'cpu-wasm';
}

export function isModelLoadMarkUpdate(value: unknown): value is ModelLoadMarkUpdate {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  return (
    record.type === 'engine:model-load' &&
    (record.modelId === null || isModelId(record.modelId)) &&
    (record.backend === undefined || isLoadBackend(record.backend))
  );
}

export function isModelLoadCrashQuery(value: unknown): value is ModelLoadCrashQuery {
  return (
    typeof value === 'object' &&
    value !== null &&
    (value as Record<string, unknown>).type === 'engine:model-load-crash'
  );
}

export function isModelLoadCrashReport(value: unknown): value is ModelLoadCrashReport {
  if (typeof value !== 'object' || value === null) return false;
  const record = value as Record<string, unknown>;
  const crashed = record.crashed as Record<string, unknown> | null | undefined;
  return (
    Array.isArray(record.gpuCrashed) &&
    record.gpuCrashed.every(isModelId) &&
    (crashed === null ||
      (typeof crashed === 'object' &&
        crashed !== undefined &&
        isModelId(crashed.modelId) &&
        isLoadBackend(crashed.backend)))
  );
}

function normalizePending(value: unknown): Required<PendingModelLoad> | null {
  if (typeof value !== 'object' || value === null) return null;
  const record = value as Record<string, unknown>;
  if (!isModelId(record.modelId) || typeof record.at !== 'number') return null;
  // A mark without a backend comes from dcdcf54, which loaded large models on WebGPU on an
  // Adreno phone: blaming the GPU keeps the model usable on the CPU.
  return {
    modelId: record.modelId,
    at: record.at,
    backend: isLoadBackend(record.backend) ? record.backend : 'webgpu',
  };
}

export function normalizeCrashedModelLoads(value: unknown): CrashedModelLoads {
  const crashed: CrashedModelLoads = {};
  if (typeof value !== 'object' || value === null) return crashed;
  for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
    if (!isModelId(key)) continue;
    // dcdcf54 stored a bare timestamp; that build crashed on WebGPU (see `normalizePending`).
    if (typeof entry === 'number') crashed[key] = { at: entry, backend: 'webgpu' };
    else if (typeof entry === 'object' && entry !== null) {
      const record = entry as Record<string, unknown>;
      if (typeof record.at === 'number' && isLoadBackend(record.backend)) {
        crashed[key] = { at: record.at, backend: record.backend };
      }
    }
  }
  return crashed;
}

/**
 * The load a mark left behind says crashed, if any. The default model on the CPU is never
 * blamed: it is the one to fall back to, and it was measured on the phone. It can be blamed
 * on WebGPU, where the CPU is still there to fall back to.
 */
export function crashedLoadFromMark(pending: unknown, now: number): ModelLoadCrash | null {
  const mark = normalizePending(pending);
  if (!mark) return null;
  if (mark.modelId === DEFAULT_MODEL_ID && mark.backend === 'cpu-wasm') return null;
  if (now - mark.at < 0 || now - mark.at > MODEL_LOAD_CRASH_WINDOW_MS) return null;
  return { modelId: mark.modelId, backend: mark.backend };
}

export function gpuCrashedModels(record: CrashedModelLoads): ModelId[] {
  return (Object.keys(record) as ModelId[]).filter((id) => record[id]?.backend === 'webgpu');
}

// --- storage side (background, options; the offscreen host goes through messages) ---

export async function writeModelLoadMark(
  modelId: ModelId | null,
  backend: LoadBackend = 'cpu-wasm',
  now = Date.now(),
): Promise<void> {
  if (modelId === null) {
    await browser.storage.local.remove(MODEL_LOAD_PENDING_KEY);
    return;
  }
  await browser.storage.local.set({
    [MODEL_LOAD_PENDING_KEY]: { modelId, at: now, backend } satisfies PendingModelLoad,
  });
}

/** Reads and clears a mark left behind; records the load it blames. */
export async function takeCrashedModelLoad(now = Date.now()): Promise<ModelLoadCrash | null> {
  const stored = await browser.storage.local.get([MODEL_LOAD_PENDING_KEY, MODEL_LOAD_CRASHED_KEY]);
  const crashed = crashedLoadFromMark(stored[MODEL_LOAD_PENDING_KEY], now);
  if (stored[MODEL_LOAD_PENDING_KEY] !== undefined) {
    await browser.storage.local.remove(MODEL_LOAD_PENDING_KEY);
  }
  if (!crashed) return null;
  const record = normalizeCrashedModelLoads(stored[MODEL_LOAD_CRASHED_KEY]);
  record[crashed.modelId] = { at: now, backend: crashed.backend };
  await browser.storage.local.set({ [MODEL_LOAD_CRASHED_KEY]: record });
  return crashed;
}

/**
 * `takeCrashedModelLoad`, then the fallback: a crash on the CPU moves the settings back to the
 * default model (when the crashed one was the chosen one), so the next engine page does not load
 * it again. A crash on WebGPU keeps the model: the report lists it in `gpuCrashed`, and the
 * engine host runs it on the CPU.
 */
export async function recoverFromCrashedModelLoad(now = Date.now()): Promise<ModelLoadCrashReport> {
  const crashed = await takeCrashedModelLoad(now);
  if (crashed?.backend === 'cpu-wasm') {
    const settings = await loadSettings().catch(() => null);
    if (settings?.modelId === crashed.modelId) {
      await saveSettings({ modelId: DEFAULT_MODEL_ID }).catch(() => undefined);
    }
  }
  return { crashed, gpuCrashed: gpuCrashedModels(await loadCrashedModelLoads()) };
}

export async function loadCrashedModelLoads(): Promise<CrashedModelLoads> {
  try {
    const stored = await browser.storage.local.get(MODEL_LOAD_CRASHED_KEY);
    return normalizeCrashedModelLoads(stored[MODEL_LOAD_CRASHED_KEY]);
  } catch {
    return {};
  }
}

async function saveCrashedModelLoads(record: CrashedModelLoads): Promise<void> {
  await browser.storage.local.set({ [MODEL_LOAD_CRASHED_KEY]: record });
}

/**
 * Chosen again after a CPU crash: no longer flagged (the guard records it again if it crashes
 * again). A WebGPU crash record stays: the model keeps running on the CPU.
 */
export async function clearCrashedModelLoad(modelId: ModelId): Promise<void> {
  const record = await loadCrashedModelLoads();
  if (record[modelId]?.backend !== 'cpu-wasm') return;
  delete record[modelId];
  await saveCrashedModelLoads(record);
}

/** WebGPU chosen again in the settings: every model may try WebGPU again. */
export async function clearGpuCrashedModelLoads(): Promise<void> {
  const record = await loadCrashedModelLoads();
  const ids = gpuCrashedModels(record);
  if (ids.length === 0) return;
  for (const id of ids) delete record[id];
  await saveCrashedModelLoads(record);
}

// --- engine host side (no storage API in the offscreen document) ---

/** Best effort: a missing mark only turns the guard off for this load. */
export async function markModelLoad(
  modelId: ModelId | null,
  backend: LoadBackend = 'cpu-wasm',
): Promise<void> {
  try {
    if (browser.storage?.local) {
      await writeModelLoadMark(modelId, backend);
      return;
    }
    await browser.runtime.sendMessage({
      type: 'engine:model-load',
      modelId,
      backend,
    } satisfies ModelLoadMarkUpdate);
  } catch {
    // The guard is a safety net; translation goes on without it.
  }
}

const NO_CRASH: ModelLoadCrashReport = { crashed: null, gpuCrashed: [] };

/** Asked once when an engine host starts, before it loads anything. */
export async function requestCrashedModelLoad(): Promise<ModelLoadCrashReport> {
  try {
    if (browser.storage?.local) return await recoverFromCrashedModelLoad();
    const response: unknown = await browser.runtime.sendMessage({
      type: 'engine:model-load-crash',
    } satisfies ModelLoadCrashQuery);
    return isModelLoadCrashReport(response) ? response : NO_CRASH;
  } catch {
    return NO_CRASH;
  }
}
