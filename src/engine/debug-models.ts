import { downloadModel, type DownloadProgress } from './model/download';
import type { ModelFileStore } from './model/file-store';
import {
  DEFAULT_MODEL_ID,
  isModelId,
  MODEL_LICENSES,
  MODEL_LIST,
  type ModelEntry,
  type ModelId,
  type PromptFormat,
} from './model/catalog';
import { buildModelPrompt, type ModelPrompt } from './prompt/model-prompt';

/**
 * Model switcher for the GPU debug harness only. Since 1.3.0 the models themselves are the
 * production catalog (`MODEL_CATALOG`); the harness downloads them through ModelManager like the
 * options page does. `HarnessModelStore` only reads or deletes copies an older harness build left
 * in its own OPFS directory.
 */

export type HarnessPromptFormat = PromptFormat;

/**
 * A harness view of a catalog model. Since 1.3.0 every harness model is a production model
 * (`MODEL_CATALOG`): the pins, sizes and prompt formats live there, and this only adds the
 * harness labels.
 */
export interface HarnessModel extends Omit<ModelEntry, 'license'> {
  /** Harness label (Traditional Chinese suffixes such as "GPU 實驗性"). */
  name: string;
  /** License display name. */
  license: string;
  /** One line, Traditional Chinese. */
  description: string;
}

/**
 * OPFS root directory where 1.3.1-dev harness builds stored the non-PLaMo models. 1.3.0 downloads
 * every model through ModelManager (shared with the options page); a copy left here is still
 * read and can be deleted from the harness.
 */
export const HARNESS_MODEL_DIRECTORY = 'sakutto-harness-models';

/** One line per model, Traditional Chinese (the harness page is Chinese-only). */
const HARNESS_DESCRIPTIONS: Readonly<Record<ModelId, string>> = {
  'qwen3-0.6b-q4_k_m': '最小最快的通用小模型，可當手機端速度與 GPU 正確性的基準，翻譯品質一般。',
  'plamo-3-translate-0.8b-q4_k_m': '目前產品預設的翻譯專用模型，日英中翻譯自然。',
  'gemma-3-1b-it-q4_k_m':
    'Google 的 1B 指令模型，多語能力均衡，架構與 PLaMo 不同，適合交叉比對 GPU 問題。',
  'qwen3-1.7b-q4_k_m': 'Qwen3 的較大版本，中文與英文翻譯明顯優於 0.6B，速度約慢一倍。',
  'hy-mt2-1.8b-q4_k_m': '騰訊混元的翻譯專用模型，支援 33 種語言互譯，品質佳。',
  'plamo-3-translate-2b-q4_k_m': 'PLaMo 3 Translate 的高品質版，檔案最大、手機記憶體壓力最高。',
};

function harnessName(model: ModelEntry): string {
  if (model.id === DEFAULT_MODEL_ID) return `${model.name}（目前預設）`;
  return model.gpuExperimental ? `${model.name}（GPU 實驗性）` : model.name;
}

/** All harness models (the whole production catalog), smallest file first. */
export const HARNESS_MODELS: readonly HarnessModel[] = Object.freeze(
  MODEL_LIST.map((model): HarnessModel => ({
    ...model,
    name: harnessName(model),
    license: MODEL_LICENSES[model.license].name,
    description: HARNESS_DESCRIPTIONS[model.id],
  })),
);

export const HARNESS_DEFAULT_MODEL_ID: string = DEFAULT_MODEL_ID;

export function isHarnessModelId(value: unknown): value is ModelId {
  return isModelId(value);
}

export function harnessModel(id: string): HarnessModel {
  const model = HARNESS_MODELS.find((m) => m.id === id);
  if (!model) throw new Error(`Unknown harness model: ${id}`);
  return model;
}

export function harnessModelPaths(id: string): { directory: string; part: string; final: string } {
  return { directory: id, part: `${id}/model.gguf.part`, final: `${id}/model.gguf` };
}

/** Decimal units, the way Hugging Face lists file sizes. */
export function formatModelSize(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${Math.round(bytes / 1e6)} MB`;
}

export function describeHarnessModel(model: HarnessModel): string {
  return `${model.name} · ${formatModelSize(model.size)} · ${model.license} — ${model.description}`;
}

export type HarnessPrompt = ModelPrompt;

/** Harness text: fixed English → Traditional Chinese, through the production prompt builder. */
export function buildHarnessPrompt(
  model: HarnessModel,
  input: { text: string; src: string; tgt: string },
): HarnessPrompt {
  return buildModelPrompt(model.format, input);
}

/**
 * Model copies an older harness build (1.3.1-dev) stored under its own OPFS directory, one
 * subdirectory per model. 1.3.0 downloads through ModelManager instead; this reads a leftover copy
 * so it is not downloaded twice, and deletes it on request.
 */
export class HarnessModelStore {
  constructor(
    private readonly store: ModelFileStore,
    private readonly download: typeof downloadModel = downloadModel,
  ) {}

  async isDownloaded(model: HarnessModel): Promise<boolean> {
    return (await this.store.size(harnessModelPaths(model.id).final)) === model.size;
  }

  async partialBytes(model: HarnessModel): Promise<number> {
    return this.store.size(harnessModelPaths(model.id).part);
  }

  async read(model: HarnessModel): Promise<Blob> {
    return this.store.read(harnessModelPaths(model.id).final);
  }

  async downloadFile(
    model: HarnessModel,
    signal: AbortSignal,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<void> {
    const paths = harnessModelPaths(model.id);
    await this.download({
      source: { url: model.url, size: model.size, sha256: model.sha256 },
      store: this.store,
      partPath: paths.part,
      finalPath: paths.final,
      signal,
      ...(onProgress ? { onProgress } : {}),
    });
  }

  async remove(model: HarnessModel): Promise<void> {
    await this.store.removeDirectory(harnessModelPaths(model.id).directory);
  }
}
