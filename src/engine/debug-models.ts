import { downloadModel, type DownloadProgress } from './model/download';
import type { ModelFileStore } from './model/file-store';
import { DEFAULT_MODEL_ID, MODEL_CATALOG } from './model/catalog';
import { buildTranslationPrompt, TRANSLATION_STOP_TOKENS } from './prompt/pfml3';

/**
 * Model switcher for the GPU debug harness only. Nothing here is reachable
 * from the product flow: the production catalog, ModelManager and default
 * model are untouched. The extra models are stored in their own OPFS
 * directory and fetched through the same `downloadModel` (pinned commit,
 * size and sha256 checked) as the production model.
 */

export type HarnessPromptFormat = 'plamo3' | 'qwen3' | 'gemma3' | 'hy-mt';

export interface HarnessModel {
  id: string;
  name: string;
  /** llama.cpp architecture name in the GGUF (`general.architecture`). */
  architecture: string;
  repository: string;
  file: string;
  commit: string;
  size: number;
  sha256: string;
  url: string;
  license: string;
  /** One line, Traditional Chinese. */
  description: string;
  format: HarnessPromptFormat;
  /**
   * The production model: stored and managed by ModelManager (options page),
   * so the harness neither downloads a second copy nor deletes it.
   */
  production?: true;
}

/** OPFS root directory for harness-only models, separate from the product's `sakutto`. */
export const HARNESS_MODEL_DIRECTORY = 'sakutto-harness-models';

const HUGGING_FACE = 'https://huggingface.co';

function resolveUrl(repository: string, commit: string, file: string): string {
  return `${HUGGING_FACE}/${repository}/resolve/${commit}/${file}`;
}

function entry(model: Omit<HarnessModel, 'url'>): HarnessModel {
  return { ...model, url: resolveUrl(model.repository, model.commit, model.file) };
}

const plamo = MODEL_CATALOG[DEFAULT_MODEL_ID];

const MODELS: readonly HarnessModel[] = [
  entry({
    id: 'qwen3-0.6b-q4_k_m',
    name: 'Qwen3 0.6B（GPU 實驗性）',
    architecture: 'qwen3',
    repository: 'unsloth/Qwen3-0.6B-GGUF',
    file: 'Qwen3-0.6B-Q4_K_M.gguf',
    commit: '50968a4468ef4233ed78cd7c3de230dd1d61a56b',
    size: 396_705_472,
    sha256: 'ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a',
    license: 'Apache-2.0',
    description: '最小最快的通用小模型，可當手機端速度與 GPU 正確性的基準，翻譯品質一般。',
    format: 'qwen3',
  }),
  {
    id: plamo.id,
    name: 'PLaMo 3 Translate 0.8B（目前預設）',
    architecture: 'plamo3',
    repository: plamo.repository,
    file: plamo.file,
    commit: plamo.commit,
    size: plamo.size,
    sha256: plamo.sha256,
    url: plamo.url,
    license: 'PLaMo Community License',
    description: '目前產品使用的翻譯專用模型，日英中翻譯自然；與設定頁共用同一份下載。',
    format: 'plamo3',
    production: true,
  },
  entry({
    id: 'gemma-3-1b-it-q4_k_m',
    name: 'Gemma 3 1B IT（GPU 實驗性）',
    architecture: 'gemma3',
    repository: 'ggml-org/gemma-3-1b-it-GGUF',
    file: 'gemma-3-1b-it-Q4_K_M.gguf',
    commit: 'f9c28bcd85737ffc5aef028638d3341d49869c27',
    size: 806_058_240,
    sha256: '8ccc5cd1f1b3602548715ae25a66ed73fd5dc68a210412eea643eb20eb75a135',
    license: 'Gemma Terms of Use',
    description: 'Google 的 1B 指令模型，多語能力均衡，架構與 PLaMo 不同，適合交叉比對 GPU 問題。',
    format: 'gemma3',
  }),
  entry({
    id: 'qwen3-1.7b-q4_k_m',
    name: 'Qwen3 1.7B（GPU 實驗性）',
    architecture: 'qwen3',
    repository: 'unsloth/Qwen3-1.7B-GGUF',
    file: 'Qwen3-1.7B-Q4_K_M.gguf',
    commit: 'd7f544eead698dbd1f15126ef60b45a1e1933222',
    size: 1_107_409_472,
    sha256: 'b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897',
    license: 'Apache-2.0',
    description: 'Qwen3 的較大版本，中文與英文翻譯明顯優於 0.6B，速度約慢一倍。',
    format: 'qwen3',
  }),
  entry({
    id: 'hy-mt2-1.8b-q4_k_m',
    name: 'Hy-MT2 1.8B',
    architecture: 'hunyuan-dense',
    repository: 'tencent/Hy-MT2-1.8B-GGUF',
    file: 'Hy-MT2-1.8B-Q4_K_M.gguf',
    commit: 'a0c709d9fac510f2c807aa3af52872340dc37a4a',
    size: 1_133_080_448,
    sha256: 'dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699',
    license: 'Apache-2.0',
    description: '騰訊混元的翻譯專用模型，支援 33 種語言互譯，品質最好但檔案最大。',
    format: 'hy-mt',
  }),
];

/** All harness models, smallest file first. */
export const HARNESS_MODELS: readonly HarnessModel[] = Object.freeze(
  [...MODELS].sort((a, b) => a.size - b.size),
);

export const HARNESS_DEFAULT_MODEL_ID: string = plamo.id;

export function isHarnessModelId(value: unknown): value is string {
  return typeof value === 'string' && HARNESS_MODELS.some((model) => model.id === value);
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

export interface HarnessPrompt {
  prompt: string;
  stop: readonly string[];
}

/**
 * Instruction for the general chat models (Qwen3, Gemma 3). The harness text is
 * fixed English → Traditional Chinese, so the instruction is fixed too.
 */
const CHAT_INSTRUCTION = '請將以下英文翻譯成繁體中文（臺灣用語），只輸出譯文，不要任何解釋。';

/** Hy-MT2's model-card prompt for translation into Chinese, verbatim, target 繁体中文. */
const HY_MT_INSTRUCTION = '将以下文本翻译为繁体中文，注意只需要输出翻译后的结果，不要额外解释：';

/**
 * Each model's own chat format; decoding is greedy (temperature 0) with the
 * same max token count for every model.
 *
 * - PLaMo 3: the product's PFML3 prompt and stop tokens.
 * - Qwen3: ChatML with an empty `<think></think>` block, the template's
 *   `enable_thinking=False` form, so no reasoning is generated.
 * - Gemma 3: `<start_of_turn>` turns; the GGUF sets add_bos, so no `<bos>` text.
 * - Hy-MT2: the model card's chat template; add_bos is off in the GGUF, so the
 *   BOS text is part of the prompt.
 */
export function buildHarnessPrompt(
  model: HarnessModel,
  input: { text: string; src: string; tgt: string },
): HarnessPrompt {
  switch (model.format) {
    case 'plamo3':
      return { prompt: buildTranslationPrompt(input), stop: TRANSLATION_STOP_TOKENS };
    case 'qwen3':
      return {
        prompt:
          `<|im_start|>user\n${CHAT_INSTRUCTION}\n\n${input.text}<|im_end|>\n` +
          '<|im_start|>assistant\n<think>\n\n</think>\n\n',
        stop: ['<|im_end|>', '<|endoftext|>'],
      };
    case 'gemma3':
      return {
        prompt:
          `<start_of_turn>user\n${CHAT_INSTRUCTION}\n\n${input.text}<end_of_turn>\n` +
          '<start_of_turn>model\n',
        stop: ['<end_of_turn>', '<eos>'],
      };
    case 'hy-mt':
      return {
        prompt:
          `<｜hy_begin▁of▁sentence｜><｜hy_User｜>${HY_MT_INSTRUCTION}\n\n${input.text}` +
          '<｜hy_Assistant｜>',
        stop: ['<｜hy_place▁holder▁no▁2｜>', '<｜hy_end▁of▁sentence｜>', '<｜hy_EOT｜>'],
      };
  }
}

/**
 * Harness-only model files under their own OPFS directory, one subdirectory per
 * model. The production model is not stored here (see `HarnessModel.production`).
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
    this.assertHarnessOwned(model);
    return this.store.read(harnessModelPaths(model.id).final);
  }

  async downloadFile(
    model: HarnessModel,
    signal: AbortSignal,
    onProgress?: (progress: DownloadProgress) => void,
  ): Promise<void> {
    this.assertHarnessOwned(model);
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
    this.assertHarnessOwned(model);
    await this.store.removeDirectory(harnessModelPaths(model.id).directory);
  }

  private assertHarnessOwned(model: HarnessModel): void {
    if (model.production) {
      throw new Error(`${model.name} is managed by the options page, not the harness store`);
    }
  }
}
