import { getModel, type ModelId } from './model/catalog';

/**
 * The model switcher menu for the GPU debug harness.
 *
 * Five models, sorted by file size ascending (sizes are the pinned catalog
 * sizes, so the order is PLaMo's actual size — 573 MB — not the "0.8B"
 * label). Each model downloads and caches separately under
 * `models/<modelId>/` through the shared `ModelManager`, so it can be
 * deleted on its own; the menu shows its downloaded state.
 *
 * The shipped model is PLaMo **3** Translate 0.8B — the request that asked
 * for this menu called it "PLaMo-2", but the repo ships PLaMo-3 and the
 * menu names what is actually installed. It is the control group: the
 * dedicated translator everything else is compared against.
 *
 * Licenses, file sizes and SHAs were verified 2026-11-21 against each
 * Hugging Face repo's file list; the catalog pins the commit, file and
 * digest. Descriptions are one line of Chinese each, as the menu shows
 * them.
 */

export interface HarnessModel {
  id: ModelId;
  /** Display name, with quantization, as the menu shows it. */
  name: string;
  license: string;
  /** One line of Chinese describing the model for the menu. */
  descriptionZh: string;
  /** Hugging Face repo page, for the source link in the menu. */
  source: string;
}

const MODELS: readonly HarnessModel[] = [
  {
    id: 'qwen3-0.6b-q4_k_m',
    name: 'Qwen3-0.6B Q4_K_M',
    license: 'Apache-2.0',
    descriptionZh: '通用小模型，速度最快；已關閉思考模式；品質待測。',
    source: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF',
  },
  {
    id: 'plamo-3-translate-0.8b-q4_k_m',
    name: 'PLaMo-3 Translate 0.8B Q4_K_M',
    license: 'PLaMo（自訂授權）',
    descriptionZh: '專用翻譯模型，現行預設；對照組。',
    source: 'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF',
  },
  {
    id: 'gemma-3-1b-it-q4_k_m',
    name: 'Gemma-3-1B-it Q4_K_M',
    license: 'Gemma',
    descriptionZh: '通用多語言模型。',
    source: 'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF',
  },
  {
    id: 'qwen3-1.7b-q4_k_m',
    name: 'Qwen3-1.7B Q4_K_M',
    license: 'Apache-2.0',
    descriptionZh: '比 0.6B 準確；已關閉思考模式。',
    source: 'https://huggingface.co/unsloth/Qwen3-1.7B-GGUF',
  },
  {
    id: 'hy-mt2-1.8b-q4_k_m',
    name: 'Hy-MT2-1.8B Q4_K_M',
    license: 'Apache-2.0',
    descriptionZh: '騰訊專用翻譯模型，支援繁體中文；重點候選。',
    source: 'https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF',
  },
];

/** Menu order: file size ascending, from the pinned catalog sizes. */
export const HARNESS_MODELS: readonly HarnessModel[] = Object.freeze(
  [...MODELS].sort((a, b) => getModel(a.id).size - getModel(b.id).size),
);

export const DEFAULT_HARNESS_MODEL_ID: ModelId = 'plamo-3-translate-0.8b-q4_k_m';

const HARNESS_MODEL_IDS: ReadonlySet<string> = new Set<string>(
  HARNESS_MODELS.map((model) => model.id),
);

export function isHarnessModelId(value: unknown): value is ModelId {
  return typeof value === 'string' && HARNESS_MODEL_IDS.has(value);
}

export function harnessModelName(modelId: ModelId): string {
  return HARNESS_MODELS.find((model) => model.id === modelId)?.name ?? modelId;
}
