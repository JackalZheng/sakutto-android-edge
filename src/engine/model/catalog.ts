/**
 * The translation models Sakutto can download and run, smallest file first (1.3.0: all six,
 * the four former GPU-harness models included). PLaMo 3 Translate 0.8B stays the default.
 *
 * Every entry is pinned to a Hugging Face commit, byte size and sha256; `downloadModel` checks
 * all three, and `tests/extension-network.test.ts` keeps the URLs on the allowlisted host.
 */
export type ModelId =
  | 'qwen3-0.6b-q4_k_m'
  | 'plamo-3-translate-0.8b-q4_k_m'
  | 'gemma-3-1b-it-q4_k_m'
  | 'qwen3-1.7b-q4_k_m'
  | 'hy-mt2-1.8b-q4_k_m'
  | 'plamo-3-translate-2b-q4_k_m';

/** How a model's translation prompt is built; see `engine/prompt/model-prompt.ts`. */
export type PromptFormat = 'plamo3' | 'qwen3' | 'gemma3' | 'hy-mt';

export type ModelLicenseId = 'plamo-community' | 'apache-2.0' | 'gemma';

export interface ModelLicense {
  id: ModelLicenseId;
  /** Display name, untranslated. */
  name: string;
  /** Where the full terms live (bundled copy for PLaMo, upstream for the others). */
  url: string;
}

export const MODEL_LICENSES: Readonly<Record<ModelLicenseId, ModelLicense>> = Object.freeze({
  'plamo-community': {
    id: 'plamo-community',
    name: 'PLaMo Community License',
    url: 'https://plamo.preferredai.jp/info/plamo-community-license-en',
  },
  'apache-2.0': {
    id: 'apache-2.0',
    name: 'Apache-2.0',
    url: 'https://www.apache.org/licenses/LICENSE-2.0',
  },
  gemma: {
    id: 'gemma',
    name: 'Gemma Terms of Use',
    url: 'https://ai.google.dev/gemma/terms',
  },
});

export interface ModelEntry {
  id: ModelId;
  /** Display name, untranslated (product names). */
  name: string;
  params: '0.6B' | '0.8B' | '1B' | '1.7B' | '1.8B' | '2B';
  quantization: 'Q4_K_M';
  /** llama.cpp architecture name in the GGUF (`general.architecture`). */
  architecture: 'plamo3' | 'qwen3' | 'gemma3' | 'hunyuan-dense';
  /** Transformer blocks (`<arch>.block_count`). */
  layers: number;
  format: PromptFormat;
  license: ModelLicenseId;
  /**
   * Known to drift from the CPU result on Adreno WebGPU (1.3.x phone runs). "Automatic" keeps
   * these on the CPU on an Adreno GPU; WebGPU stays available as an explicit choice.
   */
  gpuExperimental: boolean;
  /** Translation-only model (PLaMo 3 Translate, Hy-MT2) rather than a general chat model. */
  translationModel: boolean;
  repository: string;
  file: string;
  commit: string;
  size: number;
  sha256: string;
  url: string;
}

const HUGGING_FACE = 'https://huggingface.co';

function resolveUrl(repository: string, commit: string, file: string): string {
  return `${HUGGING_FACE}/${repository}/resolve/${commit}/${file}`;
}

function entry(model: Omit<ModelEntry, 'url'>): ModelEntry {
  return Object.freeze({ ...model, url: resolveUrl(model.repository, model.commit, model.file) });
}

const QWEN3_0_6B = entry({
  id: 'qwen3-0.6b-q4_k_m',
  name: 'Qwen3 0.6B',
  params: '0.6B',
  quantization: 'Q4_K_M',
  architecture: 'qwen3',
  layers: 28,
  format: 'qwen3',
  license: 'apache-2.0',
  gpuExperimental: true,
  translationModel: false,
  repository: 'unsloth/Qwen3-0.6B-GGUF',
  file: 'Qwen3-0.6B-Q4_K_M.gguf',
  commit: '50968a4468ef4233ed78cd7c3de230dd1d61a56b',
  size: 396_705_472,
  sha256: 'ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a',
});

const ZERO_EIGHT = entry({
  id: 'plamo-3-translate-0.8b-q4_k_m',
  name: 'PLaMo 3 Translate 0.8B',
  params: '0.8B',
  quantization: 'Q4_K_M',
  architecture: 'plamo3',
  layers: 16,
  format: 'plamo3',
  license: 'plamo-community',
  gpuExperimental: false,
  translationModel: true,
  repository: 'pfnet/plamo-3-translate-0.8b-GGUF',
  file: 'plamo-3-translate-0.8b-Q4_K_M.gguf',
  commit: 'dbeb0856cb0b004b484f190bb71dd65876297013',
  size: 573_123_200,
  sha256: '923b2f28a9856de1b8b3b610a7e59ded0b14d007466fd5dd526e6a9ecaadafdd',
});

const GEMMA3_1B = entry({
  id: 'gemma-3-1b-it-q4_k_m',
  name: 'Gemma 3 1B IT',
  params: '1B',
  quantization: 'Q4_K_M',
  architecture: 'gemma3',
  layers: 26,
  format: 'gemma3',
  license: 'gemma',
  gpuExperimental: true,
  translationModel: false,
  repository: 'ggml-org/gemma-3-1b-it-GGUF',
  file: 'gemma-3-1b-it-Q4_K_M.gguf',
  commit: 'f9c28bcd85737ffc5aef028638d3341d49869c27',
  size: 806_058_240,
  sha256: '8ccc5cd1f1b3602548715ae25a66ed73fd5dc68a210412eea643eb20eb75a135',
});

const QWEN3_1_7B = entry({
  id: 'qwen3-1.7b-q4_k_m',
  name: 'Qwen3 1.7B',
  params: '1.7B',
  quantization: 'Q4_K_M',
  architecture: 'qwen3',
  layers: 28,
  format: 'qwen3',
  license: 'apache-2.0',
  gpuExperimental: true,
  translationModel: false,
  repository: 'unsloth/Qwen3-1.7B-GGUF',
  file: 'Qwen3-1.7B-Q4_K_M.gguf',
  commit: 'd7f544eead698dbd1f15126ef60b45a1e1933222',
  size: 1_107_409_472,
  sha256: 'b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897',
});

const HY_MT2 = entry({
  id: 'hy-mt2-1.8b-q4_k_m',
  name: 'Hy-MT2 1.8B',
  params: '1.8B',
  quantization: 'Q4_K_M',
  architecture: 'hunyuan-dense',
  layers: 32,
  format: 'hy-mt',
  license: 'apache-2.0',
  gpuExperimental: false,
  translationModel: true,
  repository: 'tencent/Hy-MT2-1.8B-GGUF',
  file: 'Hy-MT2-1.8B-Q4_K_M.gguf',
  commit: 'a0c709d9fac510f2c807aa3af52872340dc37a4a',
  size: 1_133_080_448,
  sha256: 'dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699',
});

const TWO = entry({
  id: 'plamo-3-translate-2b-q4_k_m',
  name: 'PLaMo 3 Translate 2B',
  params: '2B',
  quantization: 'Q4_K_M',
  architecture: 'plamo3',
  layers: 24,
  format: 'plamo3',
  license: 'plamo-community',
  gpuExperimental: false,
  translationModel: true,
  repository: 'pfnet/plamo-3-translate-2b-GGUF',
  file: 'plamo-3-translate-2b-Q4_K_M.gguf',
  commit: '232fbd7916519edecddfdae0beae7e76fe70453f',
  size: 1_807_199_392,
  sha256: 'f36d0813dd95573a852ab5d76e374c543632d9e292929e922bcdbf7e53b494e0',
});

/** All models, smallest file first; this is the order the settings list them in. */
export const MODEL_LIST: readonly ModelEntry[] = Object.freeze(
  [QWEN3_0_6B, ZERO_EIGHT, GEMMA3_1B, QWEN3_1_7B, HY_MT2, TWO].sort((a, b) => a.size - b.size),
);

export const MODEL_IDS: readonly ModelId[] = Object.freeze(MODEL_LIST.map((model) => model.id));

export const MODEL_CATALOG: Readonly<Record<ModelId, ModelEntry>> = Object.freeze(
  Object.fromEntries(MODEL_LIST.map((model) => [model.id, model])) as Record<ModelId, ModelEntry>,
);

export const DEFAULT_MODEL_ID: ModelId = ZERO_EIGHT.id;

export const HIGH_QUALITY_MODEL_ID: ModelId = TWO.id;

const MODEL_ID_SET: ReadonlySet<string> = new Set<string>(MODEL_IDS);

export function isModelId(value: unknown): value is ModelId {
  return typeof value === 'string' && MODEL_ID_SET.has(value);
}

export function getModel(modelId: ModelId): ModelEntry {
  return MODEL_CATALOG[modelId];
}

/** The prompt format of a model id; unknown ids (tests, old records) get PLaMo's. */
export function modelPromptFormat(modelId: string): PromptFormat {
  return isModelId(modelId) ? MODEL_CATALOG[modelId].format : 'plamo3';
}

export function isGpuExperimentalModel(modelId: string): boolean {
  return isModelId(modelId) && MODEL_CATALOG[modelId].gpuExperimental;
}

export function modelLicense(modelId: ModelId): ModelLicense {
  return MODEL_LICENSES[MODEL_CATALOG[modelId].license];
}

/** Over 1 GB: the settings suggest the low-memory Adreno profile and warn on a phone. */
export const LARGE_MODEL_BYTES = 1_000_000_000;
