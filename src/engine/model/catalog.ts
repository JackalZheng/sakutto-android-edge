/**
 * Pinned model files, downloaded from Hugging Face and verified against the
 * pinned size and SHA-256 before use.
 *
 * The two PLaMo entries are the product models: the settings page offers only
 * those two, `isModelId` accepts only those two (so no production message can
 * select anything else), and `DEFAULT_MODEL_ID` stays the 0.8B model.
 *
 * The four extra entries (Qwen3 0.6B/1.7B, Gemma-3 1B, Hy-MT2 1.8B) exist
 * only for the GPU debug harness (`src/entrypoints/gpu-harness/`), which
 * A/B-compares translation quality and speed across models on the phone.
 * They are catalogued here because `ModelManager` is keyed by `ModelId` and
 * the harness reuses its per-model download cache (`models/<modelId>/`),
 * resume, digest check and per-model delete — nothing in the production
 * translation flow or the settings page ever addresses them.
 */
export type ModelId =
  | 'plamo-3-translate-0.8b-q4_k_m'
  | 'plamo-3-translate-2b-q4_k_m'
  | 'qwen3-0.6b-q4_k_m'
  | 'gemma-3-1b-it-q4_k_m'
  | 'qwen3-1.7b-q4_k_m'
  | 'hy-mt2-1.8b-q4_k_m';

export interface ModelEntry {
  id: ModelId;
  params: '0.6B' | '0.8B' | '1B' | '1.7B' | '1.8B' | '2B';
  quantization: 'Q4_K_M';
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

const ZERO_EIGHT: ModelEntry = {
  id: 'plamo-3-translate-0.8b-q4_k_m',
  params: '0.8B',
  quantization: 'Q4_K_M',
  repository: 'pfnet/plamo-3-translate-0.8b-GGUF',
  file: 'plamo-3-translate-0.8b-Q4_K_M.gguf',
  commit: 'dbeb0856cb0b004b484f190bb71dd65876297013',
  size: 573_123_200,
  sha256: '923b2f28a9856de1b8b3b610a7e59ded0b14d007466fd5dd526e6a9ecaadafdd',
  url: resolveUrl(
    'pfnet/plamo-3-translate-0.8b-GGUF',
    'dbeb0856cb0b004b484f190bb71dd65876297013',
    'plamo-3-translate-0.8b-Q4_K_M.gguf',
  ),
};

const TWO: ModelEntry = {
  id: 'plamo-3-translate-2b-q4_k_m',
  params: '2B',
  quantization: 'Q4_K_M',
  repository: 'pfnet/plamo-3-translate-2b-GGUF',
  file: 'plamo-3-translate-2b-Q4_K_M.gguf',
  commit: '232fbd7916519edecddfdae0beae7e76fe70453f',
  size: 1_807_199_392,
  sha256: 'f36d0813dd95573a852ab5d76e374c543632d9e292929e922bcdbf7e53b494e0',
  url: resolveUrl(
    'pfnet/plamo-3-translate-2b-GGUF',
    '232fbd7916519edecddfdae0beae7e76fe70453f',
    'plamo-3-translate-2b-Q4_K_M.gguf',
  ),
};

/**
 * Harness-only entries for the GPU debug harness model switcher. Sizes and
 * SHA-256 digests are the Hugging Face LFS values, verified 2026-11-21
 * against each repo's file list; the commit pins the repo revision the file
 * was listed at. The harness menu sorts by `size`, ascending.
 */
const QWEN3_ZERO_SIX: ModelEntry = {
  id: 'qwen3-0.6b-q4_k_m',
  params: '0.6B',
  quantization: 'Q4_K_M',
  repository: 'unsloth/Qwen3-0.6B-GGUF',
  file: 'Qwen3-0.6B-Q4_K_M.gguf',
  commit: '50968a4468ef4233ed78cd7c3de230dd1d61a56b',
  size: 396_705_472,
  sha256: 'ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a',
  url: resolveUrl(
    'unsloth/Qwen3-0.6B-GGUF',
    '50968a4468ef4233ed78cd7c3de230dd1d61a56b',
    'Qwen3-0.6B-Q4_K_M.gguf',
  ),
};

const GEMMA3_ONE_B: ModelEntry = {
  id: 'gemma-3-1b-it-q4_k_m',
  params: '1B',
  quantization: 'Q4_K_M',
  repository: 'ggml-org/gemma-3-1b-it-GGUF',
  file: 'gemma-3-1b-it-Q4_K_M.gguf',
  commit: 'f9c28bcd85737ffc5aef028638d3341d49869c27',
  size: 806_058_240,
  sha256: '8ccc5cd1f1b3602548715ae25a66ed73fd5dc68a210412eea643eb20eb75a135',
  url: resolveUrl(
    'ggml-org/gemma-3-1b-it-GGUF',
    'f9c28bcd85737ffc5aef028638d3341d49869c27',
    'gemma-3-1b-it-Q4_K_M.gguf',
  ),
};

const QWEN3_ONE_SEVEN: ModelEntry = {
  id: 'qwen3-1.7b-q4_k_m',
  params: '1.7B',
  quantization: 'Q4_K_M',
  repository: 'unsloth/Qwen3-1.7B-GGUF',
  file: 'Qwen3-1.7B-Q4_K_M.gguf',
  commit: 'd7f544eead698dbd1f15126ef60b45a1e1933222',
  size: 1_107_409_472,
  sha256: 'b139949c5bd74937ad8ed8c8cf3d9ffb1e99c866c823204dc42c0d91fa181897',
  url: resolveUrl(
    'unsloth/Qwen3-1.7B-GGUF',
    'd7f544eead698dbd1f15126ef60b45a1e1933222',
    'Qwen3-1.7B-Q4_K_M.gguf',
  ),
};

const HY_MT2_ONE_EIGHT: ModelEntry = {
  id: 'hy-mt2-1.8b-q4_k_m',
  params: '1.8B',
  quantization: 'Q4_K_M',
  repository: 'tencent/Hy-MT2-1.8B-GGUF',
  file: 'Hy-MT2-1.8B-Q4_K_M.gguf',
  commit: 'a0c709d9fac510f2c807aa3af52872340dc37a4a',
  size: 1_133_080_448,
  sha256: 'dc5f44fcf1fa496ee7ad725982c0c8c553a4de00259b53af84c4b89fb0c06699',
  url: resolveUrl(
    'tencent/Hy-MT2-1.8B-GGUF',
    'a0c709d9fac510f2c807aa3af52872340dc37a4a',
    'Hy-MT2-1.8B-Q4_K_M.gguf',
  ),
};

export const MODEL_CATALOG: Readonly<Record<ModelId, ModelEntry>> = Object.freeze({
  'plamo-3-translate-0.8b-q4_k_m': ZERO_EIGHT,
  'plamo-3-translate-2b-q4_k_m': TWO,
  'qwen3-0.6b-q4_k_m': QWEN3_ZERO_SIX,
  'gemma-3-1b-it-q4_k_m': GEMMA3_ONE_B,
  'qwen3-1.7b-q4_k_m': QWEN3_ONE_SEVEN,
  'hy-mt2-1.8b-q4_k_m': HY_MT2_ONE_EIGHT,
});

export const DEFAULT_MODEL_ID: ModelId = ZERO_EIGHT.id;

export const HIGH_QUALITY_MODEL_ID: ModelId = TWO.id;

/**
 * The product-selectable model ids, used to validate production messages
 * (`model:download`, `model:use`, …). Deliberately NOT every `ModelId`:
 * the harness-only entries must never be selectable from production code,
 * so they stay out of this set even though they are in the catalog.
 */
const MODEL_IDS: ReadonlySet<string> = new Set<ModelId>([ZERO_EIGHT.id, TWO.id]);

export function isModelId(value: unknown): value is ModelId {
  return typeof value === 'string' && MODEL_IDS.has(value);
}

export function getModel(modelId: ModelId): ModelEntry {
  return MODEL_CATALOG[modelId];
}
