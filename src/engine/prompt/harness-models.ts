import { EngineError } from '../errors';
import type { ModelId } from '../model/catalog';
import { buildTranslationPrompt, TRANSLATION_STOP_TOKENS } from './pfml3';

/**
 * Per-model prompts for the GPU debug harness model switcher.
 *
 * The harness runs one fixed paragraph through several models on the phone
 * and compares translation quality and speed, so every model gets the same
 * instruction — translate into Traditional Chinese and output only the
 * translation — rendered in that model's own chat format. Decoding stays
 * greedy (the runtime uses temperature 0) and the token budget stays
 * `HARNESS_MAX_TOKENS`, the same as the PLaMo reference, so the numbers
 * compare like for like.
 *
 * The renderings below are hand-written from each GGUF file's embedded
 * `tokenizer.chat_template`, verified 2026-11-21 against the files the
 * catalog pins — not from memory of the public model cards:
 *
 * - Qwen3 (0.6B and 1.7B): ChatML. The embedded template renders the
 *   generation prompt as `<|im_start|>assistant\n` plus an empty
 *   `<think>` block when `enable_thinking=false`, so that is what this
 *   builder emits — thinking off, exactly as the template does it. (The
 *   model card's equivalent is appending `/no_think` to the user message.)
 *   The GGUF has `add_bos_token=false` and the template carries no BOS, so
 *   no BOS token is prepended here either.
 * - Gemma-3: `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n`.
 *   The GGUF has `add_bos_token=true`, so llama.cpp prepends `<bos>` and
 *   the prompt omits it — the same convention as the PLaMo builder, whose
 *   comment notes llama.cpp prepends the BOS when tokenizing.
 * - Hy-MT2: the embedded template for one user turn with no system prompt
 *   is `<｜hy_begin▁of▁sentence｜><｜hy_User｜>…<｜hy_Assistant｜>`, and
 *   the instruction wording is the model card's default-translation
 *   template, with the target language as its full name ("Traditional
 *   Chinese" — the card says source/target languages take full names).
 *   The GGUF carries no `add_bos_token` key and the vendored llama.cpp
 *   does not set `add_bos` for the `hunyuan-dense` pre-tokenizer, so
 *   llama.cpp prepends nothing here: the BOS token
 *   `<｜hy_begin▁of▁sentence｜>` is included explicitly as the template's
 *   first token. (The special tokens use full-width ｜ U+FF5C and ▁ U+2581.)
 * - PLaMo: the existing PFML3 builder, unchanged — it is the control group.
 *
 * Stop tokens are the model's turn-end token, mirroring the PLaMo pattern
 * of passing an explicit stop list to the runtime (llama.cpp also stops at
 * the model's EOS on its own).
 */

export interface HarnessModelPrompt {
  prompt: string;
  /** Turn-end strings the runtime stops generation on. */
  stop: readonly string[];
}

const QWEN3_GENERATION_PROMPT = '<|im_start|>assistant\n<think>\n\n</think>\n\n';

const HY_BEGIN = '<｜hy_begin▁of▁sentence｜>';
const HY_USER = '<｜hy_User｜>';
const HY_ASSISTANT = '<｜hy_Assistant｜>';
const HY_END_OF_TURN = '<｜hy_place▁holder▁no▁2｜>';

export const HARNESS_MODEL_STOP_TOKENS: Readonly<Record<ModelId, readonly string[]>> =
  Object.freeze({
    'plamo-3-translate-0.8b-q4_k_m': TRANSLATION_STOP_TOKENS,
    'plamo-3-translate-2b-q4_k_m': TRANSLATION_STOP_TOKENS,
    'qwen3-0.6b-q4_k_m': ['<|im_end|>'],
    'gemma-3-1b-it-q4_k_m': ['<end_of_turn>'],
    'qwen3-1.7b-q4_k_m': ['<|im_end|>'],
    'hy-mt2-1.8b-q4_k_m': [HY_END_OF_TURN],
  });

/**
 * Full language name for the chat-model instruction, as the Hy-MT2 card
 * requires ("Traditional Chinese", not a code). Unknown tags pass through
 * unchanged; the harness only ever passes `zh-hant`.
 */
export function harnessTargetLanguageName(tgt: string): string {
  switch (tgt.toLowerCase().replaceAll('_', '-')) {
    case 'zh-hant':
      return 'Traditional Chinese';
    case 'zh-hans':
      return 'Simplified Chinese';
    case 'zh':
      return 'Chinese';
    case 'ja':
      return 'Japanese';
    case 'en':
      return 'English';
    default:
      return tgt;
  }
}

function checkHarnessText(text: string): string {
  if (typeof text !== 'string' || text.trim().length === 0) {
    throw new EngineError('invalid-input', 'text must not be empty.');
  }
  return text;
}

/**
 * The shared instruction for the chat models, worded after the Hy-MT2
 * model card's default-translation template. PLaMo keeps its own PFML3
 * prompt (the control group must not change).
 */
function chatInstruction(text: string, tgt: string): string {
  const target = harnessTargetLanguageName(tgt);
  return (
    `Translate the following text into ${target}. ` +
    'Note that you should only output the translated result without any additional explanation:' +
    `\n\n${text}`
  );
}

export function buildHarnessModelPrompt(
  modelId: ModelId,
  text: string,
  src: string,
  tgt: string,
): HarnessModelPrompt {
  const checked = checkHarnessText(text);
  switch (modelId) {
    case 'plamo-3-translate-0.8b-q4_k_m':
    case 'plamo-3-translate-2b-q4_k_m':
      return {
        prompt: buildTranslationPrompt({ text: checked, src, tgt }),
        stop: HARNESS_MODEL_STOP_TOKENS[modelId],
      };
    case 'qwen3-0.6b-q4_k_m':
    case 'qwen3-1.7b-q4_k_m':
      return {
        prompt:
          `<|im_start|>user\n${chatInstruction(checked, tgt)}<|im_end|>\n` +
          QWEN3_GENERATION_PROMPT,
        stop: HARNESS_MODEL_STOP_TOKENS[modelId],
      };
    case 'gemma-3-1b-it-q4_k_m':
      return {
        prompt:
          `<start_of_turn>user\n${chatInstruction(checked, tgt)}<end_of_turn>\n` +
          '<start_of_turn>model\n',
        stop: HARNESS_MODEL_STOP_TOKENS[modelId],
      };
    case 'hy-mt2-1.8b-q4_k_m':
      return {
        prompt: `${HY_BEGIN}${HY_USER}${chatInstruction(checked, tgt)}${HY_ASSISTANT}`,
        stop: HARNESS_MODEL_STOP_TOKENS[modelId],
      };
  }
}
