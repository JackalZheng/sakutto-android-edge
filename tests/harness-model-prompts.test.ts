import { describe, expect, it } from 'vitest';
import {
  DEFAULT_HARNESS_MODEL_ID,
  HARNESS_MODELS,
  harnessModelName,
  isHarnessModelId,
} from '../src/engine/debug-harness-models';
import { getModel } from '../src/engine/model/catalog';
import {
  buildHarnessModelPrompt,
  HARNESS_MODEL_STOP_TOKENS,
  harnessTargetLanguageName,
} from '../src/engine/prompt/harness-models';
import { buildTranslationPrompt } from '../src/engine/prompt/pfml3';

const TEXT = 'The morning train leaves the quiet station.';

describe('HARNESS_MODELS menu', () => {
  it('lists five models, sorted by file size ascending', () => {
    expect(HARNESS_MODELS).toHaveLength(5);
    const sizes = HARNESS_MODELS.map((model) => getModel(model.id).size);
    expect(sizes).toEqual([396_705_472, 573_123_200, 806_058_240, 1_107_409_472, 1_133_080_448]);
    expect(HARNESS_MODELS.map((model) => model.id)).toEqual([
      'qwen3-0.6b-q4_k_m',
      'plamo-3-translate-0.8b-q4_k_m',
      'gemma-3-1b-it-q4_k_m',
      'qwen3-1.7b-q4_k_m',
      'hy-mt2-1.8b-q4_k_m',
    ]);
  });

  it('shows a name, license, one-line Chinese description and source for every model', () => {
    for (const model of HARNESS_MODELS) {
      expect(model.name.length).toBeGreaterThan(0);
      expect(model.license.length).toBeGreaterThan(0);
      expect(model.descriptionZh.length).toBeGreaterThan(0);
      expect(model.descriptionZh).not.toContain('\n');
      expect(model.source.startsWith('https://huggingface.co/')).toBe(true);
    }
  });

  it('defaults to the PLaMo control group', () => {
    expect(DEFAULT_HARNESS_MODEL_ID).toBe('plamo-3-translate-0.8b-q4_k_m');
    expect(HARNESS_MODELS.some((model) => model.id === DEFAULT_HARNESS_MODEL_ID)).toBe(true);
  });

  it('accepts only the five menu models', () => {
    expect(isHarnessModelId('qwen3-0.6b-q4_k_m')).toBe(true);
    expect(isHarnessModelId('plamo-3-translate-2b-q4_k_m')).toBe(false);
    expect(isHarnessModelId('plamo-3-translate-8b-q4_k_m')).toBe(false);
    expect(isHarnessModelId(null)).toBe(false);
  });

  it('names models by their display name', () => {
    expect(harnessModelName('hy-mt2-1.8b-q4_k_m')).toBe('Hy-MT2-1.8B Q4_K_M');
    expect(harnessModelName('plamo-3-translate-0.8b-q4_k_m')).toContain('PLaMo-3');
  });
});

describe('harnessTargetLanguageName', () => {
  it('uses full language names for the chat-model instruction', () => {
    expect(harnessTargetLanguageName('zh-hant')).toBe('Traditional Chinese');
    expect(harnessTargetLanguageName('zh-hans')).toBe('Simplified Chinese');
    expect(harnessTargetLanguageName('ja')).toBe('Japanese');
    expect(harnessTargetLanguageName('en')).toBe('English');
  });
});

describe('buildHarnessModelPrompt', () => {
  it('keeps the PLaMo prompt exactly as the production builder makes it', () => {
    const built = buildHarnessModelPrompt('plamo-3-translate-0.8b-q4_k_m', TEXT, 'en', 'zh-hant');
    expect(built.prompt).toBe(buildTranslationPrompt({ text: TEXT, src: 'en', tgt: 'zh-hant' }));
    expect(built.stop).toEqual(['<|plamo:tag|>', '<|plamo:bos|>']);
  });

  it('renders Qwen3 as ChatML with thinking off via the empty think block', () => {
    for (const id of ['qwen3-0.6b-q4_k_m', 'qwen3-1.7b-q4_k_m'] as const) {
      const built = buildHarnessModelPrompt(id, TEXT, 'en', 'zh-hant');
      expect(built.prompt).toBe(
        `<|im_start|>user\nTranslate the following text into Traditional Chinese. ` +
          'Note that you should only output the translated result without any additional explanation:' +
          `\n\n${TEXT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`,
      );
      expect(built.stop).toEqual(['<|im_end|>']);
    }
  });

  it('renders Gemma-3 with its turn markers and no BOS token', () => {
    const built = buildHarnessModelPrompt('gemma-3-1b-it-q4_k_m', TEXT, 'en', 'zh-hant');
    expect(built.prompt).toBe(
      `<start_of_turn>user\nTranslate the following text into Traditional Chinese. ` +
        'Note that you should only output the translated result without any additional explanation:' +
        `\n\n${TEXT}<end_of_turn>\n<start_of_turn>model\n`,
    );
    expect(built.prompt.startsWith('<bos>')).toBe(false);
    expect(built.stop).toEqual(['<end_of_turn>']);
  });

  it('renders Hy-MT2 with its sentence-start token and full language name', () => {
    const built = buildHarnessModelPrompt('hy-mt2-1.8b-q4_k_m', TEXT, 'en', 'zh-hant');
    expect(built.prompt).toBe(
      '<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into Traditional Chinese. ' +
        'Note that you should only output the translated result without any additional explanation:' +
        `\n\n${TEXT}<｜hy_Assistant｜>`,
    );
    expect(built.stop).toEqual(['<｜hy_place▁holder▁no▁2｜>']);
  });

  it('rejects an empty text', () => {
    expect(() => buildHarnessModelPrompt('qwen3-0.6b-q4_k_m', '  ', 'en', 'zh-hant')).toThrow(
      'text must not be empty.',
    );
  });
});

describe('HARNESS_MODEL_STOP_TOKENS', () => {
  it('has a stop list for every catalog model', () => {
    for (const model of HARNESS_MODELS) {
      expect(HARNESS_MODEL_STOP_TOKENS[model.id].length).toBeGreaterThan(0);
    }
  });
});
