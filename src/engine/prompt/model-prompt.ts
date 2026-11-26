import type { PromptFormat } from '../model/catalog';
import type { Formality, GlossaryEntry } from '../types';
import type { LanguageCode } from '@/shared/languages';
import {
  buildTranslationPrompt,
  checkLanguageTag,
  cleanModelOutput,
  TRANSLATION_STOP_TOKENS,
  type PromptInput,
} from './pfml3';
import { EngineError } from '../errors';

/**
 * Production translation prompts for every model format (1.3.0).
 *
 * - `plamo3`: PFML3 (`pfml3.ts`), unchanged.
 * - `qwen3`: ChatML with an empty `<think></think>` block (the chat template's
 *   `enable_thinking=False` form), so no reasoning is generated; `<think>` blocks are still
 *   stripped from the output in case one appears.
 * - `gemma3`: `<start_of_turn>` turns; the GGUF sets add_bos, so there is no literal `<bos>`.
 * - `hy-mt`: Hy-MT2's model-card templates (Chinese instruction when either side is Chinese,
 *   English otherwise; terminology, background and style forms); add_bos is off in the GGUF, so the BOS
 *   text is part of the prompt.
 *
 * Decoding is greedy for every model (`runtime.ts`).
 */

export interface ModelPrompt {
  prompt: string;
  stop: readonly string[];
}

export interface ModelPromptInput {
  text: string;
  tgt: string;
  src?: string | undefined;
  form?: Formality | undefined;
  glossary?: readonly GlossaryEntry[] | undefined;
  context?: string | undefined;
}

const QWEN3_STOP: readonly string[] = ['<|im_end|>', '<|endoftext|>', '<|im_start|>'];
const GEMMA3_STOP: readonly string[] = ['<end_of_turn>', '<eos>', '<start_of_turn>'];
const HY_MT_STOP: readonly string[] = [
  '<｜hy_place▁holder▁no▁2｜>',
  '<｜hy_end▁of▁sentence｜>',
  '<｜hy_EOT｜>',
  '<｜hy_User｜>',
];

export function stopTokensFor(format: PromptFormat): readonly string[] {
  switch (format) {
    case 'plamo3':
      return TRANSLATION_STOP_TOKENS;
    case 'qwen3':
      return QWEN3_STOP;
    case 'gemma3':
      return GEMMA3_STOP;
    case 'hy-mt':
      return HY_MT_STOP;
  }
}

/** Control-token shapes of the chat formats; removed from user text so it cannot open a turn. */
const SPECIAL_TOKEN_PATTERN =
  /<\|(?:im_start|im_end|endoftext|plamo:[a-z]+)\|>|<\/?(?:start_of_turn|end_of_turn|bos|eos)>|<｜hy_[^｜]{1,40}｜>/gu;

function stripSpecialTokens(value: string): string {
  return value.replace(SPECIAL_TOKEN_PATTERN, '');
}

function checkText(value: string, field: string): string {
  if (typeof value !== 'string') {
    throw new EngineError('invalid-input', `${field} must be a string.`);
  }
  const clean = stripSpecialTokens(value);
  if (clean.trim().length === 0) {
    throw new EngineError('invalid-input', `${field} must not be empty.`);
  }
  return clean;
}

/** English names for the chat-model instructions and Hy-MT2's English template. */
export const LANGUAGE_NAMES_EN: Readonly<Record<LanguageCode, string>> = {
  ja: 'Japanese',
  en: 'English',
  'zh-hans': 'Simplified Chinese',
  'zh-hant': 'Traditional Chinese',
  ko: 'Korean',
  ar: 'Arabic',
  az: 'Azerbaijani',
  be: 'Belarusian',
  bg: 'Bulgarian',
  bn: 'Bengali',
  ca: 'Catalan',
  ceb: 'Cebuano',
  de: 'German',
  es: 'Spanish',
  et: 'Estonian',
  eu: 'Basque',
  fa: 'Persian',
  fr: 'French',
  gu: 'Gujarati',
  he: 'Hebrew',
  hi: 'Hindi',
  hr: 'Croatian',
  hu: 'Hungarian',
  hy: 'Armenian',
  id: 'Indonesian',
  it: 'Italian',
  jv: 'Javanese',
  ka: 'Georgian',
  kk: 'Kazakh',
  km: 'Khmer',
  lt: 'Lithuanian',
  lv: 'Latvian',
  ml: 'Malayalam',
  mr: 'Marathi',
  ms: 'Malay',
  my: 'Burmese',
  ne: 'Nepali',
  nl: 'Dutch',
  pa: 'Punjabi',
  pl: 'Polish',
  ru: 'Russian',
  si: 'Sinhala',
  sk: 'Slovak',
  sl: 'Slovenian',
  sr: 'Serbian',
  sw: 'Swahili',
  ta: 'Tamil',
  te: 'Telugu',
  th: 'Thai',
  tl: 'Filipino',
  ur: 'Urdu',
  uz: 'Uzbek',
  vi: 'Vietnamese',
};

/** Chinese names for Hy-MT2's Chinese template (the model card's own wording where it has one). */
export const LANGUAGE_NAMES_ZH: Readonly<Record<LanguageCode, string>> = {
  ja: '日语',
  en: '英语',
  'zh-hans': '中文',
  'zh-hant': '繁体中文',
  ko: '韩语',
  ar: '阿拉伯语',
  az: '阿塞拜疆语',
  be: '白俄罗斯语',
  bg: '保加利亚语',
  bn: '孟加拉语',
  ca: '加泰罗尼亚语',
  ceb: '宿务语',
  de: '德语',
  es: '西班牙语',
  et: '爱沙尼亚语',
  eu: '巴斯克语',
  fa: '波斯语',
  fr: '法语',
  gu: '古吉拉特语',
  he: '希伯来语',
  hi: '印地语',
  hr: '克罗地亚语',
  hu: '匈牙利语',
  hy: '亚美尼亚语',
  id: '印尼语',
  it: '意大利语',
  jv: '爪哇语',
  ka: '格鲁吉亚语',
  kk: '哈萨克语',
  km: '高棉语',
  lt: '立陶宛语',
  lv: '拉脱维亚语',
  ml: '马拉雅拉姆语',
  mr: '马拉地语',
  ms: '马来语',
  my: '缅甸语',
  ne: '尼泊尔语',
  nl: '荷兰语',
  pa: '旁遮普语',
  pl: '波兰语',
  ru: '俄语',
  si: '僧伽罗语',
  sk: '斯洛伐克语',
  sl: '斯洛文尼亚语',
  sr: '塞尔维亚语',
  sw: '斯瓦希里语',
  ta: '泰米尔语',
  te: '泰卢固语',
  th: '泰语',
  tl: '菲律宾语',
  ur: '乌尔都语',
  uz: '乌兹别克语',
  vi: '越南语',
};

function englishName(code: string): string {
  return (LANGUAGE_NAMES_EN as Record<string, string>)[code] ?? code;
}

function chineseName(code: string): string {
  return (LANGUAGE_NAMES_ZH as Record<string, string>)[code] ?? englishName(code);
}

function isChinese(code: string | undefined): boolean {
  return code === 'zh-hans' || code === 'zh-hant';
}

interface CheckedInput {
  text: string;
  tgt: string;
  src: string | undefined;
  form: Formality | undefined;
  glossary: GlossaryEntry[];
  context: string | undefined;
}

function checkInput(input: ModelPromptInput): CheckedInput {
  const text = checkText(input.text, 'text');
  const tgt = checkLanguageTag(input.tgt, 'target_lang');
  const src =
    input.src !== undefined && input.src.trim().length > 0
      ? checkLanguageTag(input.src, 'source_lang')
      : undefined;
  if (input.form !== undefined && tgt !== 'ja') {
    throw new EngineError('invalid-input', 'form requires target_lang ja.');
  }
  const glossary: GlossaryEntry[] = [];
  for (const entry of input.glossary ?? []) {
    const source = stripSpecialTokens(entry.source).trim();
    const target = stripSpecialTokens(entry.target).trim();
    if (source.length > 0 && target.length > 0) glossary.push({ source, target });
  }
  const context =
    input.context !== undefined && stripSpecialTokens(input.context).trim().length > 0
      ? stripSpecialTokens(input.context).trim()
      : undefined;
  return { text, tgt, src, form: input.form, glossary, context };
}

/**
 * Target wording for the chat models. The Chinese script in its own characters kept Qwen3 0.6B
 * from answering in Simplified Chinese for a Traditional target (box llama.cpp runs).
 */
function chatTarget(tgt: string): string {
  if (tgt === 'zh-hant') return 'Traditional Chinese (繁體中文, Taiwan)';
  if (tgt === 'zh-hans') return 'Simplified Chinese (简体中文)';
  return englishName(tgt);
}

/**
 * The user turn for Qwen3 / Gemma 3: one short English instruction line, the text, then any
 * style / term notes in parentheses after it. Measured on the box (llama.cpp CPU, greedy): the
 * 0.6B / 1B models follow this shape; a longer rule list before the text made Gemma 3 1B return
 * nothing or ask for the text, and Qwen3 0.6B echo the source. Context is left out for these
 * models for the same reason (they started translating it).
 */
export function chatInstruction(input: CheckedInput): string {
  const target = chatTarget(input.tgt);
  const source = input.src ? `${englishName(input.src)} ` : '';
  const lineRule = input.text.includes('\n') ? ' Keep the line breaks.' : '';
  const notes: string[] = [];
  if (input.form === 'polite') notes.push('Style: polite です/ます');
  if (input.form === 'plain') notes.push('Style: plain だ/である');
  if (input.glossary.length > 0) {
    notes.push(
      `Terms: ${input.glossary.map((entry) => `${entry.source} = ${entry.target}`).join('; ')}`,
    );
  }
  return (
    `Translate the following ${source}text into ${target}. Output only the translation.${lineRule}` +
    `\n\n${input.text}` +
    (notes.length > 0 ? `\n\n(${notes.join('. ')})` : '')
  );
}

/**
 * Hy-MT2 model-card prompts (tencent/Hy-MT2-1.8B README, "Prompt templates"): the Chinese form
 * when either side is Chinese, the English form otherwise. Glossary terms use the Terminology
 * template, context the "Structured Data 2" background template, and a Japanese target's
 * polite/plain choice the Style template.
 */
export function hyMtInstruction(input: CheckedInput): string {
  const chinese = isChinese(input.tgt) || isChinese(input.src);
  const target = chinese ? chineseName(input.tgt) : englishName(input.tgt);
  if (input.context) {
    return chinese
      ? `【背景信息】\n${input.context}\n\n请结合背景信息将以下文本翻译为${target}。\n\n【待翻译文本】\n${input.text}`
      : `[Background Information]\n${input.context}\n\nPlease translate the following text into ${target}, taking the provided background information into consideration.\n\n[Source Text]\n${input.text}`;
  }
  if (input.glossary.length > 0) {
    const terms = input.glossary
      .map((entry) =>
        chinese
          ? `${entry.source} 翻译成 ${entry.target}`
          : `${entry.source} translates to ${entry.target}`,
      )
      .join('\n');
    return chinese
      ? `参考下面的翻译：\n${terms}\n将以下文本翻译为${target}，注意只需要输出翻译后的结果，不要额外解释：\n\n${input.text}`
      : `Reference the following translations:\n${terms}\n\nTranslate the following text into ${target}. Note that you must ONLY output the translated result without any additional explanation:\n\n${input.text}`;
  }
  if (input.form !== undefined) {
    const style = chinese
      ? input.form === 'polite'
        ? '敬体（です・ます调）'
        : '简体（だ・である调）'
      : input.form === 'polite'
        ? 'polite Japanese (desu/masu style)'
        : 'plain Japanese (da/dearu style)';
    return chinese
      ? `请将以下文本翻译为${target}。\n注意翻译的风格要严格符合【${style}】\n\n${input.text}`
      : `Please translate the following text into ${target}. Note that the translation style must strictly conform to [${style}]:\n\n${input.text}`;
  }
  return chinese
    ? `将以下文本翻译为${target}，注意只需要输出翻译后的结果，不要额外解释：\n\n${input.text}`
    : `Translate the following text into ${target}. Note that you should only output the translated result without any additional explanation:\n\n${input.text}`;
}

/** The prompt and stop strings for a model format. */
export function buildModelPrompt(format: PromptFormat, input: ModelPromptInput): ModelPrompt {
  if (format === 'plamo3') {
    const plamo: PromptInput = {
      text: input.text,
      tgt: input.tgt,
      ...(input.src !== undefined ? { src: input.src } : {}),
      ...(input.form !== undefined ? { form: input.form } : {}),
      ...(input.glossary !== undefined ? { glossary: input.glossary } : {}),
      ...(input.context !== undefined ? { context: input.context } : {}),
    };
    return { prompt: buildTranslationPrompt(plamo), stop: TRANSLATION_STOP_TOKENS };
  }
  const checked = checkInput(input);
  switch (format) {
    case 'qwen3':
      return {
        prompt:
          `<|im_start|>user\n${chatInstruction(checked)}<|im_end|>\n` +
          '<|im_start|>assistant\n<think>\n\n</think>\n\n',
        stop: QWEN3_STOP,
      };
    case 'gemma3':
      return {
        prompt:
          `<start_of_turn>user\n${chatInstruction(checked)}<end_of_turn>\n` +
          '<start_of_turn>model\n',
        stop: GEMMA3_STOP,
      };
    case 'hy-mt':
      return {
        prompt: `<｜hy_begin▁of▁sentence｜><｜hy_User｜>${hyMtInstruction(checked)}<｜hy_Assistant｜>`,
        stop: HY_MT_STOP,
      };
  }
}

const THINK_BLOCK = /<think>[\s\S]*?(?:<\/think>|$)/gu;
const LEADING_LABEL = /^(?:translation|translated text|譯文|译文|翻訳)\s*[:：]\s*/iu;

/**
 * Cleans a model's raw output: PFML markers for PLaMo; for the chat models also `<think>`
 * blocks, a stray stop/control token and a leading "Translation:" label.
 */
export function cleanOutputFor(format: PromptFormat, value: string): string {
  if (format === 'plamo3') return cleanModelOutput(value);
  let output = value.replace(THINK_BLOCK, '').replace(/<\/think>/gu, '');
  for (const marker of stopTokensFor(format)) {
    const index = output.indexOf(marker);
    if (index !== -1) output = output.slice(0, index);
  }
  output = stripSpecialTokens(output).trim();
  return output.replace(LEADING_LABEL, '').trim();
}
