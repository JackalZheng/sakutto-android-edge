import { estimateTokens } from './tokens';

/**
 * A conservative check for translations that cannot be right.
 *
 * Built for one failure: llama.cpp's WebGPU backend on some mobile GPUs (reported on an
 * Adreno 730, Edge for Android) computes wrong logits, so every paragraph comes back as one
 * to three random tokens from mixed scripts ("ら", "일까지", "鄉この"). A real translation is
 * never a sixth of its source and never mostly in a script the target does not use, so the
 * check only fires on those shapes. It is not a quality metric.
 */

const KANA = /[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]/u;
const HANGUL = /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/u;
const LETTER = /\p{L}/u;

/** Source size (estimated tokens) at which a near-empty output is treated as broken. */
export const BROKEN_MIN_SOURCE_TOKENS = 16;
/** A translation shorter than this fraction of its source, in estimated tokens, is broken. */
export const BROKEN_LENGTH_RATIO = 1 / 6;
/** Below this source size the script check does not run: a label can be anything. */
const SCRIPT_MIN_SOURCE_TOKENS = 4;

interface ScriptCounts {
  letters: number;
  kana: number;
  hangul: number;
}

function countScripts(text: string): ScriptCounts {
  const counts: ScriptCounts = { letters: 0, kana: 0, hangul: 0 };
  for (const character of text) {
    if (KANA.test(character)) {
      counts.kana += 1;
      counts.letters += 1;
    } else if (HANGUL.test(character)) {
      counts.hangul += 1;
      counts.letters += 1;
    } else if (LETTER.test(character)) {
      counts.letters += 1;
    }
  }
  return counts;
}

function wrongScriptShare(target: string, output: ScriptCounts, source: ScriptCounts): number {
  if (output.letters === 0) return 0;
  const language = target.toLowerCase();
  if (language === 'ja') return output.hangul / output.letters;
  if (language === 'ko') return output.kana / output.letters;
  if (language.startsWith('zh')) return (output.kana + output.hangul) / output.letters;
  // Any other target: kana or hangul the source did not contain cannot belong there.
  if (source.kana + source.hangul > 0) return 0;
  return (output.kana + output.hangul) / output.letters;
}

function wrongScriptLimit(target: string): number {
  return target.toLowerCase().startsWith('zh') ? 0.3 : 0.5;
}

export function looksLikeBrokenTranslation(
  source: string,
  output: string,
  target: string,
): boolean {
  const input = source.trim();
  const result = output.trim();
  if (input.length === 0 || result === input) return false;

  const sourceTokens = estimateTokens(input);
  if (sourceTokens >= BROKEN_MIN_SOURCE_TOKENS) {
    if (result.length === 0) return true;
    if (estimateTokens(result) < sourceTokens * BROKEN_LENGTH_RATIO) return true;
  }

  if (sourceTokens < SCRIPT_MIN_SOURCE_TOKENS || result.length === 0) return false;
  const outputScripts = countScripts(result);
  const sourceScripts = countScripts(input);
  return wrongScriptShare(target, outputScripts, sourceScripts) >= wrongScriptLimit(target);
}
