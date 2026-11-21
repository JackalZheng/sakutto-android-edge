import { describe, expect, it, vi } from 'vitest';
import {
  cpuAfterBrokenOutput,
  resolveSelection,
  selectBackend,
  type BackendSelection,
} from '@/engine/in-browser/backend';
import { LazyInBrowserEngine } from '@/engine/in-browser/lazy-engine';
import { looksLikeBrokenTranslation } from '@/engine/text/garbage';
import type { TranslateOptions } from '@/engine/types';
import { collectChunks, FakeRuntime } from './support/fakes';

const PARAGRAPH =
  'The morning train leaves the quiet station while the baker opens a small shop on the corner of the square where children walk to school.';
const JA_PARAGRAPH =
  '京都は千年以上にわたって日本の都として栄え、多くの寺社や庭園が今も残っている。春には桜、秋には紅葉を求めて多くの観光客が訪れる。';

describe('looksLikeBrokenTranslation', () => {
  it('flags the Adreno WebGPU garbage from the user report', () => {
    expect(looksLikeBrokenTranslation(PARAGRAPH, 'ら', 'zh-hant')).toBe(true);
    expect(looksLikeBrokenTranslation(PARAGRAPH, '일까지', 'zh-hant')).toBe(true);
    expect(looksLikeBrokenTranslation(PARAGRAPH, '鄉この', 'zh-hant')).toBe(true);
    expect(looksLikeBrokenTranslation(JA_PARAGRAPH, '難', 'zh-hant')).toBe(true);
    expect(looksLikeBrokenTranslation(PARAGRAPH, '', 'ja')).toBe(true);
  });

  it('flags a paragraph in the wrong script for a CJK target', () => {
    expect(
      looksLikeBrokenTranslation('Please take a numbered ticket.', '번호표를 받아 주세요.', 'ja'),
    ).toBe(true);
    expect(
      looksLikeBrokenTranslation(
        'Please take a numbered ticket.',
        '番号札をお取りください。',
        'ko',
      ),
    ).toBe(true);
    expect(
      looksLikeBrokenTranslation(
        'Please take a numbered ticket.',
        'ばんごうふだをどうぞ',
        'zh-hant',
      ),
    ).toBe(true);
  });

  it('accepts real translations, including short and Latin-script ones', () => {
    expect(
      looksLikeBrokenTranslation(
        PARAGRAPH,
        '清晨的火車駛離安靜的車站，麵包師傅在廣場轉角開了一家小店，孩子們走路去上學。',
        'zh-hant',
      ),
    ).toBe(false);
    expect(
      looksLikeBrokenTranslation(
        JA_PARAGRAPH,
        'Kyoto flourished as the capital of Japan for over a thousand years, and many temples remain.',
        'en',
      ),
    ).toBe(false);
    expect(
      looksLikeBrokenTranslation(
        PARAGRAPH,
        '朝の列車が静かな駅を出発し、パン屋は広場の角で小さな店を開きます。',
        'ja',
      ),
    ).toBe(false);
    expect(looksLikeBrokenTranslation('Sign in', '登入', 'zh-hant')).toBe(false);
    expect(looksLikeBrokenTranslation('Sign in', 'ら', 'zh-hant')).toBe(false);
    expect(looksLikeBrokenTranslation('ソニーの新製品', 'Sonyの新製品', 'en')).toBe(false);
    expect(looksLikeBrokenTranslation(PARAGRAPH, PARAGRAPH, 'ja')).toBe(false);
  });
});

describe('resolveSelection on Edge for Android', () => {
  const probe = { webgpu: true, hardwareConcurrency: 8, sharedMemory: true };

  it('resolves auto to the CPU even when WebGPU is ready', () => {
    const selection = resolveSelection('auto', probe, { android: true });
    expect(selection.backend).toBe('cpu-wasm');
    expect(selection.nGpuLayers).toBe(0);
    expect(selection.nThreads).toBe(4);
    expect(selection.nBatch).toBe(256);
  });

  it('keeps WebGPU for an explicit choice', () => {
    expect(resolveSelection('g', probe, { android: true }).backend).toBe('webgpu');
  });

  it('keeps an explicit WebGPU choice on the CPU after broken output', () => {
    const selection = resolveSelection('g', probe, { android: true, gpuOutputBroken: true });
    expect(selection.backend).toBe('cpu-wasm');
    expect(selection.nThreads).toBe(4);
  });

  it('leaves desktop auto on WebGPU', () => {
    expect(resolveSelection('auto', probe, { android: false }).backend).toBe('webgpu');
  });

  it('moves a broken WebGPU selection to the capped CPU', () => {
    const gpu = resolveSelection('g', probe, { android: true });
    const cpu = cpuAfterBrokenOutput(gpu, { android: true });
    expect(cpu.backend).toBe('cpu-wasm');
    expect(cpu.nThreads).toBe(4);
    expect(cpu.nBatch).toBe(256);
  });
});

describe('LazyInBrowserEngine output guard', () => {
  const GPU: BackendSelection = selectBackend({ webgpu: true, hardwareConcurrency: 8 });
  const CPU: BackendSelection = selectBackend({ webgpu: false, hardwareConcurrency: 8 });
  const options: TranslateOptions = {
    tgt: 'zh-hant',
    form: 'polite',
    priority: 10,
    signal: new AbortController().signal,
  };
  const good = '清晨的火車駛離安靜的車站，麵包師傅在廣場轉角開了一家小店。';

  it('re-translates a broken WebGPU block on the CPU', async () => {
    const gpuRuntime = new FakeRuntime(() => 'ら');
    const cpuRuntime = new FakeRuntime(() => good);
    let broken = false;
    const loadRuntime = vi.fn(async () =>
      broken ? { runtime: cpuRuntime, selection: CPU } : { runtime: gpuRuntime, selection: GPU },
    );
    const onBroken = vi.fn(async () => {
      broken = true;
    });
    const engine = new LazyInBrowserEngine({
      loadRuntime,
      outputGuard: { isBroken: looksLikeBrokenTranslation, onBroken },
    });

    const chunks = await collectChunks(engine.translate([{ id: 'p', text: PARAGRAPH }], options));

    expect(onBroken).toHaveBeenCalledOnce();
    expect(loadRuntime).toHaveBeenCalledTimes(2);
    expect(chunks.filter((chunk) => chunk.final !== undefined)).toEqual([
      { id: 'p', final: good, engine: 'plamo' },
    ]);
    expect(chunks.find((chunk) => chunk.restart === true)).toBeDefined();
  });

  it('never checks CPU output', async () => {
    const isBroken = vi.fn(() => true);
    const onBroken = vi.fn(async () => undefined);
    const engine = new LazyInBrowserEngine({
      loadRuntime: async () => ({ runtime: new FakeRuntime(() => 'ら'), selection: CPU }),
      outputGuard: { isBroken, onBroken },
    });

    const chunks = await collectChunks(engine.translate([{ id: 'p', text: PARAGRAPH }], options));

    expect(isBroken).not.toHaveBeenCalled();
    expect(onBroken).not.toHaveBeenCalled();
    expect(chunks.at(-1)).toEqual({ id: 'p', final: 'ら', engine: 'plamo' });
  });
});
