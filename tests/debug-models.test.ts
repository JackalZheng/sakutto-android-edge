import { describe, expect, it, vi } from 'vitest';
import {
  buildHarnessPrompt,
  describeHarnessModel,
  formatModelSize,
  HARNESS_DEFAULT_MODEL_ID,
  HARNESS_MODELS,
  type HarnessModel,
  harnessModel,
  HarnessModelStore,
  harnessModelPaths,
  isHarnessModelId,
} from '@/engine/debug-models';
import { DEFAULT_MODEL_ID, MODEL_CATALOG } from '@/engine/model/catalog';
import { downloadModel } from '@/engine/model/download';
import { TRANSLATION_STOP_TOKENS } from '@/engine/prompt/pfml3';
import { bytesOf, MemoryFileStore } from './support/fakes';

const INPUT = { text: 'The train left.', src: 'en', tgt: 'zh-hant' };

describe('harness model list', () => {
  it('lists the five models smallest file first', () => {
    expect(HARNESS_MODELS.map((m) => m.id)).toEqual([
      'qwen3-0.6b-q4_k_m',
      'plamo-3-translate-0.8b-q4_k_m',
      'gemma-3-1b-it-q4_k_m',
      'qwen3-1.7b-q4_k_m',
      'hy-mt2-1.8b-q4_k_m',
    ]);
    const sizes = HARNESS_MODELS.map((m) => m.size);
    expect([...sizes].sort((a, b) => a - b)).toEqual(sizes);
  });

  it('keeps the production model as the default and reuses its catalog pin', () => {
    expect(HARNESS_DEFAULT_MODEL_ID).toBe(DEFAULT_MODEL_ID);
    const plamo = harnessModel(DEFAULT_MODEL_ID);
    const catalog = MODEL_CATALOG[DEFAULT_MODEL_ID];
    expect(plamo.production).toBe(true);
    expect([plamo.url, plamo.sha256, plamo.size]).toEqual([
      catalog.url,
      catalog.sha256,
      catalog.size,
    ]);
    expect(HARNESS_MODELS.filter((m) => m.production)).toHaveLength(1);
  });

  it('pins every model to a commit, a sha256 and a Hugging Face resolve url', () => {
    for (const model of HARNESS_MODELS) {
      expect(model.commit).toMatch(/^[0-9a-f]{40}$/u);
      expect(model.sha256).toMatch(/^[0-9a-f]{64}$/u);
      expect(model.url).toBe(
        `https://huggingface.co/${model.repository}/resolve/${model.commit}/${model.file}`,
      );
      expect(model.license.length).toBeGreaterThan(0);
      expect(model.description.length).toBeGreaterThan(0);
      expect(new Set(HARNESS_MODELS.map((m) => m.id)).size).toBe(HARNESS_MODELS.length);
    }
  });

  it('only uses architectures the pinned llama.cpp knows', () => {
    expect(new Set(HARNESS_MODELS.map((m) => m.architecture))).toEqual(
      new Set(['qwen3', 'plamo3', 'gemma3', 'hunyuan-dense']),
    );
  });

  it('describes a model with name, size and license', () => {
    expect(formatModelSize(396_705_472)).toBe('397 MB');
    expect(formatModelSize(1_133_080_448)).toBe('1.13 GB');
    const line = describeHarnessModel(harnessModel('qwen3-0.6b-q4_k_m'));
    expect(line).toContain('Qwen3 0.6B（GPU 實驗性） · 397 MB · Apache-2.0 — ');
  });

  it('recognizes ids and rejects unknown ones', () => {
    expect(isHarnessModelId('gemma-3-1b-it-q4_k_m')).toBe(true);
    expect(isHarnessModelId('nope')).toBe(false);
    expect(isHarnessModelId(undefined)).toBe(false);
    expect(() => harnessModel('nope')).toThrow(/Unknown harness model/u);
  });
});

describe('buildHarnessPrompt', () => {
  it('uses the product PFML3 prompt and stops for PLaMo 3', () => {
    const { prompt, stop } = buildHarnessPrompt(harnessModel(DEFAULT_MODEL_ID), INPUT);
    expect(prompt).toContain('The train left.');
    expect(stop).toEqual(TRANSLATION_STOP_TOKENS);
  });

  it('turns Qwen3 thinking off with an empty think block', () => {
    const { prompt, stop } = buildHarnessPrompt(harnessModel('qwen3-0.6b-q4_k_m'), INPUT);
    expect(prompt.startsWith('<|im_start|>user\n')).toBe(true);
    expect(prompt.endsWith('<|im_start|>assistant\n<think>\n\n</think>\n\n')).toBe(true);
    expect(prompt).toContain('The train left.<|im_end|>');
    expect(stop).toContain('<|im_end|>');
  });

  it('builds Gemma 3 turns without a literal <bos>', () => {
    const { prompt, stop } = buildHarnessPrompt(harnessModel('gemma-3-1b-it-q4_k_m'), INPUT);
    expect(prompt.startsWith('<start_of_turn>user\n')).toBe(true);
    expect(prompt).not.toContain('<bos>');
    expect(prompt.endsWith('<end_of_turn>\n<start_of_turn>model\n')).toBe(true);
    expect(stop).toContain('<end_of_turn>');
  });

  it('follows the Hy-MT2 model card template and prompt', () => {
    const { prompt, stop } = buildHarnessPrompt(harnessModel('hy-mt2-1.8b-q4_k_m'), INPUT);
    expect(prompt).toBe(
      '<｜hy_begin▁of▁sentence｜><｜hy_User｜>' +
        '将以下文本翻译为繁体中文，注意只需要输出翻译后的结果，不要额外解释：\n\nThe train left.' +
        '<｜hy_Assistant｜>',
    );
    expect(stop).toContain('<｜hy_place▁holder▁no▁2｜>');
  });
});

describe('HarnessModelStore', () => {
  const payload = bytesOf('gguf!');
  const model: HarnessModel = {
    ...harnessModel('qwen3-0.6b-q4_k_m'),
    size: payload.length,
  };

  it('stores each model under its own directory and reports it downloaded', async () => {
    const files = new MemoryFileStore();
    const download = vi.fn<typeof downloadModel>(async (request) => {
      await files.append(request.finalPath, payload, 0);
      return { bytes: payload.length, sha256: request.source.sha256, resumedFrom: 0 };
    });
    const store = new HarnessModelStore(files, download);
    expect(await store.isDownloaded(model)).toBe(false);
    await store.downloadFile(model, new AbortController().signal);
    const request = download.mock.calls[0]?.[0];
    expect(request?.source).toEqual({ url: model.url, size: model.size, sha256: model.sha256 });
    expect(request?.finalPath).toBe(harnessModelPaths(model.id).final);
    expect(request?.partPath).toBe(harnessModelPaths(model.id).part);
    expect(await store.isDownloaded(model)).toBe(true);
    expect((await store.read(model)).size).toBe(payload.length);
    await store.remove(model);
    expect(await store.isDownloaded(model)).toBe(false);
  });

  it('reports a partial download', async () => {
    const files = new MemoryFileStore();
    await files.append(harnessModelPaths(model.id).part, bytesOf('gg'), 0);
    const store = new HarnessModelStore(files);
    expect(await store.partialBytes(model)).toBe(2);
    expect(await store.isDownloaded(model)).toBe(false);
  });

  it('refuses to download, read or delete the production model', async () => {
    const store = new HarnessModelStore(new MemoryFileStore());
    const plamo = harnessModel(DEFAULT_MODEL_ID);
    await expect(store.remove(plamo)).rejects.toThrow(/options page/u);
    await expect(store.read(plamo)).rejects.toThrow(/options page/u);
    await expect(store.downloadFile(plamo, new AbortController().signal)).rejects.toThrow(
      /options page/u,
    );
  });
});
