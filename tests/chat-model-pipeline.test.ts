import { describe, expect, it } from 'vitest';
import {
  InBrowserEngine,
  keepsInlineTags,
  packsLines,
  stripInlineTags,
} from '@/engine/in-browser/engine';
import type { TranslateOptions, TranslationChunk } from '@/engine/types';
import { collectChunks, FakeRuntime } from './support/fakes';

// Regression (1.3.0, POCO F5 Pro): "用 qwen3 0.6b，會翻譯三次". A short block went through a
// newline-packed request (Qwen3 0.6B lost the line count, even answered in Japanese), then a
// line-by-line retry with the <i1> link tags (which Qwen3 drops), then the page's flattened
// retry: the reader saw the same block translated three times.

function options(overrides: Partial<TranslateOptions> = {}): TranslateOptions {
  return {
    tgt: 'zh-hant',
    src: 'en',
    form: 'polite',
    priority: 10,
    signal: new AbortController().signal,
    ...overrides,
  };
}

function finals(chunks: readonly TranslationChunk[]): Record<string, string> {
  return Object.fromEntries(
    chunks.filter((chunk) => chunk.final !== undefined).map((chunk) => [chunk.id, chunk.final]),
  );
}

const LABELS = [
  { id: 'a', text: 'Home' },
  { id: 'b', text: 'News' },
  { id: 'c', text: 'Contact us' },
];

describe('chat-model translation pipeline', () => {
  it('only PLaMo packs lines and keeps inline tags', () => {
    expect(packsLines('plamo3')).toBe(true);
    expect(keepsInlineTags('plamo3')).toBe(true);
    for (const format of ['qwen3', 'gemma3', 'hy-mt'] as const) {
      expect(packsLines(format)).toBe(false);
      expect(keepsInlineTags(format)).toBe(false);
    }
    expect(stripInlineTags('Read the <i1>release notes</i1>  first.')).toBe(
      'Read the release notes first.',
    );
  });

  it('sends one request per short block for Qwen3, never a packed one', async () => {
    const runtime = new FakeRuntime(() => '譯文');
    const engine = new InBrowserEngine(runtime, { format: 'qwen3' });

    const chunks = await collectChunks(engine.translate(LABELS, options()));

    expect(runtime.requests).toHaveLength(3);
    for (const request of runtime.requests) {
      expect(request.prompt.startsWith('<|im_start|>user')).toBe(true);
      expect(request.prompt).not.toContain('Keep the line breaks');
    }
    expect(finals(chunks)).toEqual({ a: '譯文', b: '譯文', c: '譯文' });
  });

  it('still packs short blocks for PLaMo', async () => {
    const runtime = new FakeRuntime(() => '一\n二\n三');
    const engine = new InBrowserEngine(runtime);

    const chunks = await collectChunks(engine.translate(LABELS, options({ tgt: 'ja' })));

    expect(runtime.requests).toHaveLength(1);
    expect(finals(chunks)).toEqual({ a: '一', b: '二', c: '三' });
  });

  it('gives a chat model the text without the inline tags or their glossary locks', async () => {
    const runtime = new FakeRuntime(() => '更新前請先閱讀版本說明。');
    const engine = new InBrowserEngine(runtime, { format: 'gemma3' });

    await collectChunks(
      engine.translate(
        [{ id: 'a', text: 'Read the <i1>release notes</i1> before you update.' }],
        options({
          glossary: [
            { source: '<i1>', target: '<i1>' },
            { source: '</i1>', target: '</i1>' },
            { source: 'release notes', target: '版本說明' },
          ],
        }),
      ),
    );

    expect(runtime.requests).toHaveLength(1);
    const prompt = runtime.requests[0]?.prompt ?? '';
    expect(prompt).toContain('Read the release notes before you update.');
    expect(prompt).not.toMatch(/<\/?i\d/);
    expect(prompt).toContain('release notes = 版本說明');
  });

  it('keeps the inline tags for PLaMo', async () => {
    const runtime = new FakeRuntime(() => '<i1>版本說明</i1>');
    const engine = new InBrowserEngine(runtime);

    await collectChunks(
      engine.translate([{ id: 'a', text: 'Read the <i1>release notes</i1>.' }], options()),
    );

    expect(runtime.requests[0]?.prompt).toContain('<i1>release notes</i1>');
  });
});
