// @vitest-environment happy-dom
import { afterEach, describe, expect, it } from 'vitest';
import { PageSession, type PageEngine, type SessionPreferences } from '@/dom/session';
import {
  pageRequestsInFlight,
  TRANSLATION_CONCURRENCY,
  WAVE_TRANSLATION_CONCURRENCY,
} from '@/dom/priority';
import type { TranslateOptions } from '@/engine/types';
import { deferred } from './support/ports';

// 1.3.0: the phone runs one slot on the GPU and the CPU alike, so the page keeps the same small
// number of requests in flight on both, and a block scrolling into view takes over a prefetched
// request that is still waiting (POCO F5 Pro harness: multi-slot was slower everywhere).

const labels = {
  original: '原文',
  chromeNote: 'Chrome',
  machine: 'PLaMo',
  selectionPending: '…',
  selectionFailed: 'x',
  selectionTitle: 't',
  selectionIcon: 'i',
  close: 'c',
  bilingualOn: 'b',
  replaceOn: 'r',
  barLabel: 'bar',
  barProgress: (done: number, total: number) => `${String(done)}/${String(total)}`,
  barRate: (tokens: string, segments: string) => `${tokens} ${segments}`,
  barDone: 'done',
  barPartial: (blocks: number) => String(blocks),
  barOriginal: 'o',
  barBilingual: 'b',
  barReplace: 'r',
  barAlways: 'a',
  barAlwaysOn: 'a',
  fabLabel: 'f',
};

function preferences(tier: SessionPreferences['tier']): SessionPreferences {
  return {
    targetLanguage: 'ja',
    targetExplicit: true,
    form: 'polite',
    display: 'replace',
    auto: 'ask',
    host: 'example.com',
    context: false,
    tier,
    pace: 'quality',
  };
}

interface Call {
  text: string;
  options: TranslateOptions;
}

/** Holds every request until `gate` opens; nothing streams before that. */
function heldEngine(calls: Call[], gate: Promise<void>, active: { now: number; max: number }) {
  const engine: PageEngine = {
    async *translate(segments, options) {
      calls.push({ text: segments.map((s) => s.text).join('\n'), options });
      active.now += 1;
      active.max = Math.max(active.max, active.now);
      try {
        await Promise.race([
          gate,
          new Promise<void>((resolve) =>
            options.signal.addEventListener('abort', () => resolve(), { once: true }),
          ),
        ]);
        if (options.signal.aborted) return;
        for (const segment of segments) {
          yield { id: segment.id, final: `訳:${segment.text}`, engine: 'plamo' as const };
        }
      } finally {
        active.now -= 1;
      }
    },
  };
  return engine;
}

const sessions: PageSession[] = [];
const originalRect = Element.prototype.getBoundingClientRect;
const originalObserver = (globalThis as { IntersectionObserver?: unknown }).IntersectionObserver;

afterEach(() => {
  for (const session of sessions) session.dispose();
  sessions.length = 0;
  document.body.innerHTML = '';
  Element.prototype.getBoundingClientRect = originalRect;
  (window as unknown as { IntersectionObserver?: unknown }).IntersectionObserver = originalObserver;
});

function headings(count: number): void {
  document.documentElement.lang = 'en';
  document.body.innerHTML = Array.from(
    { length: count },
    (_, i) => `<h2 id="h${String(i)}">Heading number ${String(i)} of the page</h2>`,
  ).join('');
}

async function settle(): Promise<void> {
  for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setTimeout(resolve, 5));
}

describe('page requests in flight', () => {
  it('sizes the limit by engine slots: 3 for one slot, 32 for a multi-slot GPU', () => {
    expect(pageRequestsInFlight(1)).toBe(TRANSLATION_CONCURRENCY);
    expect(pageRequestsInFlight(null)).toBe(3);
    expect(pageRequestsInFlight(2)).toBe(WAVE_TRANSLATION_CONCURRENCY);
    expect(pageRequestsInFlight(16)).toBe(32);
  });

  it('keeps 3 requests in flight on a one-slot GPU and on the CPU, 32 on a 4-slot GPU', async () => {
    for (const [tier, slots, expected] of [
      ['g', 1, 3],
      ['c8', 1, 3],
      ['g', 4, 12],
    ] as const) {
      headings(12);
      const calls: Call[] = [];
      const gate = deferred<void>();
      const active = { now: 0, max: 0 };
      const session = new PageSession(document, {
        engine: heldEngine(calls, gate.promise, active),
        labels,
        preferences: preferences(tier),
        scheduleFrame: (callback) => callback(),
        debounceMs: 5,
      });
      sessions.push(session);
      session.setEngineSlots(slots);
      const translating = session.translate();
      await settle();
      expect(active.max).toBe(expected);
      gate.resolve();
      await translating;
      await session.whenIdle();
      expect(calls).toHaveLength(12);
      session.dispose();
    }
  });

  it('hands a waiting prefetched request back for a block that scrolls into view', async () => {
    headings(8);
    // Every heading starts far below the screen.
    const visible = new Set<Element>();
    Element.prototype.getBoundingClientRect = function (this: Element) {
      const on = visible.has(this);
      const top = on ? 10 : 5000;
      return {
        top,
        bottom: top + 30,
        left: 0,
        right: 100,
        width: 100,
        height: 30,
        x: 0,
        y: top,
        toJSON: () => ({}),
      } as DOMRect;
    };
    const observers: Array<(entries: IntersectionObserverEntry[]) => void> = [];
    (window as unknown as { IntersectionObserver: unknown }).IntersectionObserver = class {
      constructor(callback: (entries: IntersectionObserverEntry[]) => void) {
        observers.push(callback);
      }
      observe(): void {}
      unobserve(): void {}
      disconnect(): void {}
    };
    const calls: Call[] = [];
    const gate = deferred<void>();
    const active = { now: 0, max: 0 };
    const session = new PageSession(document, {
      engine: heldEngine(calls, gate.promise, active),
      labels,
      preferences: preferences('g'),
      scheduleFrame: (callback) => callback(),
      debounceMs: 5,
    });
    sessions.push(session);
    session.setEngineSlots(1);
    const translating = session.translate();
    await settle();
    expect(calls).toHaveLength(3);
    expect(calls.every((call) => !call.text.includes('number 7'))).toBe(true);

    // The last heading scrolls into view while three prefetched requests wait.
    const target = document.querySelector('#h7')!;
    visible.add(target);
    for (const callback of observers) {
      callback([
        {
          target,
          boundingClientRect: target.getBoundingClientRect(),
          rootBounds: { height: 800 } as DOMRectReadOnly,
          isIntersecting: true,
        } as unknown as IntersectionObserverEntry,
      ]);
    }
    await settle();
    const visibleCall = calls.find((call) => call.text.includes('number 7'));
    expect(visibleCall).toBeDefined();
    // One prefetched request was aborted to make room, and the limit still holds.
    expect(calls.slice(0, 3).filter((call) => call.options.signal.aborted)).toHaveLength(1);
    // Nothing has finished yet: the requests not handed back are the 3 the page keeps in flight.
    expect(calls.filter((call) => !call.options.signal.aborted)).toHaveLength(3);
    expect(visibleCall!.options.priority).toBeLessThan(calls[0]!.options.priority);

    gate.resolve();
    await translating;
    await session.whenIdle();
    // Every heading is translated in the end, the handed-back one too.
    expect(document.body.textContent?.match(/訳:/gu)).toHaveLength(8);
  });
});
