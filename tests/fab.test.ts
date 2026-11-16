// @vitest-environment happy-dom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { FloatingActionButton, FAB_DRAG_PX, FAB_LONG_PRESS_MS, FAB_PX } from '@/dom/fab';
import { uiShadow } from '@/dom/overlay';

function pointer(type: string, init: PointerEventInit): PointerEvent {
  return new window.PointerEvent(type, {
    bubbles: true,
    cancelable: true,
    button: 0,
    pointerId: 7,
    ...init,
  });
}

const fabs: FloatingActionButton[] = [];

function setup(): {
  fab: FloatingActionButton;
  host: HTMLElement;
  button: HTMLButtonElement;
  taps: number[];
  longs: number[];
} {
  const taps: number[] = [];
  const longs: number[] = [];
  const fab = new FloatingActionButton(document, 'Translate this page', {
    onTap: () => taps.push(1),
    onLongPress: () => longs.push(1),
  });
  fabs.push(fab);
  const host = document.querySelector('sakutto-fab');
  if (!(host instanceof HTMLElement)) throw new Error('missing fab host');
  const button = uiShadow(host)?.querySelector('button');
  if (!(button instanceof HTMLButtonElement)) throw new Error('missing fab button');
  return { fab, host, button, taps, longs };
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  for (const fab of fabs) fab.destroy();
  fabs.length = 0;
  vi.useRealTimers();
  document.body.innerHTML = '';
});

describe('FloatingActionButton', () => {
  it('renders a 48px button with the label, visible by default', () => {
    const { fab, host, button } = setup();
    expect(host.style.width).toBe(`${String(FAB_PX)}px`);
    expect(host.style.height).toBe(`${String(FAB_PX)}px`);
    expect(button.getAttribute('aria-label')).toBe('Translate this page');
    expect(fab.visible()).toBe(true);
  });

  it('taps on a quick press', () => {
    const { button, taps, longs } = setup();
    button.dispatchEvent(pointer('pointerdown', { clientX: 100, clientY: 100 }));
    button.dispatchEvent(pointer('pointerup', { clientX: 100, clientY: 100 }));
    button.dispatchEvent(pointer('click', { clientX: 100, clientY: 100 }));
    expect(taps).toHaveLength(1);
    expect(longs).toHaveLength(0);
  });

  it('long-presses after 500ms and does not also tap', () => {
    const { button, taps, longs } = setup();
    button.dispatchEvent(pointer('pointerdown', { clientX: 100, clientY: 100 }));
    vi.advanceTimersByTime(FAB_LONG_PRESS_MS);
    expect(longs).toHaveLength(1);
    button.dispatchEvent(pointer('pointerup', { clientX: 100, clientY: 100 }));
    button.dispatchEvent(pointer('click', { clientX: 100, clientY: 100 }));
    expect(taps).toHaveLength(0);
    expect(longs).toHaveLength(1);
  });

  it('still taps when released just before the long-press fires', () => {
    const { button, taps, longs } = setup();
    button.dispatchEvent(pointer('pointerdown', { clientX: 100, clientY: 100 }));
    vi.advanceTimersByTime(FAB_LONG_PRESS_MS - 1);
    button.dispatchEvent(pointer('pointerup', { clientX: 100, clientY: 100 }));
    button.dispatchEvent(pointer('click', { clientX: 100, clientY: 100 }));
    expect(taps).toHaveLength(1);
    expect(longs).toHaveLength(0);
  });

  it('drags out of the way instead of tapping', () => {
    const { host, button, taps, longs } = setup();
    button.dispatchEvent(pointer('pointerdown', { clientX: 200, clientY: 200 }));
    button.dispatchEvent(
      pointer('pointermove', { clientX: 200, clientY: 200 + FAB_DRAG_PX + 20 }),
    );
    button.dispatchEvent(pointer('pointerup', { clientX: 200, clientY: 200 + FAB_DRAG_PX + 20 }));
    button.dispatchEvent(pointer('click', { clientX: 200, clientY: 200 + FAB_DRAG_PX + 20 }));
    expect(taps).toHaveLength(0);
    expect(longs).toHaveLength(0);
    expect(host.style.left).not.toBe('');
    expect(host.style.top).not.toBe('');
  });

  it('cancels the long-press when the pointer moves, then drags', () => {
    const { button, taps, longs } = setup();
    button.dispatchEvent(pointer('pointerdown', { clientX: 200, clientY: 200 }));
    button.dispatchEvent(
      pointer('pointermove', { clientX: 200, clientY: 200 + FAB_DRAG_PX + 1 }),
    );
    vi.advanceTimersByTime(FAB_LONG_PRESS_MS);
    expect(longs).toHaveLength(0);
    button.dispatchEvent(pointer('pointerup', { clientX: 200, clientY: 200 + FAB_DRAG_PX + 1 }));
    button.dispatchEvent(pointer('click', { clientX: 200, clientY: 200 + FAB_DRAG_PX + 1 }));
    expect(taps).toHaveLength(0);
  });

  it('still taps after a small move inside the drag threshold', () => {
    const { button, taps, longs } = setup();
    button.dispatchEvent(pointer('pointerdown', { clientX: 100, clientY: 100 }));
    button.dispatchEvent(pointer('pointermove', { clientX: 100 + FAB_DRAG_PX - 1, clientY: 100 }));
    button.dispatchEvent(pointer('pointerup', { clientX: 100 + FAB_DRAG_PX - 1, clientY: 100 }));
    button.dispatchEvent(pointer('click', { clientX: 100 + FAB_DRAG_PX - 1, clientY: 100 }));
    expect(taps).toHaveLength(1);
    expect(longs).toHaveLength(0);
  });

  it('hides, shows, and renames', () => {
    const { fab, button } = setup();
    fab.hide();
    expect(fab.visible()).toBe(false);
    fab.show();
    expect(fab.visible()).toBe(true);
    fab.setLabel('Translate');
    expect(button.getAttribute('aria-label')).toBe('Translate');
  });

  it('removes the host on destroy', () => {
    const { fab } = setup();
    fab.destroy();
    expect(document.querySelector('sakutto-fab')).toBeNull();
  });
});
