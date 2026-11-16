import { appendSelectionIcon, attachUiShadow, uiShadow } from './overlay';

/**
 * Floating action button for touch runtimes.
 *
 * Edge for Android has no keyboard, so the desktop shortcuts (Alt+A/B/S/T) do not exist
 * there. The button is the touch replacement for them: a tap toggles the page
 * translation, a long-press translates the current selection (or opens the writing
 * translator when a field is focused and nothing is selected), and it can be dragged
 * out of the way. It deliberately has no double-tap (the browser reserves it for
 * zoom), no edge swipe (back/forward) and no two-finger tap (undiscoverable).
 */

/** Diameter of the button. Matches the port's 44px touch-target standard, plus a border. */
export const FAB_PX = 48;

/** How long the pointer must stay down before the gesture becomes a long-press. */
export const FAB_LONG_PRESS_MS = 500;

/** Pointer travel that cancels a long-press and starts a drag instead. */
export const FAB_DRAG_PX = 10;

const FAB_STYLE = `
:host { all: initial; }
button {
  box-sizing: border-box;
  display: block;
  width: ${String(FAB_PX)}px;
  height: ${String(FAB_PX)}px;
  padding: 0;
  border: 1px solid rgb(255 255 255 / 88%);
  border-radius: 50%;
  background: #101828;
  box-shadow: 0 4px 16px rgb(15 23 32 / 32%);
  overflow: hidden;
  cursor: pointer;
  user-select: none;
  -webkit-user-select: none;
  -webkit-touch-callout: none;
  touch-action: none;
}
button:hover,
button:focus-visible {
  border-color: #67e8f9;
}
button:focus-visible { outline: 2px solid #0e7490; outline-offset: 2px; }
svg { display: block; width: 100%; height: 100%; }
@media (prefers-reduced-motion: reduce) {
  button { transition: none; }
}
`;

export interface FabActions {
  /** A tap, or a keyboard activation of the focused button. */
  onTap: () => void;
  /** A press held still for at least FAB_LONG_PRESS_MS. */
  onLongPress: () => void;
}

export class FloatingActionButton {
  private readonly host: HTMLElement;
  private readonly button: HTMLButtonElement;
  private readonly actions: FabActions;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private pointerId: number | null = null;
  private startX = 0;
  private startY = 0;
  private dragging = false;
  private longPressed = false;
  private suppressClick = false;
  private readonly onResize: () => void;

  constructor(
    private readonly doc: Document,
    label: string,
    actions: FabActions,
  ) {
    this.actions = actions;
    this.host = doc.createElement('sakutto-fab');
    this.host.setAttribute('translate', 'no');
    this.host.style.position = 'fixed';
    this.host.style.zIndex = '2147483646';
    this.host.style.right = '16px';
    this.host.style.bottom = '16px';
    this.host.style.width = `${String(FAB_PX)}px`;
    this.host.style.height = `${String(FAB_PX)}px`;
    this.host.hidden = false;
    this.host.style.display = 'block';
    const shadow = attachUiShadow(this.host);
    const style = doc.createElement('style');
    style.textContent = FAB_STYLE;
    this.button = doc.createElement('button');
    this.button.type = 'button';
    this.button.setAttribute('aria-label', label);
    this.button.title = label;
    appendSelectionIcon(doc, this.button);
    shadow.append(style, this.button);
    doc.documentElement.append(this.host);

    this.button.addEventListener('pointerdown', (event) => this.onPointerDown(event));
    this.button.addEventListener('pointermove', (event) => this.onPointerMove(event));
    this.button.addEventListener('pointerup', (event) => this.onPointerUp(event));
    this.button.addEventListener('pointercancel', () => this.cancelGesture());
    // A long-press also opens the browser's context menu; the button has its own menu.
    this.button.addEventListener('contextmenu', (event) => event.preventDefault());
    this.button.addEventListener('click', (event) => {
      if (!this.suppressClick) {
        this.actions.onTap();
        return;
      }
      this.suppressClick = false;
      event.preventDefault();
      event.stopPropagation();
    });
    this.onResize = () => this.clampToViewport();
    doc.defaultView?.addEventListener('resize', this.onResize);
  }

  setLabel(label: string): void {
    this.button.setAttribute('aria-label', label);
    this.button.title = label;
  }

  show(): void {
    this.host.hidden = false;
    this.host.style.display = 'block';
  }

  hide(): void {
    this.host.hidden = true;
    this.host.style.display = 'none';
  }

  visible(): boolean {
    return !this.host.hidden && this.host.style.display !== 'none';
  }

  contains(node: Node): boolean {
    return (
      this.host === node || this.host.contains(node) || uiShadow(this.host)?.contains(node) === true
    );
  }

  destroy(): void {
    this.cancelGesture();
    this.doc.defaultView?.removeEventListener('resize', this.onResize);
    this.host.remove();
  }

  private onPointerDown(event: PointerEvent): void {
    if (event.button !== undefined && event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    this.cancelGesture();
    this.pointerId = event.pointerId;
    this.startX = event.clientX;
    this.startY = event.clientY;
    if (typeof this.button.setPointerCapture === 'function' && event.pointerId !== undefined) {
      try {
        this.button.setPointerCapture(event.pointerId);
      } catch {
        // The pointer is already gone; the gesture ends with the next up/cancel.
      }
    }
    this.timer = setTimeout(() => {
      this.timer = null;
      this.longPressed = true;
      this.suppressClick = true;
      this.actions.onLongPress();
    }, FAB_LONG_PRESS_MS);
  }

  private onPointerMove(event: PointerEvent): void {
    event.stopPropagation();
    if (this.pointerId === null || event.pointerId !== this.pointerId) return;
    const dx = event.clientX - this.startX;
    const dy = event.clientY - this.startY;
    if (!this.dragging) {
      if (Math.hypot(dx, dy) <= FAB_DRAG_PX) return;
      this.dragging = true;
      this.cancelTimer();
      this.suppressClick = true;
      this.anchorToViewport();
    }
    this.moveBy(dx, dy);
    this.startX = event.clientX;
    this.startY = event.clientY;
  }

  private onPointerUp(event: PointerEvent): void {
    event.stopPropagation();
    if (this.pointerId === null || event.pointerId !== this.pointerId) return;
    const wasLongPress = this.longPressed;
    const wasDragging = this.dragging;
    this.cancelGesture();
    if (wasLongPress || wasDragging) {
      // The click that follows a long-press or a drag must not also tap.
      this.suppressClick = true;
    }
  }

  private cancelTimer(): void {
    if (this.timer === null) return;
    clearTimeout(this.timer);
    this.timer = null;
  }

  private cancelGesture(): void {
    this.cancelTimer();
    this.pointerId = null;
    this.longPressed = false;
    this.dragging = false;
    // A cancelled gesture fires no click, so a flag it set must not eat the next tap.
    this.suppressClick = false;
  }

  /** Switches from the bottom-right anchor to explicit coordinates for dragging. */
  private anchorToViewport(): void {
    const rect = this.host.getBoundingClientRect();
    this.host.style.right = 'auto';
    this.host.style.bottom = 'auto';
    this.host.style.left = `${String(Math.round(rect.left))}px`;
    this.host.style.top = `${String(Math.round(rect.top))}px`;
  }

  private moveBy(dx: number, dy: number): void {
    const left = Number.parseFloat(this.host.style.left || '0') + dx;
    const top = Number.parseFloat(this.host.style.top || '0') + dy;
    this.host.style.left = `${String(Math.round(left))}px`;
    this.host.style.top = `${String(Math.round(top))}px`;
    this.clampToViewport();
  }

  private clampToViewport(): void {
    if (this.host.style.left === '') return;
    const view = this.doc.defaultView;
    const width = view?.innerWidth ?? 800;
    const height = view?.innerHeight ?? 600;
    const margin = 8;
    const left = Math.min(
      Math.max(Number.parseFloat(this.host.style.left || '0'), margin),
      Math.max(margin, width - FAB_PX - margin),
    );
    const top = Math.min(
      Math.max(Number.parseFloat(this.host.style.top || '0'), margin),
      Math.max(margin, height - FAB_PX - margin),
    );
    this.host.style.left = `${String(Math.round(left))}px`;
    this.host.style.top = `${String(Math.round(top))}px`;
  }
}
