import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { isPageCommandRequest } from '@/shared/messages';
import { boundShortcut, pageCommandAllowed } from '@/shared/shortcuts';

// Regression (1.3.0, Muse): the popup's "Translate this page" did nothing. The content script
// ran every page command through the keyboard-shortcut gate. The Edge package had no
// `suggested_key`, the browser reported an empty binding that the stored "assigned" flag
// marked as a user clear, and the gate dropped the popup's toggle: page:state stayed
// {active:false, done:0, total:0} with no error anywhere.

const read = (path: string): string => readFileSync(new URL(path, import.meta.url), 'utf8');

describe('page command source', () => {
  it('lets the popup and the context menu run with no shortcut bound', () => {
    const cleared = boundShortcut({ 'translate-page': '' } as never, 'translate-page', {
      assigned: true,
    });
    expect(cleared).toBe('');
    expect(pageCommandAllowed('action', () => cleared.length > 0)).toBe(true);
    expect(pageCommandAllowed(undefined, () => cleared.length > 0)).toBe(true);
    expect(pageCommandAllowed('shortcut', () => cleared.length > 0)).toBe(false);
    expect(pageCommandAllowed('shortcut', () => true)).toBe(true);
  });

  it('does not ask the shortcut gate about an action', () => {
    let asked = false;
    expect(
      pageCommandAllowed('action', () => {
        asked = true;
        return false;
      }),
    ).toBe(true);
    expect(asked).toBe(false);
  });

  it('accepts a command message with or without a source', () => {
    expect(isPageCommandRequest({ type: 'page:command', command: 'toggle' })).toBe(true);
    expect(
      isPageCommandRequest({ type: 'page:command', command: 'toggle', source: 'shortcut' }),
    ).toBe(true);
  });

  it('marks only browser keyboard commands as shortcuts', () => {
    const actions = read('../src/background/page-actions.ts');
    const messages = read('../src/shared/messages.ts');
    const content = read('../src/entrypoints/content.ts');
    const popup = read('../src/entrypoints/popup/App.svelte');
    expect(actions).toContain("runOnActiveTab(SHORTCUT_PAGE_COMMANDS[command], 'shortcut')");
    expect(actions).toContain('void runCommand(tabId, pageCommand);');
    expect(messages).toContain("source: PageCommandSource = 'action'");
    expect(popup).toContain("requestPageCommand(tabId, view?.active ? 'restore' : 'toggle')");
    expect(content).toContain('pageCommandAllowed(source,');
    expect(content).toContain('run(message.command, message.source)');
  });
});
