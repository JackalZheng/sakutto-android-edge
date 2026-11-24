import {
  FIREFOX_EXTENSION_ID,
  FIREFOX_STRICT_MIN_VERSION,
  type ExtensionBrowser,
} from './browser.ts';
import { defaultShortcutsFor, SHORTCUT_COMMANDS } from './shortcuts.ts';

export const CHROME_PERMISSIONS = [
  'offscreen',
  'storage',
  'unlimitedStorage',
  'activeTab',
  'scripting',
  'contextMenus',
] as const;

export const FIREFOX_PERMISSIONS = [
  'storage',
  'unlimitedStorage',
  'activeTab',
  'scripting',
  'contextMenus',
] as const;

/**
 * Edge permissions. Edge ships the same Chromium MV3 APIs Sakutto uses
 * (including `offscreen`, which Microsoft lists for Android), but Edge
 * has no Chrome Translator API, so Edge builds are PLaMo-only.
 * `contextMenus` stays declared: on Android it is a no-op at runtime
 * (guarded in `installPageActions`), the popup is the touch entry point.
 */
export const EDGE_PERMISSIONS = [
  'offscreen',
  'storage',
  'unlimitedStorage',
  'activeTab',
  'scripting',
  'contextMenus',
] as const;

export const OPTIONAL_HOST_PERMISSIONS = ['https://*/*', 'http://*/*'] as const;

const COMMAND_MESSAGES = {
  'translate-page': '__MSG_commandTranslate__',
  'toggle-bilingual': '__MSG_commandBilingual__',
  'translate-selection': '__MSG_commandSelection__',
  'translate-compose': '__MSG_commandCompose__',
} as const;

export function commandSuggestions(
  browser: ExtensionBrowser,
): Record<
  (typeof SHORTCUT_COMMANDS)[number],
  { suggested_key: { default: string }; description: string }
> {
  const shortcuts = defaultShortcutsFor(browser);
  return {
    'translate-page': {
      suggested_key: { default: shortcuts['translate-page'] },
      description: COMMAND_MESSAGES['translate-page'],
    },
    'toggle-bilingual': {
      suggested_key: { default: shortcuts['toggle-bilingual'] },
      description: COMMAND_MESSAGES['toggle-bilingual'],
    },
    'translate-selection': {
      suggested_key: { default: shortcuts['translate-selection'] },
      description: COMMAND_MESSAGES['translate-selection'],
    },
    'translate-compose': {
      suggested_key: { default: shortcuts['translate-compose'] },
      description: COMMAND_MESSAGES['translate-compose'],
    },
  };
}

/**
 * Edge commands: the same four actions and the same Alt+A / Alt+B / Alt+S / Alt+T suggested
 * keys as Chrome. Edge on a desktop binds them at install; Edge for Android has no keyboard
 * and ignores `commands`, and its settings page shows the gesture panel instead of the keys.
 * 1.2.x–1.3.0-dev Edge packages left `suggested_key` out, so desktop Edge/Chromium got no
 * shortcut at install.
 */
export function edgeCommands(): ReturnType<typeof commandSuggestions> {
  return commandSuggestions('chrome');
}

export function firefoxGeckoSettings(): {
  id: string;
  strict_min_version: string;
  data_collection_permissions: { required: ['none'] };
} {
  return {
    id: FIREFOX_EXTENSION_ID,
    strict_min_version: FIREFOX_STRICT_MIN_VERSION,
    data_collection_permissions: { required: ['none'] },
  };
}
