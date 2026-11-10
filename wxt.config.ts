import { defineConfig } from 'wxt';
import {
  CHROME_PERMISSIONS,
  EDGE_PERMISSIONS,
  FIREFOX_PERMISSIONS,
  OPTIONAL_HOST_PERMISSIONS,
  commandSuggestions,
  edgeCommands,
  firefoxGeckoSettings,
} from './src/shared/manifest';

const fixtureHostPermissions =
  process.env.SAKUTTO_FAKE_ENGINE === '1' || process.env.SAKUTTO_E2E_REAL === '1'
    ? ['http://127.0.0.1/*', 'http://localhost/*']
    : null;

export default defineConfig({
  srcDir: 'src',
  modules: ['@wxt-dev/module-svelte'],
  imports: false,
  manifestVersion: 3,
  vite: () => ({
    define: {
      __SAKUTTO_FAKE_ENGINE__: JSON.stringify(process.env.SAKUTTO_FAKE_ENGINE === '1'),
    },
  }),
  manifest: ({ browser }) => {
    const firefox = browser === 'firefox';
    const edge = browser === 'edge';
    // WXT passes `edge` through as the browser name; the build is
    // Chromium MV3, the same as Chrome, with Edge-specific manifest
    // differences below (no Translator API, no keyboard shortcuts).
    return {
      name: '__MSG_extensionName__',
      short_name: '__MSG_extensionShortName__',
      description: '__MSG_extensionDescription__',
      default_locale: 'en',
      ...(firefox || edge ? {} : { minimum_chrome_version: '116' }),
      permissions: [
        ...(firefox ? FIREFOX_PERMISSIONS : edge ? EDGE_PERMISSIONS : CHROME_PERMISSIONS),
      ],
      optional_host_permissions: [...OPTIONAL_HOST_PERMISSIONS],
      // Release builds ask for site access only when auto-translate is turned on.
      // The e2e build may script the local fixtures directly.
      ...(fixtureHostPermissions ? { host_permissions: fixtureHostPermissions } : {}),
      commands: edge ? edgeCommands() : commandSuggestions(firefox ? 'firefox' : 'chrome'),
      ...(firefox
        ? {
            browser_specific_settings: {
              gecko: firefoxGeckoSettings(),
            },
          }
        : {
            cross_origin_embedder_policy: { value: 'require-corp' },
            cross_origin_opener_policy: { value: 'same-origin' },
          }),
      content_security_policy: {
        extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
      },
      icons: {
        16: 'icon/16.png',
        32: 'icon/32.png',
        48: 'icon/48.png',
        128: 'icon/128.png',
      },
    };
  },
});
