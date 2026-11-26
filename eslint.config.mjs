import js from '@eslint/js';
import prettier from 'eslint-config-prettier';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  {
    ignores: [
      '.output/**',
      '.wxt/**',
      '.orchestrator/**',
      'node_modules/**',
      'coverage/**',
      'test-results/**',
      'playwright-report/**',
      'public/**',
      'vendor/**',
      '.tools/**',
      'bench/.cache/**',
      'launch-video/**',
      'video/**',
    ],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...svelte.configs['flat/recommended'],
  {
    files: ['**/*.svelte'],
    languageOptions: {
      parserOptions: {
        parser: tseslint.parser,
      },
    },
  },
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.webextensions,
        // Build-time constants from `wxt.config.ts` (`vite.define`).
        __SAKUTTO_DEBUG_BUILD__: 'readonly',
        __SAKUTTO_FAKE_ENGINE__: 'readonly',
      },
    },
  },
  {
    files: ['src/**/*.{ts,svelte}'],
    rules: {
      'no-console': 'error',
    },
  },
  {
    files: ['*.{js,mjs,cjs,ts}', 'scripts/**/*', 'tests/**/*', 'e2e/**/*'],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
  prettier,
);
