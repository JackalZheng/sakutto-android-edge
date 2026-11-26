<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from 'wxt/browser';
  import {
    DEFAULT_MODEL_ID,
    LARGE_MODEL_BYTES,
    MODEL_LIST,
    getModel,
    modelLicense,
    type ModelEntry,
    type ModelId,
  } from '@/engine/model/catalog';
  import type { ModelStatus } from '@/shared/engine-status';
  import { hasLicenseConsent } from '@/shared/agreement';
  import { t, type MessageKey } from '@/shared/i18n';
  import {
    isModelProgressMessage,
    requestEngineStatus,
    requestModelCancel,
    requestModelDelete,
    requestModelDownload,
    requestModelInspect,
    requestModelUse,
  } from '@/shared/messages';
  import { loadSettings } from '@/shared/storage';
  import { isAndroidEdgeRuntime, isMobileModelAllowed } from '@/shared/mobile';
  import { loadCrashedModelLoads, type CrashedModelLoads } from '@/shared/model-load-guard';
  import { formatByteSize } from './format';

  interface Props {
    /** The active model changed (the engine section shows GPU notes for it). */
    onActiveChange?: (id: ModelId) => void;
  }

  let { onActiveChange }: Props = $props();

  const androidEdge = isAndroidEdgeRuntime();
  // 1.3.0: every catalog model, smallest file first, on desktop and on Edge for Android (where
  // a model over 1 GB runs on the CPU; see `phoneKeepsOffGpu`).
  const models: readonly ModelEntry[] = MODEL_LIST;

  function allowedHere(id: ModelId): boolean {
    return !androidEdge || isMobileModelAllowed(id);
  }

  /** Models whose load closed the engine page on this phone (`shared/model-load-guard.ts`). */
  let crashed = $state<CrashedModelLoads>({});

  const descriptionKey: Record<ModelId, MessageKey> = {
    'qwen3-0.6b-q4_k_m': 'modelDescQwen3Small',
    'plamo-3-translate-0.8b-q4_k_m': 'modelDescPlamoSmall',
    'gemma-3-1b-it-q4_k_m': 'modelDescGemma',
    'qwen3-1.7b-q4_k_m': 'modelDescQwen3Large',
    'hy-mt2-1.8b-q4_k_m': 'modelDescHyMt',
    'plamo-3-translate-2b-q4_k_m': 'modelDescPlamoLarge',
  };

  /** PLaMo links the bundled copy of its license; the others link the upstream terms. */
  function licenseUrl(id: ModelId): string {
    const license = modelLicense(id);
    if (license.id === 'plamo-community') {
      return browser.runtime.getURL('/licenses/PLaMo-Community-License-en.txt');
    }
    return license.url;
  }

  let statuses = $state<Partial<Record<ModelId, ModelStatus>>>({});
  let activeId = $state<ModelId>(DEFAULT_MODEL_ID);
  let busy = $state<ModelId | null>(null);

  function label(id: ModelId): string {
    return getModel(id).name;
  }

  function setActive(id: ModelId) {
    activeId = id;
    onActiveChange?.(id);
  }

  function apply(status: ModelStatus) {
    statuses = { ...statuses, [status.id]: status };
    if (status.state === 'loaded') setActive(status.id);
  }

  async function refresh() {
    const engine = await requestEngineStatus();
    if (engine) setActive(engine.model.id);
    const next: Partial<Record<ModelId, ModelStatus>> = { ...statuses };
    for (const model of models) {
      const status = await requestModelInspect(model.id);
      if (status) next[model.id] = status;
    }
    statuses = next;
    if (androidEdge) crashed = await loadCrashedModelLoads();
  }

  async function download(id: ModelId) {
    const stored = await loadSettings().catch(() => null);
    if (!hasLicenseConsent(stored)) {
      await browser.tabs
        .create({ url: browser.runtime.getURL('/onboarding.html') })
        .catch(() => undefined);
      return;
    }
    busy = id;
    await requestModelDownload(id);
    busy = null;
  }

  async function use(id: ModelId) {
    busy = id;
    const status = await requestModelUse(id);
    if (status) apply(status);
    setActive(id);
    // A CPU crash record is cleared by choosing the model again; a WebGPU one stays.
    if (crashed[id]?.backend === 'cpu-wasm') {
      const next = { ...crashed };
      delete next[id];
      crashed = next;
    }
    busy = null;
  }

  async function remove(id: ModelId) {
    busy = id;
    const status = await requestModelDelete(id);
    if (status) apply(status);
    if (activeId === id) setActive(DEFAULT_MODEL_ID);
    busy = null;
  }

  onMount(() => {
    const listener = (message: unknown) => {
      if (!isModelProgressMessage(message)) return;
      apply(message.status);
    };
    browser.runtime.onMessage.addListener(listener);
    void refresh();
    return () => browser.runtime.onMessage.removeListener(listener);
  });
</script>

<ul class="models">
  {#each models as model (model.id)}
    {@const id = model.id}
    {@const status = statuses[id]}
    {@const state = status?.state ?? 'absent'}
    {@const downloading = state === 'downloading' || state === 'verifying'}
    <li>
      <div class="summary">
        <strong>{label(id)}</strong>
        <span class="muted">{formatByteSize(model.size)}</span>
        {#if id === DEFAULT_MODEL_ID}
          <span class="tag">{t('modelDefaultBadge')}</span>
        {/if}
        {#if model.gpuExperimental}
          <span class="tag experimental">{t('modelGpuExperimental')}</span>
        {/if}
        {#if activeId === id && (state === 'ready' || state === 'loaded')}
          <span class="badge">{t('modelInUse')}</span>
        {:else if state === 'ready' || state === 'loaded'}
          <span class="muted downloaded">{t('modelDownloaded')}</span>
        {/if}
      </div>
      <p class="muted detail">{t(descriptionKey[id])}</p>
      <p class="muted detail">
        {t('modelLicenseLabel')}:
        <a href={licenseUrl(id)} target="_blank" rel="noreferrer">{modelLicense(id).name}</a>
      </p>
      {#if model.license === 'gemma'}
        <p class="muted detail">{t('modelGemmaTerms')}</p>
      {/if}
      {#if model.gpuExperimental}
        <p class="muted detail">{t('modelGpuExperimentalHelp')}</p>
      {/if}
      {#if androidEdge && model.size > LARGE_MODEL_BYTES}
        <p class="muted detail warning">{t('modelLargeMobileWarning')}</p>
      {/if}
      {#if androidEdge && crashed[id] !== undefined}
        <p class="muted detail warning" role="status">
          {t(crashed[id]?.backend === 'webgpu' ? 'modelLoadCrashedGpu' : 'modelLoadCrashed')}
        </p>
      {/if}
      {#if status && (downloading || state === 'partial' || state === 'loading')}
        <p class="muted">
          {formatByteSize(status.received)} / {formatByteSize(status.total)}
        </p>
      {/if}
      <div class="actions">
        {#if downloading}
          <button
            class="button secondary"
            type="button"
            aria-label={`${label(id)} ${t('downloadCancel')}`}
            onclick={() => requestModelCancel(id)}
          >
            {t('downloadCancel')}
          </button>
        {:else if state === 'ready' || state === 'loaded'}
          {#if activeId !== id && allowedHere(id)}
            <button
              class="button"
              type="button"
              aria-label={`${label(id)} ${t('modelUse')}`}
              disabled={busy === id}
              onclick={() => use(id)}
            >
              {t('modelUse')}
            </button>
          {/if}
        {:else if state === 'loading'}
          <!-- The file is here and the runtime is reading it. A Download button in this
               state is a start control for something that is already under way. -->
          <p class="muted">{t('runtimeLoading')}</p>
        {:else if allowedHere(id)}
          <button
            class="button"
            type="button"
            aria-label={`${label(id)} ${state === 'partial' ? t('downloadResume') : t('modelDownload')}`}
            disabled={busy === id}
            onclick={() => download(id)}
          >
            {state === 'partial' ? t('downloadResume') : t('modelDownload')}
          </button>
        {/if}
        {#if state !== 'absent'}
          <button
            class="button secondary"
            type="button"
            aria-label={`${label(id)} ${t('modelDelete')}`}
            disabled={busy === id}
            onclick={() => remove(id)}
          >
            {t('modelDelete')}
          </button>
        {/if}
      </div>
    </li>
  {/each}
</ul>

<style>
  .models {
    display: grid;
    gap: var(--space-3);
  }

  li {
    display: grid;
    gap: var(--space-2);
  }

  .summary {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: var(--space-2);
  }

  .badge {
    color: var(--accent-strong);
    font-size: 11px;
    font-weight: 700;
  }

  .tag {
    border: 1px solid currentColor;
    border-radius: 999px;
    padding: 0 6px;
    font-size: 11px;
    color: var(--accent-strong);
  }

  .tag.experimental {
    color: var(--warning, #b45309);
  }

  .downloaded {
    font-size: 11px;
  }

  .detail {
    margin: 0;
    font-size: 12px;
  }

  .warning {
    color: var(--warning, #b45309);
  }

  .actions {
    display: flex;
    flex-wrap: wrap;
    gap: var(--space-2);
  }
</style>
