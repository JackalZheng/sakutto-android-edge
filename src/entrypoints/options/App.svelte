<script lang="ts">
  import { onMount } from 'svelte';
  import { browser } from 'wxt/browser';
  import { shouldHintBuiltInFallback } from '@/engine/model/suggestion';
  import type { GlossaryEntry } from '@/engine/types';
  import type { SiteAuto, SiteRule } from '@/dom/sites';
  import { loadSiteRules, pageHost, saveSiteRule, subscribeSiteRules } from '@/dom/sites';
  import type { EngineCapabilities } from '@/shared/capabilities';
  import type { EngineStatus } from '@/shared/engine-status';
  import { hasBuiltInTranslatorUi, isFirefoxBuild } from '@/shared/browser';
  import { currentUiLanguage, t, tForBrowser, type MessageKey } from '@/shared/i18n';
  import { ADRENO_GPU_PROFILES } from '@/shared/adreno-profiles';
  import { DEFAULT_MODEL_ID, isGpuExperimentalModel, type ModelId } from '@/engine/model/catalog';
  import { hasAllSitesAccess, requestAllSitesAccess } from '@/shared/all-sites';
  import {
    requestCacheClear,
    requestEngineCapabilities,
    requestEngineStatus,
    requestEngineTier,
    requestQualitySample,
  } from '@/shared/messages';
  import type { QualitySample } from '@/shared/quality';
  import { PLAMO_OFFICIAL_LINKS, PROJECT_LINKS } from '@/shared/plamo-license';
  import type {
    AdrenoGpuProfile,
    IdleMinutes,
    Settings,
    SettingsPatch,
    TierPreference,
  } from '@/shared/settings';
  import { isAdrenoGpuProfile, isIdleMinutes, isTierPreference } from '@/shared/settings';
  import { isAndroidEdgeRuntime } from '@/shared/mobile';
  import { webGpuDebugProfileFromUrl } from '@/shared/gpu-diagnostic';
  import type { TargetLanguage } from '@/shared/languages';
  import { loadSettings, saveSettings, subscribeSettings } from '@/shared/storage';
  import EngineStatusView from '@/ui/EngineStatus.svelte';
  import GpuDiagnosticView from '@/ui/GpuDiagnostic.svelte';
  import LanguageField from '@/ui/LanguageField.svelte';
  import GlossaryEditor from '@/ui/GlossaryEditor.svelte';
  import ModelManager from '@/ui/ModelManager.svelte';
  import { runtimeLabel } from '@/ui/runtime-label';
  import SettingsFields from '@/ui/SettingsFields.svelte';
  import ShortcutSettings from '@/ui/ShortcutSettings.svelte';

  type SettingsError = 'load' | 'save' | null;
  type CacheState = 'cleared' | 'failed' | null;

  const licenseLinks = {
    plamoJa: browser.runtime.getURL('/licenses/PLaMo-Community-License-ja.txt'),
    plamoEn: browser.runtime.getURL('/licenses/PLaMo-Community-License-en.txt'),
    mit: browser.runtime.getURL('/licenses/LICENSE.txt'),
    notice: browser.runtime.getURL('/licenses/NOTICE.txt'),
  };

  const officialLinks = PLAMO_OFFICIAL_LINKS;
  const projectLinks = PROJECT_LINKS;
  /** Edge for Android has no keyboard, so this panel shows gestures, not shortcuts. */
  const androidEdge = isAndroidEdgeRuntime();
  /**
   * The GPU harness is a debug tool with no address-bar entry on phones: show
   * its button on Android Edge, or on desktop when the settings page was opened
   * with a WebGPU debug profile (?webgpuDebug=…), the same signal the engine uses.
   */
  // Release builds do not ship the harness page (it is in `pnpm build:debug` only).
  const showGpuHarness =
    __SAKUTTO_DEBUG_BUILD__ && (androidEdge || webGpuDebugProfileFromUrl(location.href) !== null);
  // Every profile from `shared/adreno-profiles.ts` (the non-garbling GPU harness setups), each
  // with its recommendation shown under the selector.
  const gpuProfiles: readonly AdrenoGpuProfile[] = ADRENO_GPU_PROFILES;
  const gpuProfileHelpKey = {
    optimized: 'gpuProfileOptimizedHelp',
    q4v2: 'gpuProfileQ4v2Help',
    r6: 'gpuProfileR6Help',
    'low-memory': 'gpuProfileLowMemoryHelp',
    compat: 'gpuProfileCompatHelp',
    r4: 'gpuProfileR4Help',
  } as const satisfies Record<AdrenoGpuProfile, MessageKey>;
  const gpuProfileLabelKey = {
    optimized: 'gpuProfileOptimized',
    q4v2: 'gpuProfileQ4v2',
    r6: 'gpuProfileR6',
    'low-memory': 'gpuProfileLowMemory',
    compat: 'gpuProfileCompat',
    r4: 'gpuProfileR4',
  } as const satisfies Record<AdrenoGpuProfile, MessageKey>;
  let activeModelId = $state<ModelId>(DEFAULT_MODEL_ID);

  let settings = $state<Settings | null>(null);
  let settingsError = $state<SettingsError>(null);
  let uiLanguage = $state(currentUiLanguage());
  let rules = $state<Record<string, SiteRule>>({});
  let hostInput = $state('');
  let hostError = $state(false);
  let glossaryScope = $state('global');
  let permissionError = $state(false);
  let engine = $state<EngineStatus | null>(null);
  let capabilities = $state<EngineCapabilities | null>(null);
  let capabilitiesFailed = $state(false);
  let quality = $state<QualitySample | null>(null);
  let cacheState = $state<CacheState>(null);
  let actionFailed = $state(false);
  let harnessOpenFailed = $state(false);
  let allSites = $state<boolean | null>(null);
  let allSitesFailed = $state(false);

  let siteList = $derived(
    Object.values(rules).sort((left, right) => left.host.localeCompare(right.host)),
  );
  let glossaryEntries = $derived(
    glossaryScope === 'global'
      ? (settings?.glossary ?? [])
      : (rules[glossaryScope]?.glossary ?? []),
  );
  let showSlow = $derived(
    hasBuiltInTranslatorUi() &&
      shouldHintBuiltInFallback(engine?.tier ?? null, quality?.milliseconds ?? null),
  );
  let engineLine = $derived.by(() => {
    const label = runtimeLabel({
      status: engine,
      pageSimple: false,
      targetLanguage: settings?.targetLanguage ?? 'en',
    });
    return t(label.key, label.substitutions);
  });

  async function changeFabEverywhere(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    allSitesFailed = false;
    // The grant prompt has to open inside this click, before anything else is awaited.
    if (input.checked && allSites !== true) {
      allSites = await requestAllSitesAccess();
      allSitesFailed = !allSites;
    }
    await updateSettings({ fabEverywhere: input.checked });
  }

  async function grantAllSites() {
    allSites = await requestAllSitesAccess();
    allSitesFailed = !allSites;
  }

  async function updateSettings(patch: SettingsPatch) {
    try {
      settings = await saveSettings(patch);
      settingsError = null;
    } catch {
      settingsError = 'save';
    }
  }

  async function loadStoredSettings() {
    try {
      settings = await loadSettings();
      settingsError = null;
    } catch {
      settingsError = 'load';
    }
  }

  async function loadRules() {
    rules = await loadSiteRules();
  }

  async function patchSite(host: string, patch: Partial<SiteRule>) {
    const current = rules[host];
    if (!current) return;
    await saveSiteRule({ ...current, ...patch, host });
    await loadRules();
  }

  async function changeAuto(host: string, auto: SiteAuto) {
    permissionError = false;
    if (auto === 'always') {
      const granted = await browser.permissions
        .request({ origins: [`https://${host}/*`, `http://${host}/*`] })
        .catch(() => false);
      if (!granted) {
        permissionError = true;
        return;
      }
    }
    const current = rules[host];
    if (!current) return;
    await saveSiteRule({ ...current, auto });
    await loadRules();
  }

  async function addSite() {
    const host = pageHost(`https://${hostInput.trim()}`);
    hostError = host === null;
    if (!host) return;
    const current = rules[host] ?? { host, auto: 'ask' as const };
    await saveSiteRule(current);
    hostInput = '';
    glossaryScope = host;
    await loadRules();
  }

  async function saveGlossary(entries: GlossaryEntry[]) {
    if (glossaryScope === 'global') {
      await updateSettings({ glossary: entries });
      return;
    }
    const current = rules[glossaryScope];
    if (!current) return;
    await saveSiteRule({ ...current, host: glossaryScope, glossary: entries });
    await loadRules();
  }

  async function changeTier(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    if (!isTierPreference(select.value)) return;
    const tier: TierPreference = select.value;
    await updateSettings({ tier });
    await requestEngineTier(tier);
    engine = await requestEngineStatus();
  }

  async function changeGpuProfile(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    if (!isAdrenoGpuProfile(select.value)) return;
    const gpuProfile: AdrenoGpuProfile = select.value;
    await updateSettings({ gpuProfile });
    // The engine reads the profile when the tier is applied; reload only if WebGPU may be in use
    // (an explicit GPU choice, or Automatic on an Adreno).
    const tier = settings?.tier;
    if (tier === 'g' || (tier === 'auto' && capabilities?.gpuAdreno === true)) {
      await requestEngineTier(tier);
      engine = await requestEngineStatus();
    }
  }

  async function changeIdle(event: Event) {
    const select = event.currentTarget as HTMLSelectElement;
    const minutes = Number(select.value);
    if (!isIdleMinutes(minutes)) return;
    const idleMinutes: IdleMinutes = minutes;
    await updateSettings({ idleMinutes });
  }

  async function clearCache() {
    cacheState = (await requestCacheClear()) ? 'cleared' : 'failed';
  }

  async function openOnboarding() {
    actionFailed = false;
    try {
      await browser.tabs.create({ url: browser.runtime.getURL('/onboarding.html') });
    } catch {
      actionFailed = true;
    }
  }

  async function openTyped() {
    actionFailed = false;
    try {
      await browser.tabs.create({ url: browser.runtime.getURL('/typed.html') });
    } catch {
      actionFailed = true;
    }
  }

  async function openGpuHarness() {
    harnessOpenFailed = false;
    try {
      await browser.tabs.create({ url: browser.runtime.getURL('/gpu-harness.html') });
    } catch {
      harnessOpenFailed = true;
    }
  }

  onMount(() => {
    const unsubscribe = subscribeSettings((next) => {
      settings = next;
      settingsError = null;
    });
    const unsubscribeRules = subscribeSiteRules((next) => {
      rules = next;
    });
    void loadStoredSettings();
    void loadRules();
    if (androidEdge) {
      void hasAllSitesAccess().then((granted) => {
        allSites = granted;
      });
    }
    void requestEngineStatus().then((status) => {
      engine = status;
    });
    void requestEngineCapabilities().then((value) => {
      capabilities = value;
      capabilitiesFailed = value === null;
    });
    void requestQualitySample().then((sample) => {
      quality = sample;
    });
    return () => {
      unsubscribe();
      unsubscribeRules();
    };
  });
</script>

<main class="page">
  <header class="page-header">
    <p class="eyebrow">{t('optionsTitle')}</p>
    <h1>{t('extensionName')}</h1>
    <p class="muted">{t('optionsLead')}</p>
  </header>

  <section class="panel">
    <h2>{t('settingsLabel')}</h2>
    {#if settings}
      <SettingsFields {settings} {uiLanguage} onChange={updateSettings} />
      {#if androidEdge}
        <label class="check">
          <input type="checkbox" checked={settings.fabEverywhere} onchange={changeFabEverywhere} />
          <span>{t('fabEverywhereSetting')}</span>
        </label>
        <p class="muted help">{t('fabEverywhereSettingHelp')}</p>
        {#if settings.fabEverywhere && allSites === false}
          <button class="link-button" type="button" onclick={grantAllSites}>
            {t('fabEverywhereGrant')}
          </button>
        {/if}
        {#if allSitesFailed}
          <p class="error" role="alert">{t('fabEverywhereGrantFailed')}</p>
        {/if}
      {/if}
      <p class="muted help">{t('optionsDefaultHint')}</p>
      {#if settingsError === 'save'}
        <p class="error" role="alert">{t('errorSaveSettings')}</p>
      {/if}
    {:else if settingsError === 'load'}
      <p class="error" role="alert">{t('errorSettings')}</p>
      <button class="link-button" type="button" onclick={loadStoredSettings}>{t('retry')}</button>
    {:else}
      <p class="muted">{t('statusChecking')}</p>
    {/if}
  </section>

  <section class="panel">
    <h2>{t(androidEdge ? 'gestureHeading' : 'shortcutHeading')}</h2>
    {#if !androidEdge}
      <p class="muted help">{t(isFirefoxBuild() ? 'shortcutHelpFirefox' : 'shortcutHelp')}</p>
    {/if}
    <p class="muted help">{t('composeHelp')}</p>
    <ShortcutSettings
      enabled={settings?.shortcutsEnabled}
      onToggle={(command, on) => {
        void updateSettings({ shortcutsEnabled: { [command]: on } });
      }}
    />
  </section>

  <section class="panel">
    <h2>{t('siteRulesHeading')}</h2>
    <p class="muted help">{t('optionsSiteLive')}</p>
    {#if siteList.length === 0}
      <p class="muted">{t('noSiteRules')}</p>
    {:else}
      <ul class="sites">
        {#each siteList as rule (rule.host)}
          <li>
            <span class="site-host">{rule.host}</span>
            <div class="site-controls">
              <LanguageField
                value={rule.targetLanguage ?? settings?.targetLanguage ?? 'en'}
                {uiLanguage}
                onChange={(language: TargetLanguage) => {
                  void patchSite(rule.host, { targetLanguage: language });
                }}
              />
              <label class="field">
                <span>{t('displayLabel')}</span>
                <select
                  aria-label={`${rule.host} ${t('displayLabel')}`}
                  value={rule.display ?? settings?.display ?? 'replace'}
                  onchange={(event) => {
                    const select = event.currentTarget as HTMLSelectElement;
                    void patchSite(rule.host, {
                      display: select.value === 'bilingual' ? 'bilingual' : 'replace',
                    });
                  }}
                >
                  <option value="replace">{t('displayReplace')}</option>
                  <option value="bilingual">{t('displayBilingual')}</option>
                </select>
              </label>
              <label class="field">
                <span>{t('siteRulesHeading')}</span>
                <select
                  aria-label={`${rule.host} ${t('siteRulesHeading')}`}
                  value={rule.auto}
                  onchange={(event) => {
                    const select = event.currentTarget as HTMLSelectElement;
                    const auto =
                      select.value === 'always' || select.value === 'never' ? select.value : 'ask';
                    void changeAuto(rule.host, auto);
                  }}
                >
                  <option value="ask">{t('siteRuleAsk')}</option>
                  <option value="always">{t('siteRuleAlways')}</option>
                  <option value="never">{t('siteRuleNever')}</option>
                </select>
              </label>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
    {#if permissionError}
      <p class="error" role="alert">{t('errorPermission')}</p>
    {/if}
    <form
      class="add-site"
      onsubmit={(event) => {
        event.preventDefault();
        void addSite();
      }}
    >
      <label>
        <span>{t('siteHostLabel')}</span>
        <input bind:value={hostInput} autocomplete="off" spellcheck="false" />
      </label>
      <button class="button secondary" type="submit">{t('glossaryAddSite')}</button>
    </form>
    {#if hostError}
      <p class="error" role="alert">{t('siteHostInvalid')}</p>
    {/if}
  </section>

  <section class="panel">
    <h2>{t('glossaryHeading')}</h2>
    <p id="glossary-help" class="muted help">{t('glossaryHelp')}</p>
    <label class="field">
      <span>{t('glossaryScopeLabel')}</span>
      <select bind:value={glossaryScope}>
        <option value="global">{t('glossaryGlobal')}</option>
        {#each siteList as rule (rule.host)}
          <option value={rule.host}>{rule.host}</option>
        {/each}
      </select>
    </label>
    <GlossaryEditor
      entries={glossaryEntries}
      describedBy="glossary-help"
      onChange={(entries) => void saveGlossary(entries)}
    />
  </section>

  <section class="panel">
    <h2>{t('modelHeading')}</h2>
    <ModelManager onActiveChange={(id) => (activeModelId = id)} />
    {#if showSlow}
      <p class="muted help">{t('slowDeviceHint')}</p>
    {/if}
    <button class="button secondary" type="button" onclick={clearCache}>{t('cacheClear')}</button>
    {#if cacheState === 'cleared'}
      <p class="muted" role="status">{t('cacheCleared')}</p>
    {:else if cacheState === 'failed'}
      <p class="error" role="alert">{t('cacheClearFailed')}</p>
    {/if}
  </section>

  <section class="panel">
    <h2>{t('engineHeading')}</h2>
    {#if settings}
      <label class="field">
        <span>{t('tierLabel')}</span>
        <select value={settings.tier} onchange={changeTier}>
          <option value="auto">{t('tierAuto')}</option>
          <option value="g">{t('tierGpu')}</option>
          {#if isFirefoxBuild()}
            <option value="c1">{t('tierCpu1')}</option>
          {:else}
            <option value="c8">{t('tierCpu8')}</option>
            <option value="c4">{t('tierCpu4')}</option>
          {/if}
        </select>
      </label>
      <p class="muted help">{t('tierHelp')}</p>
      {#if capabilities?.gpuAdreno === true || androidEdge}
        <p class="muted help">{t('tierAdrenoAutoHelp')}</p>
      {/if}
      {#if androidEdge && settings.tier === 'g'}
        <p class="error" role="note">{t('tierGpuMobileWarning')}</p>
      {/if}
      {#if (capabilities?.gpuAdreno === true || androidEdge) && (settings.tier === 'g' || (settings.tier === 'auto' && capabilities?.gpuAdreno === true))}
        <label class="field">
          <span>{t('gpuProfileLabel')}</span>
          <select value={settings.gpuProfile} onchange={changeGpuProfile}>
            {#each gpuProfiles as profile (profile)}
              <option value={profile}>{t(gpuProfileLabelKey[profile])}</option>
            {/each}
          </select>
        </label>
        <p class="muted help" role="note">{t(gpuProfileHelpKey[settings.gpuProfile])}</p>
        {#if settings.tier === 'auto' && isGpuExperimentalModel(activeModelId)}
          <p class="muted help">{t('gpuProfileExperimentalAuto')}</p>
        {/if}
      {:else if capabilities?.gpuAdreno === true || androidEdge}
        <p class="muted help">{t('gpuProfileInactive')}</p>
      {/if}
      <label class="field">
        <span>{t('idleLabel')}</span>
        <select value={String(settings.idleMinutes)} onchange={changeIdle}>
          <option value="1">{t('idle1')}</option>
          <option value="5">{t('idle5')}</option>
          <option value="15">{t('idle15')}</option>
          <option value="0">{t('idleNever')}</option>
        </select>
      </label>
    {/if}
    <p class="engine-line">{engineLine}</p>
    <h3>{t('diagHeading')}</h3>
    <dl class="diag">
      <div>
        <dt>{t('diagGpu')}</dt>
        <dd>
          {capabilities?.gpuName ?? t('diagGpuNone')}{capabilities?.gpuAdreno === true
            ? ` · ${t('diagGpuAdreno')}`
            : ''}
        </dd>
      </div>
      <div>
        <dt>{t('diagBackend')}</dt>
        <dd>
          {engine?.backend === 'webgpu'
            ? t('diagBackendGpu')
            : engine?.backend === 'cpu-wasm'
              ? t('diagBackendCpu')
              : t('diagBackendUnknown')}
        </dd>
      </div>
      <div>
        <dt>{t('diagThreads')}</dt>
        <dd>{engine?.threads ?? '—'}</dd>
      </div>
      <div>
        <dt>{t('diagParallel')}</dt>
        <dd>{engine?.parallel ?? '—'}</dd>
      </div>
    </dl>
    <p class="muted help">{t('diagRateHint')}</p>
    <EngineStatusView {capabilities} failed={capabilitiesFailed} detailed />
    <GpuDiagnosticView diagnostic={engine?.gpuDiagnostic} />
    {#if showGpuHarness}
      <button class="button secondary" type="button" onclick={openGpuHarness}>
        {t('openGpuHarness')}
      </button>
      <p class="muted help">{t('gpuHarnessHelp')}</p>
      {#if harnessOpenFailed}
        <p class="error" role="alert">{t('errorAction')}</p>
      {/if}
    {/if}
  </section>

  <section class="panel">
    <h2>{t('privacyHeading')}</h2>
    <p>{t('privacyBody')}</p>
    <p class="muted">{tForBrowser('privacyDestinations', 'privacyDestinationsFirefox')}</p>
    <ul class="links">
      <li>
        <a href={projectLinks.privacy} target="_blank" rel="noreferrer">{t('privacyPolicyLink')}</a>
      </li>
      <li>
        <a href={projectLinks.source} target="_blank" rel="noreferrer">{t('sourceCodeLink')}</a>
      </li>
    </ul>
  </section>

  <section class="panel">
    <h2>{t('licensesHeading')}</h2>
    <p>{t('licensesBody')}</p>
    <h3>{t('onboardingCommercialHeading')}</h3>
    <div class="terms">
      <div>
        <h4>{t('onboardingPersonalHeading')}</h4>
        <p class="muted">{t('onboardingPersonalBody')}</p>
      </div>
      <div>
        <h4>{t('onboardingCommercialHeading2')}</h4>
        <p class="muted">{t('onboardingCommercialBody')}</p>
        <ul class="conditions">
          <li>{t('onboardingCommercialRegistration')}</li>
          <li>{t('onboardingCommercialRevenue')}</li>
        </ul>
      </div>
      <div>
        <h4>{t('onboardingCommercialUnmetHeading')}</h4>
        <p class="muted">{t('onboardingCommercialUnmetBody')}</p>
      </div>
    </div>
    <p class="muted">{t('onboardingAgeNotice')}</p>
    <details class="more">
      <summary>{t('onboardingOtherLicenses')}</summary>
      <ul class="links">
        <li>
          <a href={officialLinks.licenseJa} target="_blank" rel="noreferrer"
            >{t('plamoOfficialLicenseJa')}</a
          >
        </li>
        <li>
          <a href={officialLinks.licenseEn} target="_blank" rel="noreferrer"
            >{t('plamoOfficialLicenseEn')}</a
          >
        </li>
        <li>
          <a href={officialLinks.registration} target="_blank" rel="noreferrer"
            >{t('plamoRegistrationLink')}</a
          >
        </li>
        <li>
          <a href={licenseLinks.plamoJa} target="_blank" rel="noreferrer">{t('licensePlaMoJa')}</a>
        </li>
        <li>
          <a href={licenseLinks.plamoEn} target="_blank" rel="noreferrer">{t('licensePlaMoEn')}</a>
        </li>
        <li><a href={licenseLinks.mit} target="_blank" rel="noreferrer">{t('licenseMit')}</a></li>
        <li>
          <a href={licenseLinks.notice} target="_blank" rel="noreferrer">{t('licenseNotice')}</a>
        </li>
      </ul>
    </details>
    <p class="muted">{t('licenseModelNote')}</p>
  </section>

  <div class="actions">
    <button class="button" type="button" onclick={openOnboarding}>{t('openOnboarding')}</button>
    <button class="button secondary" type="button" onclick={openTyped}>{t('openTyped')}</button>
  </div>
  {#if actionFailed}
    <p class="error" role="alert">{t('errorAction')}</p>
  {/if}

  <footer class="footer">
    <p class="notice">{t('machineTranslationNotice')}</p>
    <p class="notice">{t('unofficialNotice')}</p>
  </footer>
</main>

<style>
  .check {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    color: var(--text);
    font-size: 13px;
  }
  .page {
    display: grid;
    gap: var(--space-4);
    max-width: 640px;
    margin: 0 auto;
    padding: var(--space-6) var(--space-5);
  }

  .page-header,
  .field,
  .add-site {
    display: grid;
    gap: var(--space-1);
  }

  .eyebrow {
    color: var(--accent-strong);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  h1 {
    font-size: 22px;
    line-height: 1.3;
  }

  h2 {
    margin-bottom: var(--space-3);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
  }

  .help,
  .engine-line {
    margin-bottom: var(--space-3);
  }

  .engine-line {
    font-weight: 700;
  }

  h3 {
    margin: var(--space-3) 0 var(--space-2);
    font-size: 12px;
    font-weight: 700;
  }

  h4 {
    margin: 0 0 var(--space-1);
    font-size: 13px;
    font-weight: 700;
  }

  .terms {
    display: grid;
    gap: var(--space-3);
  }

  .terms .conditions {
    display: grid;
    gap: var(--space-1);
    margin-top: var(--space-2);
    padding-left: 18px;
    list-style: disc;
  }

  .more {
    margin-top: var(--space-3);
  }

  .more > summary {
    cursor: pointer;
    color: var(--accent-strong);
    font-weight: 600;
  }

  .more .links {
    margin-top: var(--space-2);
  }

  .diag {
    display: grid;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }

  .diag div {
    display: flex;
    justify-content: space-between;
    gap: var(--space-3);
  }

  .diag dt {
    color: var(--muted);
  }

  .diag dd {
    margin: 0;
    font-variant-numeric: tabular-nums;
  }

  .sites {
    display: grid;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }

  .sites li {
    display: grid;
    gap: var(--space-2);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--border);
  }

  .site-host {
    font-weight: 700;
  }

  .site-controls {
    display: grid;
    gap: var(--space-2);
  }

  .add-site {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--space-3);
  }

  .add-site {
    align-items: end;
    margin-top: var(--space-3);
  }

  .add-site label,
  .field {
    display: grid;
    flex: 1;
    gap: var(--space-1);
  }

  .field > span,
  .add-site span {
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
  }

  select,
  input {
    width: 100%;
    padding: 7px 9px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--surface);
    color: var(--text);
  }

  .links {
    display: grid;
    gap: var(--space-2);
    margin-bottom: var(--space-3);
  }

  .actions {
    display: flex;
    justify-content: flex-end;
  }

  .footer {
    display: grid;
    gap: var(--space-1);
  }
</style>
