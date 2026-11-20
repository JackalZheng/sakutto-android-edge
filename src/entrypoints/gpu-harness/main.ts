import {
  buildHarnessConfigs,
  compareAdapterToReference,
  compareRuns,
  firstStreamedToken,
  HARNESS_MAX_TOKENS,
  HARNESS_SOURCE_LANGUAGE,
  HARNESS_SOURCE_TEXT,
  HARNESS_TARGET_LANGUAGE,
  type HarnessConfig,
  type HarnessRunResult,
  verdictForRun,
} from '@/engine/debug-harness';
import { createWllamaRuntime, gpuEvidence, lastWebGpuLoadLog } from '@/engine/in-browser/runtime';
import { DEFAULT_MODEL_ID } from '@/engine/model/catalog';
import { OpfsModelFileStore } from '@/engine/model/file-store';
import { ModelManager } from '@/engine/model/manager';
import { buildTranslationPrompt } from '@/engine/prompt/pfml3';
import { probeCapabilities, probeWebGpuAdapter } from '@/shared/capabilities';
import {
  parseWebGpuLog,
  resetWebGpuDebugProfile,
  setWebGpuDebugProfile,
} from '@/shared/gpu-diagnostic';
import { isAndroidEdgeRuntime } from '@/shared/mobile';

/**
 * Stage 1 debug harness page. Deliberately standalone: it drives ModelManager
 * directly instead of going through the offscreen engine host, so the product
 * fallback logic (Auto → CPU on Android, garbage guard) cannot mask what the
 * GPU actually does. Selections are built with `selectionForTier`, bypassing
 * `resolveSelection` for the same reason.
 */

const statusEl = document.querySelector<HTMLElement>('#status')!;
const adapterEl = document.querySelector<HTMLElement>('#adapter')!;
const logEl = document.querySelector<HTMLElement>('#log')!;
const resultsEl = document.querySelector<HTMLElement>('#results')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;

function log(line: string): void {
  logEl.textContent += `${line}\n`;
}

function describe(value: unknown): string {
  return value === null || value === undefined ? 'n/a' : String(value);
}

async function runConfig(
  manager: ModelManager,
  config: HarnessConfig,
  prompt: string,
): Promise<HarnessRunResult> {
  const result: HarnessRunResult = {
    configId: config.id,
    ok: false,
    output: '',
    firstToken: '',
    ttftMs: null,
    totalMs: null,
    error: null,
    gpuEvidence: null,
    correctnessProbe: null,
    correctnessProbeSubgroupSize: null,
    correctnessProbeMs: null,
  };
  log(`--- ${config.label} ---`);
  try {
    if (config.debugProfile) setWebGpuDebugProfile(config.debugProfile);
    else resetWebGpuDebugProfile();
    // Force a fresh load per config: the debug profile is read when the
    // wllama worker is created, and two configs can share one selection.
    await manager.unload();
    const loaded = await manager.ensureLoaded(DEFAULT_MODEL_ID, config.selection);
    if (config.selection.backend === 'webgpu') {
      const lines = lastWebGpuLoadLog();
      result.gpuEvidence = gpuEvidence(lines);
      const parsed = parseWebGpuLog(lines);
      result.correctnessProbe = parsed.correctnessProbe;
      result.correctnessProbeSubgroupSize = parsed.correctnessProbeSubgroupSize;
      result.correctnessProbeMs = parsed.correctnessProbeMs;
      log(`gpu evidence: ${result.gpuEvidence}`);
      for (const line of lines) {
        if (line.includes('ggml_webgpu') || line.includes('offload')) log(`  ${line}`);
      }
    }
    const chunks: string[] = [];
    const controller = new AbortController();
    const started = performance.now();
    let firstAt: number | null = null;
    await loaded.runtime.complete({
      prompt,
      maxTokens: HARNESS_MAX_TOKENS,
      signal: controller.signal,
      onToken: (text) => {
        if (firstAt === null) firstAt = performance.now();
        chunks.push(text);
      },
    });
    const ended = performance.now();
    result.ok = true;
    result.output = chunks.join('');
    result.firstToken = firstStreamedToken(chunks);
    result.ttftMs = firstAt === null ? null : Math.round(firstAt - started);
    result.totalMs = Math.round(ended - started);
    log(
      `ok: ttft=${describe(result.ttftMs)}ms total=${result.totalMs}ms first token=${JSON.stringify(result.firstToken)}`,
    );
    log(`output: ${result.output}`);
  } catch (error) {
    result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    log(`FAILED: ${result.error}`);
  } finally {
    if (config.debugProfile) resetWebGpuDebugProfile();
  }
  if (config.selection.backend === 'webgpu') {
    // The settings diagnostic only fills these after a load, and the text
    // copied from a phone may carry just the adapter section — so the
    // harness prints the probe verdict explicitly after every GPU run.
    log(
      `correctness probe: ${describe(result.correctnessProbe)} ` +
        `subgroup_size=${describe(result.correctnessProbeSubgroupSize)} ` +
        `ms=${describe(result.correctnessProbeMs)}`,
    );
  }
  return result;
}

function renderResults(
  configs: readonly HarnessConfig[],
  results: readonly HarnessRunResult[],
): void {
  resultsEl.replaceChildren();
  const table = document.createElement('table');
  const head = document.createElement('tr');
  for (const label of ['Config', 'First token', 'Output', 'TTFT', 'Total', 'Broken?']) {
    const th = document.createElement('th');
    th.textContent = label;
    head.append(th);
  }
  table.append(head);
  const cpu = results.find((r) => r.configId === 'cpu');
  for (const result of results) {
    const config = configs.find((c) => c.id === result.configId);
    const row = document.createElement('tr');
    const comparison = cpu && result.configId !== 'cpu' ? compareRuns(cpu, result) : null;
    const broken =
      result.configId === 'cpu'
        ? cpu
          ? compareRuns(cpu, cpu).cpuLooksBroken
          : null
        : (comparison?.gpuLooksBroken ?? null);
    const cells = [
      `${config?.label ?? result.configId}${result.ok ? '' : ` — FAILED: ${result.error ?? ''}`}`,
      result.firstToken,
      result.output,
      result.ttftMs === null ? 'n/a' : `${result.ttftMs} ms`,
      result.totalMs === null ? 'n/a' : `${result.totalMs} ms`,
      broken === null ? 'n/a' : broken ? 'YES' : 'no',
    ];
    for (const text of cells) {
      const td = document.createElement('td');
      td.textContent = text;
      row.append(td);
    }
    table.append(row);
    if (comparison && cpu) {
      const verdict = verdictForRun(cpu, result);
      const note = document.createElement('p');
      note.textContent =
        `${config?.label ?? result.configId} vs CPU: first token ` +
        `${comparison.firstTokenMatches ? 'matches' : 'DIFFERS'}, output ` +
        `${comparison.outputsIdentical ? 'identical' : 'differs'}, gpu evidence ` +
        `${describe(result.gpuEvidence)}, correctness probe ${describe(result.correctnessProbe)} ` +
        `subgroup_size=${describe(result.correctnessProbeSubgroupSize)} — verdict: ` +
        `${verdict.pass ? 'PASS' : `FAIL (${verdict.reasons.join('; ')})`} ` +
        '(verdict uses the output diff only; the built-in probe is recorded, not trusted).';
      resultsEl.append(note);
    }
  }
  resultsEl.append(table);
}

async function main(): Promise<void> {
  const [capabilities, adapterInfo] = await Promise.all([
    probeCapabilities(),
    probeWebGpuAdapter(),
  ]);
  const diag = adapterInfo.diagnostic;
  const referenceRows = compareAdapterToReference(diag);
  adapterEl.textContent = [
    `webgpu status: ${capabilities.webgpu}`,
    `vendor: ${describe(diag.vendor)}`,
    `architecture: ${describe(diag.architecture)}`,
    `device: ${describe(diag.device)}`,
    `description: ${describe(diag.description)}`,
    `fallback adapter: ${describe(diag.fallbackAdapter)}`,
    `shader-f16: ${describe(diag.shaderF16)}`,
    `subgroups: ${describe(diag.subgroups)} (min ${describe(diag.subgroupMinSize)}, max ${describe(diag.subgroupMaxSize)})`,
    `features: ${diag.features.join(', ') || 'none'}`,
    `maxBufferSize: ${describe(diag.maxBufferSize)}`,
    `maxStorageBufferBindingSize: ${describe(diag.maxStorageBufferBindingSize)}`,
    `maxComputeWorkgroupStorageSize: ${describe(diag.maxComputeWorkgroupStorageSize)}`,
    `maxComputeInvocationsPerWorkgroup: ${describe(diag.maxComputeInvocationsPerWorkgroup)}`,
    `adapter attempts: ${diag.adapterAttempts.join(' | ') || 'none'}`,
    `hardwareConcurrency: ${capabilities.hardwareConcurrency}`,
    `crossOriginIsolated: ${capabilities.crossOriginIsolated}`,
    `gpu fast batches: ${describe(capabilities.gpuFastBatches)}`,
    `android edge runtime: ${isAndroidEdgeRuntime()}`,
    '',
    'POCO F5 Pro reference check (owner diagnostic, 2026-11-21):',
    ...referenceRows.map(
      (row) =>
        `${row.matches ? 'MATCH' : 'DIFF '} ${row.field}: reference=${row.reference} actual=${row.actual}`,
    ),
  ].join('\n');
  const diffs = referenceRows.filter((row) => !row.matches);
  log(
    `adapter reference: ${referenceRows.length - diffs.length}/${referenceRows.length} fields match the POCO F5 Pro diagnostic`,
  );
  for (const row of diffs) {
    log(`DIFF ${row.field}: reference=${row.reference} actual=${row.actual}`);
  }

  const store = await OpfsModelFileStore.open();
  const manager = new ModelManager({ store, createRuntime: createWllamaRuntime });
  const modelStatus = await manager.status();
  if (modelStatus.state !== 'ready' && modelStatus.state !== 'loaded') {
    statusEl.textContent =
      `Model is not downloaded (state: ${modelStatus.state}). ` +
      'Download the model from the options page first, then reopen this page.';
    return;
  }
  statusEl.textContent = 'Model ready. Keep other Sakutto pages closed while the harness runs.';
  runButton.disabled = false;
  runButton.addEventListener('click', () => {
    void (async () => {
      runButton.disabled = true;
      try {
        const probe = {
          webgpu: capabilities.webgpu === 'ready',
          hardwareConcurrency: capabilities.hardwareConcurrency,
          fastBatches: capabilities.gpuFastBatches === true,
          sharedMemory: capabilities.crossOriginIsolated,
        };
        const configs = buildHarnessConfigs(probe, isAndroidEdgeRuntime());
        const prompt = buildTranslationPrompt({
          text: HARNESS_SOURCE_TEXT,
          src: HARNESS_SOURCE_LANGUAGE,
          tgt: HARNESS_TARGET_LANGUAGE,
        });
        log(`source: ${HARNESS_SOURCE_TEXT}`);
        const results: HarnessRunResult[] = [];
        for (const config of configs) {
          const result = await runConfig(manager, config, prompt);
          results.push(result);
          const cpuResult = results.find((r) => r.configId === 'cpu');
          if (cpuResult && config.id !== 'cpu') {
            const verdict = verdictForRun(cpuResult, result);
            log(
              `verdict ${config.id}: ${verdict.pass ? 'PASS' : `FAIL (${verdict.reasons.join('; ')})`}`,
            );
          }
          renderResults(configs, results);
        }
        await manager.unload();
        log('done');
      } finally {
        runButton.disabled = false;
      }
    })();
  });
}

void main();
