import {
  buildHarnessConfigs,
  compareAdapterToReference,
  compareRuns,
  describeErrorChain,
  emptyIsolatedState,
  firstStreamedToken,
  ISOLATED_RELOAD_DELAY_MS,
  ISOLATED_STATE_KEY,
  type IsolatedRunState,
  parseIsolatedState,
  planIsolatedQueue,
  recordIsolatedResult,
  HARNESS_MAX_TOKENS,
  HARNESS_SOURCE_LANGUAGE,
  HARNESS_SOURCE_TEXT,
  HARNESS_TARGET_LANGUAGE,
  type HarnessConfig,
  type HarnessRunResult,
  verdictForRun,
} from '@/engine/debug-harness';
import {
  createWllamaRuntime,
  gpuEvidence,
  lastRuntimeLogTail,
  lastWebGpuLoadLog,
} from '@/engine/in-browser/runtime';
import {
  formatNumber,
  isOpDiagnosticContextLine,
  type OpGroupOutcome,
  opGroupArgs,
  parseOpDiagnosticLines,
  PLAMO3_OP_GROUPS,
  summarizeOpGroup,
} from '@/engine/debug-op-diagnostics';
import { createWllama } from '@/engine/wllama';
import type { BackendSelection } from '@/engine/in-browser/backend';
import { isEngineError } from '@/engine/errors';
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
const runIsolatedButton = document.querySelector<HTMLButtonElement>('#run-isolated')!;
const clearButton = document.querySelector<HTMLButtonElement>('#clear-saved')!;
const configsEl = document.querySelector<HTMLElement>('#configs')!;
const opsButton = document.querySelector<HTMLButtonElement>('#run-ops')!;
const opsProfileEl = document.querySelector<HTMLSelectElement>('#ops-profile')!;
const opsBackendEl = document.querySelector<HTMLSelectElement>('#ops-backend')!;
const opsEl = document.querySelector<HTMLElement>('#ops')!;

function log(line: string): void {
  logEl.textContent += `${line}\n`;
}

function describe(value: unknown): string {
  return value === null || value === undefined ? 'n/a' : String(value);
}

/**
 * The harness runs on a phone with no devtools, so a startup or run failure
 * has to be visible on the page itself — otherwise the page just sits at
 * "Probing…" with the failure only in the console.
 */
function reportFatal(error: unknown): void {
  const detail =
    error instanceof Error
      ? `${error.name}: ${error.message}${error.stack ? `\n${error.stack}` : ''}`
      : String(error);
  statusEl.textContent = `Harness failed: ${detail.split('\n')[0] ?? detail}`;
  log(`FATAL: ${detail}`);
}

/** Load one selection and run the fixed prompt through it, filling `result`. */
async function runLoaded(
  manager: ModelManager,
  selection: BackendSelection,
  prompt: string,
  result: HarnessRunResult,
): Promise<void> {
  const loaded = await manager.ensureLoaded(DEFAULT_MODEL_ID, selection);
  if (selection.backend === 'webgpu') {
    const lines = lastWebGpuLoadLog();
    result.gpuEvidence = gpuEvidence(lines);
    const parsed = parseWebGpuLog(lines);
    result.correctnessProbe = parsed.correctnessProbe;
    result.correctnessProbeSubgroupSize = parsed.correctnessProbeSubgroupSize;
    result.correctnessProbeMs = parsed.correctnessProbeMs;
    log(`gpu evidence: ${result.gpuEvidence}`);
    for (const line of lines) {
      if (/ggml_webgpu|offload|sakutto:|flash_attn/u.test(line)) log(`  ${line}`);
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
}

async function runConfig(
  manager: ModelManager,
  config: HarnessConfig,
  prompt: string,
  cpuSelection: BackendSelection,
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
    fellBackToCpu: false,
  };
  log(`--- ${config.label} ---`);
  log(
    `  selection: backend=${config.selection.backend} n_gpu_layers=${config.selection.nGpuLayers} ` +
      `n_parallel=${config.selection.nParallel} n_ctx=${config.selection.nCtx} ` +
      `n_batch=${describe(config.selection.nBatch)} profile=${config.debugProfile ?? 'none'}` +
      `${config.selection.debugLoad ? ` debugLoad=${JSON.stringify(config.selection.debugLoad)}` : ''}`,
  );
  try {
    if (config.debugProfile) setWebGpuDebugProfile(config.debugProfile);
    else resetWebGpuDebugProfile();
    // Force a fresh load per config: the debug profile is read when the
    // wllama worker is created, and two configs can share one selection.
    await manager.unload();
    try {
      await runLoaded(manager, config.selection, prompt, result);
    } catch (error) {
      if (config.fallbackToCpu && isEngineError(error) && error.code === 'gpu-unavailable') {
        // The no-f16 profile refuses the device on purpose: record the GPU
        // evidence from the failed load, then run the same prompt on the CPU
        // so the config reports whether the model still translates without
        // shader-f16 instead of a load error.
        const lines = lastWebGpuLoadLog();
        result.gpuEvidence = gpuEvidence(lines);
        log(`gpu evidence: ${result.gpuEvidence} — WebGPU refused the device, falling back to CPU`);
        for (const line of lines) {
          if (line.includes('ggml_webgpu') || line.includes('offload')) log(`  ${line}`);
        }
        await manager.unload();
        await runLoaded(manager, cpuSelection, prompt, result);
        result.fellBackToCpu = true;
      } else {
        throw error;
      }
    }
  } catch (error) {
    result.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    log(`FAILED: ${result.error}`);
    log(`  error chain: ${describeErrorChain(error)}`);
    // Device-lost reason / device error text logged by ggml-webgpu during the run.
    for (const line of lastRuntimeLogTail()) log(`  runtime: ${line}`);
    for (const line of lastWebGpuLoadLog()) {
      if (/lost|abort|error|fail|sakutto:/iu.test(line)) log(`  load: ${line}`);
    }
  } finally {
    if (config.debugProfile) resetWebGpuDebugProfile();
  }
  if (config.selection.backend === 'webgpu') {
    // The settings diagnostic only fills these after a load, and the text
    // copied from a phone may carry just the adapter section — so the
    // harness prints the probe verdict explicitly after every GPU run.
    log(
      result.fellBackToCpu
        ? 'correctness probe: n/a (ran on CPU fallback)'
        : `correctness probe: ${describe(result.correctnessProbe)} ` +
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
      `${config?.label ?? result.configId}${result.fellBackToCpu ? ' (CPU fallback)' : ''}${result.ok ? '' : ` — FAILED: ${result.error ?? ''}`}`,
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
        `${describe(result.gpuEvidence)}${result.fellBackToCpu ? ' (ran on CPU fallback after WebGPU refused the device)' : ''}, correctness probe ${describe(result.correctnessProbe)} ` +
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

  // `?forceWebgpuConfigs=1` lists the WebGPU configs even when the page's own probe says
  // WebGPU is unavailable (the worker may still get a device, or the run shows why not).
  const forceGpu = new URLSearchParams(location.search).get('forceWebgpuConfigs') === '1';
  const probe = {
    webgpu: forceGpu || capabilities.webgpu === 'ready',
    hardwareConcurrency: capabilities.hardwareConcurrency,
    fastBatches: capabilities.gpuFastBatches === true,
    sharedMemory: capabilities.crossOriginIsolated,
  };
  const configs = buildHarnessConfigs(probe, isAndroidEdgeRuntime());
  const cpuConfig = configs.find((c) => c.id === 'cpu');
  if (!cpuConfig) throw new Error('harness has no CPU reference config');

  // The op diagnostics need no model file, so they are wired up first.
  setupOpDiagnostics();

  const store = await OpfsModelFileStore.open();
  const manager = new ModelManager({ store, createRuntime: createWllamaRuntime });
  const modelStatus = await manager.status();
  if (modelStatus.state !== 'ready' && modelStatus.state !== 'loaded') {
    statusEl.textContent =
      `Model is not downloaded (state: ${modelStatus.state}). ` +
      'Download the model from the options page first, then reopen this page. ' +
      'The op diagnostics below still work without it.';
    return;
  }
  statusEl.textContent = 'Model ready. Keep other Sakutto pages closed while the harness runs.';

  const prompt = buildTranslationPrompt({
    text: HARNESS_SOURCE_TEXT,
    src: HARNESS_SOURCE_LANGUAGE,
    tgt: HARNESS_TARGET_LANGUAGE,
  });

  const buttons = (): HTMLButtonElement[] => [
    runButton,
    runIsolatedButton,
    clearButton,
    opsButton,
    ...configsEl.querySelectorAll<HTMLButtonElement>('button'),
  ];
  const setBusy = (busy: boolean): void => {
    for (const button of buttons()) button.disabled = busy;
  };

  const logVerdict = (results: readonly HarnessRunResult[], config: HarnessConfig): void => {
    const cpuResult = results.find((r) => r.configId === 'cpu');
    const result = results.find((r) => r.configId === config.id);
    if (cpuResult && result && config.id !== 'cpu') {
      const verdict = verdictForRun(cpuResult, result);
      log(
        `verdict ${config.id}: ${verdict.pass ? 'PASS' : `FAIL (${verdict.reasons.join('; ')})`}`,
      );
    }
  };

  // --- isolated mode: one config per page load ---------------------------
  const readState = (): IsolatedRunState =>
    parseIsolatedState(sessionStorage.getItem(ISOLATED_STATE_KEY)) ?? emptyIsolatedState();
  const writeState = (state: IsolatedRunState): void => {
    sessionStorage.setItem(ISOLATED_STATE_KEY, JSON.stringify(state));
  };
  const startIsolated = (requested: readonly string[]): void => {
    const state = readState();
    const queue = planIsolatedQueue(
      configs.map((c) => c.id),
      requested,
      state,
    );
    writeState({ ...state, queue });
    location.reload();
  };

  for (const config of configs) {
    const row = document.createElement('div');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'small';
    button.textContent = 'Run alone';
    button.addEventListener('click', () => {
      startIsolated([config.id]);
    });
    const label = document.createElement('span');
    label.textContent = ` ${config.id}: ${config.label}`;
    row.append(button, label);
    configsEl.append(row);
  }

  runIsolatedButton.addEventListener('click', () => {
    startIsolated(configs.map((c) => c.id));
  });
  clearButton.addEventListener('click', () => {
    sessionStorage.removeItem(ISOLATED_STATE_KEY);
    resultsEl.replaceChildren();
    logEl.textContent = '';
    log('saved isolated results cleared');
  });

  const saved = readState();
  if (saved.results.length > 0 || saved.queue.length > 0) {
    logEl.textContent = `${saved.log}${logEl.textContent ?? ''}`;
    renderResults(configs, saved.results);
  }
  const next = saved.queue[0];
  if (next !== undefined) {
    const config = configs.find((c) => c.id === next);
    setBusy(true);
    const before = logEl.textContent;
    if (!config) {
      writeState({ ...saved, queue: saved.queue.slice(1) });
    } else {
      statusEl.textContent = `Isolated run: ${config.id} (${saved.queue.length} left in the queue)…`;
      log(`=== isolated run (fresh page, fresh worker and WebGPU device): ${config.id} ===`);
      const result = await runConfig(manager, config, prompt, cpuConfig.selection);
      await manager.unload().catch(() => undefined);
      const results = [...saved.results.filter((r) => r.configId !== result.configId), result];
      logVerdict(results, config);
      const chunk = (logEl.textContent ?? '').slice((before ?? '').length);
      writeState(recordIsolatedResult(saved, result, chunk));
    }
    const remaining = readState().queue.length;
    if (remaining > 0) {
      statusEl.textContent = `Isolated run: reloading for the next config (${remaining} left)…`;
      await new Promise((resolve) => setTimeout(resolve, ISOLATED_RELOAD_DELAY_MS));
      location.reload();
      return;
    }
    log('isolated run done');
    statusEl.textContent = 'Isolated run done. Results are kept until "Clear saved results".';
    renderResults(configs, readState().results);
  }
  setBusy(false);

  // --- same-page mode (the original Stage 1/2 harness) --------------------
  runButton.addEventListener('click', () => {
    void (async () => {
      setBusy(true);
      try {
        log(`source: ${HARNESS_SOURCE_TEXT}`);
        const results: HarnessRunResult[] = [];
        for (const config of configs) {
          const result = await runConfig(manager, config, prompt, cpuConfig.selection);
          results.push(result);
          logVerdict(results, config);
          renderResults(configs, results);
        }
        await manager.unload();
        log('done');
      } catch (error) {
        reportFatal(error);
      } finally {
        setBusy(false);
      }
    })();
  });
}

/**
 * Op-level WebGPU vs CPU comparison (test-backend-ops with the PLaMo 3 op
 * suite). Each op group runs in its own throwaway wllama worker, so a group
 * that loses the WebGPU device only takes itself down.
 */
function setupOpDiagnostics(): void {
  opsButton.disabled = false;
  opsButton.addEventListener('click', () => {
    void (async () => {
      opsButton.disabled = true;
      opsEl.replaceChildren();
      const profile = opsProfileEl.value === 'no-subgroups' ? 'no-subgroups' : null;
      const backend = opsBackendEl.value === 'CPU' ? 'CPU' : undefined;
      log(
        `=== op diagnostics (profile: ${profile ?? 'default'}, backend: ${backend ?? 'WebGPU'}) ===`,
      );
      const outcomes: OpGroupOutcome[] = [];
      try {
        for (const group of PLAMO3_OP_GROUPS) {
          const outcome = await runOpGroup(group, profile, backend);
          outcomes.push(outcome);
          renderOpOutcomes(outcomes);
        }
        log('op diagnostics done');
      } catch (error) {
        reportFatal(error);
      } finally {
        opsButton.disabled = false;
      }
    })();
  });
}

async function runOpGroup(
  group: (typeof PLAMO3_OP_GROUPS)[number],
  profile: 'no-subgroups' | null,
  backend: string | undefined,
): Promise<OpGroupOutcome> {
  const lines: string[] = [];
  const note = (...data: unknown[]): void => {
    if (lines.length < 20_000) lines.push(data.map((value) => String(value)).join(' '));
  };
  const started = performance.now();
  let error: string | null = null;
  let retcode: number | null = null;
  if (profile) setWebGpuDebugProfile(profile);
  else resetWebGpuDebugProfile();
  try {
    const wllama = await createWllama({
      suppressNativeLog: false,
      logger: { debug: note, log: note, warn: note, error: note },
    });
    if (typeof wllama.testBackendOps !== 'function') {
      throw new Error('this wllama build has no testBackendOps (patch 0016 missing)');
    }
    const outcome = await wllama.testBackendOps(opGroupArgs(group, backend));
    retcode = outcome.retcode;
  } catch (caught) {
    error = describeErrorChain(caught);
  } finally {
    if (profile) resetWebGpuDebugProfile();
  }
  const rows = parseOpDiagnosticLines(lines);
  const outcome: OpGroupOutcome = { group, rows, error, retcode };
  const summary = summarizeOpGroup(outcome);
  log(
    `op ${group.id}: ${summary.verdict} ok=${summary.ok} fail=${summary.fail} ` +
      `unsupported=${summary.unsupported} incomplete=${summary.incomplete} ` +
      `retcode=${describe(retcode)} ${Math.round(performance.now() - started)}ms` +
      `${error ? ` error=${error}` : ''}`,
  );
  for (const row of rows) {
    if (row.status !== 'OK') {
      const w = row.worst;
      log(
        `  ${row.status} ${row.op}(${row.vars})` +
          (w
            ? ` nmse=${formatNumber(w.nmse)} thr=${formatNumber(w.threshold)} maxabs=${formatNumber(w.maxAbs)} nan=${w.nanGpu}/${w.nanCpu} inf=${w.infGpu}/${w.infCpu}`
            : ''),
      );
    }
  }
  const context = lines.filter(isOpDiagnosticContextLine);
  for (const line of context.slice(0, 12)) log(`  ${line}`);
  return outcome;
}

function renderOpOutcomes(outcomes: readonly OpGroupOutcome[]): void {
  opsEl.replaceChildren();
  const summaryTable = document.createElement('table');
  const head = document.createElement('tr');
  for (const label of [
    'Op group',
    'Verdict',
    'OK',
    'FAIL',
    'n/s',
    'Incomplete',
    'Worst NMSE',
    'Max abs',
  ]) {
    const th = document.createElement('th');
    th.textContent = label;
    head.append(th);
  }
  summaryTable.append(head);
  for (const outcome of outcomes) {
    const summary = summarizeOpGroup(outcome);
    let worstNmse = 0;
    let worstAbs = 0;
    for (const row of outcome.rows) {
      if (row.worst) {
        worstNmse = Number.isNaN(row.worst.nmse) ? worstNmse : Math.max(worstNmse, row.worst.nmse);
        worstAbs = Number.isNaN(row.worst.maxAbs) ? worstAbs : Math.max(worstAbs, row.worst.maxAbs);
      }
    }
    const row = document.createElement('tr');
    for (const text of [
      outcome.group.label,
      `${summary.verdict}${outcome.error ? ` (${outcome.error})` : ''}`,
      String(summary.ok),
      String(summary.fail),
      String(summary.unsupported),
      String(summary.incomplete),
      formatNumber(worstNmse),
      formatNumber(worstAbs),
    ]) {
      const td = document.createElement('td');
      td.textContent = text;
      row.append(td);
    }
    summaryTable.append(row);
  }
  opsEl.append(summaryTable);

  const detail = document.createElement('table');
  const detailHead = document.createElement('tr');
  for (const label of [
    'Case',
    'Status',
    'NMSE',
    'Threshold',
    'Max abs',
    'NaN gpu/cpu',
    'Inf gpu/cpu',
  ]) {
    const th = document.createElement('th');
    th.textContent = label;
    detailHead.append(th);
  }
  detail.append(detailHead);
  // Every non-OK case, plus the closest-to-threshold OK case of each group, so a phone
  // screenshot stays readable; the full list is in the log on failure.
  for (const outcome of outcomes) {
    const bad = outcome.rows.filter((row) => row.status !== 'OK');
    const closest = outcome.rows
      .filter((row) => row.status === 'OK' && row.worst)
      .reduce<(typeof outcome.rows)[number] | null>((best, row) => {
        const ratio = (r: typeof row): number =>
          r.worst && r.worst.threshold > 0 ? r.worst.nmse / r.worst.threshold : 0;
        return best === null || ratio(row) > ratio(best) ? row : best;
      }, null);
    const shown = closest ? [...bad, closest] : bad;
    for (const caseRow of shown) {
      const w = caseRow.worst;
      const tr = document.createElement('tr');
      if (caseRow.status !== 'OK') tr.className = 'bad';
      for (const text of [
        `${caseRow.op}(${caseRow.vars})`,
        caseRow.status,
        w ? formatNumber(w.nmse) : 'n/a',
        w ? formatNumber(w.threshold) : 'n/a',
        w ? formatNumber(w.maxAbs) : 'n/a',
        w ? `${w.nanGpu}/${w.nanCpu}` : 'n/a',
        w ? `${w.infGpu}/${w.infCpu}` : 'n/a',
      ]) {
        const td = document.createElement('td');
        td.textContent = text;
        tr.append(td);
      }
      detail.append(tr);
    }
  }
  opsEl.append(detail);
}

void main().catch(reportFatal);
