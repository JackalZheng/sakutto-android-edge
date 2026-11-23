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
  sameRun,
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
  type LlamaRuntime,
  setRuntimeLogListener,
} from '@/engine/in-browser/runtime';
import {
  buildPerfConfigs,
  crashedPerfResult,
  describeSelection,
  emptyPerfState,
  formatPerfReport,
  HARNESS_MODEL_LAYERS,
  parseGpuOpsLine,
  parseGpuStatLine,
  parsePerfState,
  perfBreakdownQueue,
  type PerfConfig,
  perfDebugString,
  type PerfResult,
  type PerfRunMetrics,
  PERF_STATE_KEY,
  type PerfSweepState,
  perfSweepQueue,
  recordPerfResult,
  resolvePerfConfig,
  summarizeGpuOps,
  summarizeGpuStats,
  withPerfTimeout,
} from '@/engine/debug-perf';
import type { WllamaCompletionTimings } from '@/engine/wllama/module';
import {
  buildHarnessPrompt,
  describeHarnessModel,
  formatModelSize,
  HARNESS_DEFAULT_MODEL_ID,
  HARNESS_MODEL_DIRECTORY,
  HARNESS_MODELS,
  type HarnessModel,
  harnessModel,
  HarnessModelStore,
  type HarnessPrompt,
  isHarnessModelId,
} from '@/engine/debug-models';
import {
  ADRENO_OP_PROFILES,
  ALL_OPS_PROFILES,
  formatNumber,
  isOpDiagnosticContextLine,
  MUL_MAT_GROUP_IDS,
  type OpGroup,
  type OpGroupOutcome,
  opGroupArgs,
  parseOpDiagnosticLines,
  PLAMO3_OP_GROUPS,
  summarizeOpGroup,
  withOpGroupTimeout,
} from '@/engine/debug-op-diagnostics';
import { createWllama } from '@/engine/wllama';
import type { BackendSelection } from '@/engine/in-browser/backend';
import { isEngineError } from '@/engine/errors';
import { DEFAULT_MODEL_ID } from '@/engine/model/catalog';
import { OpfsModelFileStore } from '@/engine/model/file-store';
import { ModelManager } from '@/engine/model/manager';
import { probeCapabilities, probeWebGpuAdapter } from '@/shared/capabilities';
import {
  isWebGpuDebugProfile,
  parseWebGpuLog,
  resetWebGpuDebugProfile,
  setWebGpuDebugProfile,
  setWebGpuDebugTuning,
  type WebGpuDebugProfile,
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
const opsMatrixButton = document.querySelector<HTMLButtonElement>('#run-ops-matrix')!;
const opsAllButtons = [...document.querySelectorAll<HTMLButtonElement>('.run-ops-all')];
const opsProfileEl = document.querySelector<HTMLSelectElement>('#ops-profile')!;
const opsBackendEl = document.querySelector<HTMLSelectElement>('#ops-backend')!;
const opsEl = document.querySelector<HTMLElement>('#ops')!;
const modelSelect = document.querySelector<HTMLSelectElement>('#model')!;
const modelInfoEl = document.querySelector<HTMLElement>('#model-info')!;
const modelStatusEl = document.querySelector<HTMLElement>('#model-status')!;
const modelDownloadButton = document.querySelector<HTMLButtonElement>('#model-download')!;
const modelDeleteButton = document.querySelector<HTMLButtonElement>('#model-delete')!;
const modelCancelButton = document.querySelector<HTMLButtonElement>('#model-cancel')!;
const perfConfigEl = document.querySelector<HTMLSelectElement>('#perf-config')!;
const perfRunButton = document.querySelector<HTMLButtonElement>('#run-perf')!;
const perfSweepButton = document.querySelector<HTMLButtonElement>('#run-perf-sweep')!;
const perfCopyButton = document.querySelector<HTMLButtonElement>('#perf-copy')!;
const perfClearButton = document.querySelector<HTMLButtonElement>('#perf-clear')!;
const perfReportEl = document.querySelector<HTMLElement>('#perf-report')!;

/** sessionStorage key for the model picked in the switcher. */
const MODEL_CHOICE_KEY = 'sakuttoGpuHarnessModel';

/** Loads one harness model for a selection; every config gets a fresh load. */
interface ModelRunner {
  readonly model: HarnessModel;
  load(selection: BackendSelection): Promise<LlamaRuntime>;
  unload(): Promise<void>;
}

/** The production model goes through ModelManager, as before the switcher. */
function productionRunner(manager: ModelManager, model: HarnessModel): ModelRunner {
  return {
    model,
    load: async (selection) => (await manager.ensureLoaded(DEFAULT_MODEL_ID, selection)).runtime,
    unload: () => manager.unload(),
  };
}

/** A harness-only model: read from its own OPFS directory into a fresh runtime. */
function harnessRunner(store: HarnessModelStore, model: HarnessModel): ModelRunner {
  let runtime: LlamaRuntime | null = null;
  const unload = async (): Promise<void> => {
    const current = runtime;
    runtime = null;
    if (current) await current.dispose();
  };
  return {
    model,
    async load(selection) {
      await unload();
      const blob = await store.read(model);
      runtime = await createWllamaRuntime(blob, selection);
      return runtime;
    },
    unload,
  };
}

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
  runner: ModelRunner,
  selection: BackendSelection,
  prompt: HarnessPrompt,
  result: HarnessRunResult,
): Promise<void> {
  const runtime = await runner.load(selection);
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
  await runtime.complete({
    prompt: prompt.prompt,
    stop: prompt.stop,
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
  runner: ModelRunner,
  config: HarnessConfig,
  prompt: HarnessPrompt,
  cpuSelection: BackendSelection,
): Promise<HarnessRunResult> {
  const result: HarnessRunResult = {
    configId: config.id,
    modelId: runner.model.id,
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
  log(`--- ${runner.model.name}: ${config.label} ---`);
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
    await runner.unload();
    try {
      await runLoaded(runner, config.selection, prompt, result);
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
        await runner.unload();
        await runLoaded(runner, cpuSelection, prompt, result);
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

/** One timed completion with the prompt cache off, so every run evaluates the whole prompt. */
async function timedPerfRun(runtime: LlamaRuntime, prompt: HarnessPrompt): Promise<PerfRunMetrics> {
  const chunks: string[] = [];
  let timings: WllamaCompletionTimings | null = null;
  const started = performance.now();
  let firstAt: number | null = null;
  await runtime.complete({
    prompt: prompt.prompt,
    stop: prompt.stop,
    maxTokens: HARNESS_MAX_TOKENS,
    signal: new AbortController().signal,
    cachePrompt: false,
    onToken: (text) => {
      if (firstAt === null) firstAt = performance.now();
      chunks.push(text);
    },
    onTimings: (value) => {
      timings = value;
    },
  });
  const ended = performance.now();
  const t = timings as WllamaCompletionTimings | null;
  return {
    ttftMs: firstAt === null ? null : Math.round(firstAt - started),
    totalMs: ended - started,
    promptTokens: t?.prompt_n ?? null,
    promptMs: t?.prompt_ms ?? null,
    predictedTokens: t?.predicted_n ?? null,
    predictedMs: t?.predicted_ms ?? null,
    output: chunks.join(''),
  };
}

const PERF_NOTE_PATTERN =
  /ggml_webgpu: (adreno|tuning|perf diagnostics|device ready)|offloaded \d+\/\d+ layers|flash_attn|n_ubatch|n_batch/iu;

function newPerfResult(config: PerfConfig, modelId: string): PerfResult {
  return {
    configId: config.id,
    modelId,
    label: config.label,
    selectionText: describeSelection(config.selection),
    debugString: config.selection.backend === 'webgpu' ? perfDebugString(config) : null,
    ok: false,
    error: null,
    loadMs: null,
    gpuEvidence: null,
    runs: [],
    gpuStats: null,
    gpuOps: null,
    notes: [],
  };
}

/** Device-lost / error lines logged during the runs (also kept when the config times out). */
function runtimeProblemLines(): string[] {
  return lastRuntimeLogTail()
    .filter((line) => !/SAKUTTO_GPU(STAT|OPS) /u.test(line))
    .slice(-8)
    .map((line) => `runtime: ${line}`);
}

/**
 * Fills `result` as it goes and calls `progress` after the load and after each run, so a config
 * that hangs or kills the page still reports what it got to (round 3: a timed-out config showed
 * no selection and no debug string).
 */
async function runPerfConfig(
  runner: ModelRunner,
  config: PerfConfig,
  prompt: HarnessPrompt,
  result: PerfResult,
  progress: (result: PerfResult) => void,
): Promise<PerfResult> {
  log(`--- perf ${runner.model.name}: ${config.id} (${config.label}) ---`);
  log(`  ${result.selectionText} debug=${result.debugString ?? 'none'}`);
  const statLines: string[] = [];
  let capturing = false;
  setRuntimeLogListener((line) => {
    if (capturing && /SAKUTTO_GPU(STAT|OPS) /u.test(line)) statLines.push(line);
  });
  try {
    if (config.selection.backend === 'webgpu') {
      setWebGpuDebugProfile(config.debugProfile);
      setWebGpuDebugTuning(config.tuning);
    } else {
      resetWebGpuDebugProfile();
    }
    await runner.unload();
    const loadStarted = performance.now();
    const runtime = await runner.load(config.selection);
    result.loadMs = performance.now() - loadStarted;
    log(`  load: ${Math.round(result.loadMs)} ms (threads ${runtime.threads})`);
    progress(result);
    if (config.selection.backend === 'webgpu') {
      const lines = lastWebGpuLoadLog();
      result.gpuEvidence = gpuEvidence(lines);
      result.notes = lines.filter((line) => PERF_NOTE_PATTERN.test(line)).slice(0, 12);
    }
    for (let run = 0; run < 2; run += 1) {
      capturing = run === 1;
      const metrics = await timedPerfRun(runtime, prompt);
      result.runs.push(metrics);
      progress(result);
      log(
        `  run ${run + 1}: ttft ${metrics.ttftMs ?? 'n/a'} ms, total ${Math.round(metrics.totalMs)} ms, ` +
          `prompt ${metrics.promptTokens ?? 'n/a'} tok / ${Math.round(metrics.promptMs ?? 0)} ms, ` +
          `decode ${metrics.predictedTokens ?? 'n/a'} tok / ${Math.round(metrics.predictedMs ?? 0)} ms`,
      );
    }
    capturing = false;
    result.gpuStats = summarizeGpuStats(
      statLines.map(parseGpuStatLine).filter((line) => line !== null),
    );
    result.gpuOps = summarizeGpuOps(statLines.map(parseGpuOpsLine).filter((line) => line !== null));
    result.notes.push(...runtimeProblemLines());
    result.ok = true;
  } catch (error) {
    result.error = describeErrorChain(error);
    log(`  FAILED: ${result.error}`);
    result.notes.push(...runtimeProblemLines());
  } finally {
    setRuntimeLogListener(null);
    resetWebGpuDebugProfile();
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
  for (const label of ['Model', 'Config', 'First token', 'Output', 'TTFT', 'Total', 'Broken?']) {
    const th = document.createElement('th');
    th.textContent = label;
    head.append(th);
  }
  table.append(head);
  for (const result of results) {
    // Each model is compared with its own CPU run.
    const cpu = results.find((r) => sameRun(r, 'cpu', result.modelId));
    const config = configs.find((c) => c.id === result.configId);
    const modelName = result.modelId
      ? (HARNESS_MODELS.find((m) => m.id === result.modelId)?.name ?? result.modelId)
      : 'n/a';
    const row = document.createElement('tr');
    const comparison = cpu && result.configId !== 'cpu' ? compareRuns(cpu, result) : null;
    const broken =
      result.configId === 'cpu'
        ? cpu
          ? compareRuns(cpu, cpu).cpuLooksBroken
          : null
        : (comparison?.gpuLooksBroken ?? null);
    const cells = [
      modelName,
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
        `${modelName} · ${config?.label ?? result.configId} vs CPU: first token ` +
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
  const allConfigs = buildHarnessConfigs(probe, isAndroidEdgeRuntime());
  const cpuConfig = allConfigs.find((c) => c.id === 'cpu');
  if (!cpuConfig) throw new Error('harness has no CPU reference config');

  // The op diagnostics need no model file, so they are wired up first.
  setupOpDiagnostics();

  // --- model switcher (harness only) -------------------------------------
  const savedState = parseIsolatedState(sessionStorage.getItem(ISOLATED_STATE_KEY));
  const storedChoice = sessionStorage.getItem(MODEL_CHOICE_KEY);
  // A pending isolated queue keeps the model it was started with.
  const savedPerf = parsePerfState(sessionStorage.getItem(PERF_STATE_KEY));
  const modelId =
    savedState && savedState.queue.length > 0 && isHarnessModelId(savedState.modelId)
      ? savedState.modelId
      : savedPerf && savedPerf.queue.length > 0 && isHarnessModelId(savedPerf.modelId)
        ? savedPerf.modelId
        : isHarnessModelId(storedChoice)
          ? storedChoice
          : HARNESS_DEFAULT_MODEL_ID;
  const model = harnessModel(modelId);
  for (const candidate of HARNESS_MODELS) {
    const option = document.createElement('option');
    option.value = candidate.id;
    option.textContent = `${candidate.name} — ${formatModelSize(candidate.size)}`;
    modelSelect.append(option);
  }
  modelSelect.value = model.id;
  modelInfoEl.textContent = describeHarnessModel(model);
  modelSelect.addEventListener('change', () => {
    // The config list and runner depend on the model, so a switch reloads the page.
    sessionStorage.setItem(MODEL_CHOICE_KEY, modelSelect.value);
    location.reload();
  });

  const store = await OpfsModelFileStore.open();
  const manager = new ModelManager({ store, createRuntime: createWllamaRuntime });
  const harnessStore = new HarnessModelStore(
    await OpfsModelFileStore.open(HARNESS_MODEL_DIRECTORY),
  );

  const isReady = async (): Promise<boolean> => {
    if (model.production) {
      const status = await manager.status();
      return status.state === 'ready' || status.state === 'loaded';
    }
    return harnessStore.isDownloaded(model);
  };
  const refreshModelStatus = async (): Promise<boolean> => {
    const ready = await isReady();
    if (model.production) {
      modelStatusEl.textContent = ready
        ? 'Downloaded (managed by the options page).'
        : 'Not downloaded. Download it from the options page.';
      modelDownloadButton.hidden = true;
      modelDeleteButton.hidden = true;
      return ready;
    }
    const partial = ready ? 0 : await harnessStore.partialBytes(model);
    modelStatusEl.textContent = ready
      ? `Downloaded (${formatModelSize(model.size)}).`
      : partial > 0
        ? `Partly downloaded (${formatModelSize(partial)} of ${formatModelSize(model.size)}); Download resumes.`
        : `Not downloaded (${formatModelSize(model.size)}).`;
    modelDownloadButton.disabled = ready;
    modelDeleteButton.disabled = !ready && partial === 0;
    return ready;
  };

  let downloadController: AbortController | null = null;
  modelDownloadButton.addEventListener('click', () => {
    if (
      !confirm(
        `下載 ${model.name}？檔案大小 ${formatModelSize(model.size)}（${model.size.toLocaleString()} bytes）。`,
      )
    ) {
      return;
    }
    void (async () => {
      downloadController = new AbortController();
      modelDownloadButton.disabled = true;
      modelDeleteButton.disabled = true;
      modelSelect.disabled = true;
      modelCancelButton.hidden = false;
      log(`download ${model.id}: ${model.url}`);
      try {
        await harnessStore.downloadFile(model, downloadController.signal, (progress) => {
          modelStatusEl.textContent =
            progress.phase === 'verify'
              ? 'Verifying sha256…'
              : `Downloading ${Math.floor(progress.ratio * 100)}% (${formatModelSize(progress.received)} of ${formatModelSize(progress.total)})`;
        });
        log(`download ${model.id}: done, sha256 verified`);
        location.reload();
      } catch (error) {
        log(`download ${model.id} failed: ${describeErrorChain(error)}`);
        await refreshModelStatus();
      } finally {
        downloadController = null;
        modelSelect.disabled = false;
        modelCancelButton.hidden = true;
      }
    })();
  });
  modelCancelButton.addEventListener('click', () => {
    downloadController?.abort();
  });
  modelDeleteButton.addEventListener('click', () => {
    if (!confirm(`刪除 ${model.name} 的下載檔（${formatModelSize(model.size)}）？`)) return;
    void (async () => {
      await harnessStore.remove(model);
      log(`deleted ${model.id}`);
      location.reload();
    })();
  });

  if (!(await refreshModelStatus())) {
    statusEl.textContent =
      `${model.name} is not downloaded. ` +
      (model.production
        ? 'Download the model from the options page first, then reopen this page. '
        : 'Use Download above. ') +
      'The op diagnostics below still work without it.';
    return;
  }
  statusEl.textContent = `${model.name} ready. Keep other Sakutto pages closed while the harness runs.`;

  const runner = model.production
    ? productionRunner(manager, model)
    : harnessRunner(harnessStore, model);
  // Layer-index tensor overrides follow the PLaMo 3 layout, so other models skip them.
  const configs = allConfigs.filter((c) => !c.plamoOnly || model.format === 'plamo3');
  const prompt = buildHarnessPrompt(model, {
    text: HARNESS_SOURCE_TEXT,
    src: HARNESS_SOURCE_LANGUAGE,
    tgt: HARNESS_TARGET_LANGUAGE,
  });

  const buttons = (): HTMLButtonElement[] => [
    runButton,
    runIsolatedButton,
    modelDownloadButton,
    modelDeleteButton,
    clearButton,
    opsButton,
    opsMatrixButton,
    ...opsAllButtons,
    perfRunButton,
    perfSweepButton,
    perfClearButton,
    ...configsEl.querySelectorAll<HTMLButtonElement>('button'),
  ];
  const setBusy = (busy: boolean): void => {
    for (const button of buttons()) button.disabled = busy;
    modelSelect.disabled = busy;
    if (!busy) void refreshModelStatus();
  };

  const logVerdict = (results: readonly HarnessRunResult[], config: HarnessConfig): void => {
    const cpuResult = results.find((r) => sameRun(r, 'cpu', model.id));
    const result = results.find((r) => sameRun(r, config.id, model.id));
    if (cpuResult && result && config.id !== 'cpu') {
      const verdict = verdictForRun(cpuResult, result);
      log(
        `verdict ${config.id}: ${verdict.pass ? 'PASS' : `FAIL (${verdict.reasons.join('; ')})`}`,
      );
    }
  };

  // --- GPU perf breakdown / sweep: one config per page load -----------------
  const perfConfigs = buildPerfConfigs(
    probe,
    isAndroidEdgeRuntime(),
    HARNESS_MODEL_LAYERS[model.id] ?? null,
  );
  for (const config of perfConfigs) {
    const option = document.createElement('option');
    option.value = config.id;
    option.textContent = `${config.id}: ${config.label}`;
    perfConfigEl.append(option);
  }
  perfConfigEl.value = perfConfigs.some((c) => c.id === 'gpu-mv') ? 'gpu-mv' : 'cpu';
  const readPerf = (): PerfSweepState =>
    parsePerfState(sessionStorage.getItem(PERF_STATE_KEY)) ?? emptyPerfState(model.id);
  const writePerf = (state: PerfSweepState): void => {
    sessionStorage.setItem(PERF_STATE_KEY, JSON.stringify(state));
  };
  const perfHeader = (): string[] => [
    `model: ${model.name} (${model.id})`,
    `date: ${new Date().toISOString()}`,
    `adapter: vendor=${describe(diag.vendor)} architecture=${describe(diag.architecture)} description=${describe(diag.description)}`,
    `subgroups=${describe(diag.subgroups)} shader-f16=${describe(diag.shaderF16)} maxStorageBufferBindingSize=${describe(diag.maxStorageBufferBindingSize)}`,
    `hardwareConcurrency=${capabilities.hardwareConcurrency} crossOriginIsolated=${capabilities.crossOriginIsolated} android=${isAndroidEdgeRuntime()}`,
    `prompt: harness paragraph (${HARNESS_SOURCE_TEXT.length} chars), max ${HARNESS_MAX_TOKENS} tokens, greedy, prompt cache off`,
  ];
  const renderPerf = (): void => {
    const state = readPerf();
    perfReportEl.textContent =
      state.results.length > 0 ? formatPerfReport(perfHeader(), state.results) : '';
    perfCopyButton.disabled = state.results.length === 0;
  };
  const startPerf = (queue: readonly string[]): void => {
    const state = readPerf();
    writePerf({
      modelId: model.id,
      queue: [...queue],
      running: null,
      partial: null,
      results: state.modelId === model.id ? state.results : [],
    });
    location.reload();
  };
  perfRunButton.addEventListener('click', () => {
    startPerf(perfBreakdownQueue(perfConfigs, perfConfigEl.value));
  });
  perfSweepButton.addEventListener('click', () => {
    startPerf(perfSweepQueue(perfConfigs));
  });
  perfClearButton.addEventListener('click', () => {
    sessionStorage.removeItem(PERF_STATE_KEY);
    renderPerf();
    log('saved perf results cleared');
  });
  perfCopyButton.addEventListener('click', () => {
    const text = perfReportEl.textContent ?? '';
    void navigator.clipboard.writeText(text).then(
      () => log('perf report copied'),
      () => {
        const range = document.createRange();
        range.selectNodeContents(perfReportEl);
        const selection = getSelection();
        selection?.removeAllRanges();
        selection?.addRange(range);
        log('clipboard blocked: the report is selected, copy it from the page');
      },
    );
  });
  renderPerf();
  const perfState = readPerf();
  if (perfState.running !== null) {
    // The page died during that config (crash, out of memory, device loss, or the timeout reload).
    writePerf(
      recordPerfResult(
        perfState,
        perfState.partial?.configId === perfState.running
          ? {
              ...perfState.partial,
              ok: false,
              error: 'page reloaded or crashed mid-run',
            }
          : crashedPerfResult(
              perfState.running,
              perfState.modelId,
              'page reloaded or crashed mid-run',
            ),
      ),
    );
    renderPerf();
  }
  const nextPerf = readPerf().queue[0];
  if (nextPerf !== undefined) {
    setBusy(true);
    const config = resolvePerfConfig(perfConfigs, nextPerf);
    const current = readPerf();
    if (!config) {
      writePerf({ ...current, queue: current.queue.slice(1) });
    } else {
      statusEl.textContent = `GPU perf: ${config.id} (${current.queue.length} left in the queue)…`;
      const partial = newPerfResult(config, model.id);
      writePerf({ ...current, running: config.id, partial });
      let result: PerfResult;
      try {
        result = await withPerfTimeout(
          runPerfConfig(runner, config, prompt, partial, (value) => {
            writePerf({ ...readPerf(), partial: value });
          }),
        );
      } catch (error) {
        // Timed out: keep what the config got to, plus any device-lost lines.
        result = {
          ...partial,
          ok: false,
          error: describeErrorChain(error),
          notes: [...partial.notes, ...runtimeProblemLines()],
        };
      }
      writePerf(recordPerfResult(readPerf(), result));
      renderPerf();
      if (result.ok) await runner.unload().catch(() => undefined);
    }
    const remaining = readPerf().queue.length;
    statusEl.textContent =
      remaining > 0
        ? `GPU perf: reloading for the next config (${remaining} left)…`
        : 'GPU perf done. Copy the report below.';
    if (remaining > 0) {
      // A fresh page per config: a timed-out or crashed run leaves its worker behind.
      await new Promise((resolve) => setTimeout(resolve, ISOLATED_RELOAD_DELAY_MS));
      location.reload();
      return;
    }
    log('GPU perf done');
  }

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
      model.id,
    );
    writeState({ ...state, queue, modelId: model.id });
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
    renderResults(allConfigs, saved.results);
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
      log(
        `=== isolated run (fresh page, fresh worker and WebGPU device): ${model.id} / ${config.id} ===`,
      );
      const result = await runConfig(runner, config, prompt, cpuConfig.selection);
      await runner.unload().catch(() => undefined);
      const results = [
        ...saved.results.filter((r) => !sameRun(r, result.configId, result.modelId)),
        result,
      ];
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
    renderResults(allConfigs, readState().results);
  }
  setBusy(false);

  // --- same-page mode (the original Stage 1/2 harness) --------------------
  runButton.addEventListener('click', () => {
    void (async () => {
      setBusy(true);
      try {
        log(`model: ${model.id} (${model.architecture})`);
        log(`source: ${HARNESS_SOURCE_TEXT}`);
        const results: HarnessRunResult[] = [];
        for (const config of configs) {
          const result = await runConfig(runner, config, prompt, cpuConfig.selection);
          results.push(result);
          logVerdict(results, config);
          renderResults(allConfigs, results);
        }
        await runner.unload();
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
  const opButtons = [opsButton, opsMatrixButton, ...opsAllButtons];
  const setOpsBusy = (busy: boolean): void => {
    for (const button of opButtons) button.disabled = busy;
  };
  setOpsBusy(false);
  const run = async (
    plan: ReadonlyArray<{ group: OpGroup; profile: WebGpuDebugProfile | null }>,
    title: string,
  ): Promise<void> => {
    setOpsBusy(true);
    opsEl.replaceChildren();
    const backend = opsBackendEl.value === 'CPU' ? 'CPU' : undefined;
    log(`=== op diagnostics (${title}, backend: ${backend ?? 'WebGPU'}) ===`);
    const outcomes: OpGroupOutcome[] = [];
    try {
      // Every group runs, whatever the previous one did: runOpGroup records a crash or a
      // timeout as that group's outcome instead of throwing.
      for (const step of plan) {
        outcomes.push(await runOpGroup(step.group, step.profile, backend));
        renderOpOutcomes(outcomes);
      }
      log('op diagnostics done');
    } catch (error) {
      reportFatal(error);
    } finally {
      setOpsBusy(false);
    }
  };
  opsButton.addEventListener('click', () => {
    const profile = isWebGpuDebugProfile(opsProfileEl.value) ? opsProfileEl.value : null;
    void run(
      PLAMO3_OP_GROUPS.map((group) => ({ group, profile })),
      `profile: ${profile ?? 'default'}`,
    );
  });
  opsMatrixButton.addEventListener('click', () => {
    const groups = PLAMO3_OP_GROUPS.filter((group) => MUL_MAT_GROUP_IDS.includes(group.id));
    const plan = ADRENO_OP_PROFILES.flatMap((profile) =>
      groups.map((group) => ({
        group,
        profile: isWebGpuDebugProfile(profile) ? profile : null,
      })),
    );
    void run(plan, 'MUL_MAT × Adreno profiles');
  });
  for (const button of opsAllButtons) {
    const profile = button.dataset.profile ?? '';
    if (!isWebGpuDebugProfile(profile) || !ALL_OPS_PROFILES.includes(profile)) continue;
    button.addEventListener('click', () => {
      void run(
        PLAMO3_OP_GROUPS.map((group) => ({ group, profile })),
        `all ops × ${profile}`,
      );
    });
  }
}

async function runOpGroup(
  group: OpGroup,
  profile: WebGpuDebugProfile | null,
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
    const outcome = await withOpGroupTimeout(wllama.testBackendOps(opGroupArgs(group, backend)));
    retcode = outcome.retcode;
  } catch (caught) {
    error = describeErrorChain(caught);
  } finally {
    if (profile) resetWebGpuDebugProfile();
  }
  const rows = parseOpDiagnosticLines(lines);
  const outcome: OpGroupOutcome = {
    group,
    rows,
    error,
    retcode,
    ...(profile ? { profile } : {}),
  };
  const summary = summarizeOpGroup(outcome);
  log(
    `op ${group.id}${profile ? ` [${profile}]` : ''}: ${summary.verdict} ok=${summary.ok} fail=${summary.fail} ` +
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
      `${outcome.group.label}${outcome.profile ? ` [${outcome.profile}]` : ''}`,
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
