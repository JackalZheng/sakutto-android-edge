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
import {
  DEFAULT_HARNESS_MODEL_ID,
  HARNESS_MODELS,
  harnessModelName,
} from '@/engine/debug-harness-models';
import { createWllamaRuntime, gpuEvidence, lastWebGpuLoadLog } from '@/engine/in-browser/runtime';
import type { BackendSelection } from '@/engine/in-browser/backend';
import { isEngineError } from '@/engine/errors';
import { getModel, type ModelId } from '@/engine/model/catalog';
import { OpfsModelFileStore } from '@/engine/model/file-store';
import { ModelManager } from '@/engine/model/manager';
import { buildHarnessModelPrompt } from '@/engine/prompt/harness-models';
import type { ModelStatus } from '@/shared/engine-status';
import { probeCapabilities, probeWebGpuAdapter } from '@/shared/capabilities';
import {
  parseWebGpuLog,
  resetWebGpuDebugProfile,
  setWebGpuDebugProfile,
} from '@/shared/gpu-diagnostic';
import { isAndroidEdgeRuntime } from '@/shared/mobile';
import { formatByteSize } from '@/ui/format';

/**
 * Stage 1 debug harness page. Deliberately standalone: it drives ModelManager
 * directly instead of going through the offscreen engine host, so the product
 * fallback logic (Auto → CPU on Android, garbage guard) cannot mask what the
 * GPU actually does. Selections are built with `selectionForTier`, bypassing
 * `resolveSelection` for the same reason.
 *
 * Model switcher: the menu above the Run button lists five models (sorted by
 * file size ascending) with their size, license and one line of Chinese
 * description. Each model downloads into its own `models/<modelId>/` cache
 * through the shared ModelManager — separate keys, individually deletable —
 * and the run uses the selected model's own chat prompt and turn-end stop
 * token, greedy decoding, and the same `HARNESS_MAX_TOKENS` budget as the
 * PLaMo reference. The 8 harness configs are unchanged; the switcher selects
 * the model, the configs still apply. Debug-page only: this page never
 * touches the production translation flow or the settings page's default
 * model.
 */

const statusEl = document.querySelector<HTMLElement>('#status')!;
const adapterEl = document.querySelector<HTMLElement>('#adapter')!;
const logEl = document.querySelector<HTMLElement>('#log')!;
const resultsEl = document.querySelector<HTMLElement>('#results')!;
const modelsEl = document.querySelector<HTMLElement>('#models')!;
const runButton = document.querySelector<HTMLButtonElement>('#run')!;

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

function describeStatus(status: ModelStatus | undefined): string {
  if (!status) return '狀態未知';
  switch (status.state) {
    case 'ready':
    case 'loaded':
      return '已下載';
    case 'partial':
      return `部分下載（${formatByteSize(status.received)} / ${formatByteSize(status.total)}）`;
    case 'downloading':
      return `下載中 ${formatByteSize(status.received)} / ${formatByteSize(status.total)}`;
    case 'verifying':
      return '驗證中…';
    case 'loading':
      return '載入中…';
    case 'error':
      return `下載失敗：${status.error ?? '未知錯誤'}`;
    case 'absent':
      return '未下載';
  }
}

/** Load one selection and run the fixed prompt through it, filling `result`. */
async function runLoaded(
  manager: ModelManager,
  modelId: ModelId,
  selection: BackendSelection,
  prompt: string,
  stop: readonly string[],
  result: HarnessRunResult,
): Promise<void> {
  const loaded = await manager.ensureLoaded(modelId, selection);
  if (selection.backend === 'webgpu') {
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
    stop,
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
  modelId: ModelId,
  config: HarnessConfig,
  prompt: string,
  stop: readonly string[],
  cpuSelection: BackendSelection,
): Promise<HarnessRunResult> {
  const result: HarnessRunResult = {
    configId: config.id,
    modelId,
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
  try {
    if (config.debugProfile) setWebGpuDebugProfile(config.debugProfile);
    else resetWebGpuDebugProfile();
    // Force a fresh load per config: the debug profile is read when the
    // wllama worker is created, and two configs can share one selection.
    await manager.unload();
    try {
      await runLoaded(manager, modelId, config.selection, prompt, stop, result);
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
        await runLoaded(manager, modelId, cpuSelection, prompt, stop, result);
        result.fellBackToCpu = true;
      } else {
        throw error;
      }
    }
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
  for (const label of ['模型', 'Config', 'First token', 'Output', 'TTFT', 'Total', 'Broken?']) {
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
      harnessModelName(result.modelId),
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

  const store = await OpfsModelFileStore.open();
  const manager = new ModelManager({ store, createRuntime: createWllamaRuntime });

  // Model switcher state. The selected model is the one the Run button
  // tests; it defaults to the PLaMo control group on every page load.
  let selectedId: ModelId = DEFAULT_HARNESS_MODEL_ID;
  let running = false;
  let statuses: Partial<Record<ModelId, ModelStatus>> = {};

  const refreshStatuses = async (): Promise<void> => {
    const next: Partial<Record<ModelId, ModelStatus>> = {};
    for (const model of HARNESS_MODELS) {
      next[model.id] = await manager.status(model.id);
    }
    statuses = next;
  };

  const updateRunButton = (): void => {
    const status = statuses[selectedId];
    const ready = status?.state === 'ready' || status?.state === 'loaded';
    runButton.disabled = running || !ready;
    if (running) {
      statusEl.textContent = `Running ${harnessModelName(selectedId)}… Keep other Sakutto pages closed.`;
    } else if (ready) {
      statusEl.textContent = `已選 ${harnessModelName(selectedId)}（${formatByteSize(getModel(selectedId).size)}）。按「Run harness」開始測試。`;
    } else {
      statusEl.textContent = `已選 ${harnessModelName(selectedId)}，尚未下載。請先在上方下載，下載前會先顯示大小確認。`;
    }
  };

  const renderModels = (): void => {
    modelsEl.replaceChildren();
    const list = document.createElement('ul');
    list.className = 'models';
    for (const model of HARNESS_MODELS) {
      const entry = getModel(model.id);
      const status = statuses[model.id];
      const downloaded = status?.state === 'ready' || status?.state === 'loaded';
      const busy = status?.state === 'downloading' || status?.state === 'verifying';

      const item = document.createElement('li');
      if (model.id === selectedId) item.className = 'selected';

      const title = document.createElement('div');
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'harness-model';
      radio.checked = model.id === selectedId;
      radio.disabled = running || !downloaded;
      radio.setAttribute('aria-label', `選取 ${model.name}`);
      radio.addEventListener('change', () => {
        if (!downloaded) return;
        selectedId = model.id;
        renderModels();
        updateRunButton();
      });
      const name = document.createElement('strong');
      name.textContent = model.name;
      const meta = document.createElement('span');
      meta.className = 'muted';
      meta.textContent = ` ${formatByteSize(entry.size)} · ${model.license} · ${describeStatus(status)}`;
      title.append(radio, name, meta);
      item.append(title);

      const desc = document.createElement('p');
      desc.className = 'muted';
      desc.textContent = model.descriptionZh;
      item.append(desc);

      const source = document.createElement('p');
      source.className = 'muted';
      const link = document.createElement('a');
      link.href = model.source;
      link.textContent = entry.repository;
      link.target = '_blank';
      link.rel = 'noreferrer';
      source.append('來源：', link);
      item.append(source);

      const actions = document.createElement('div');
      actions.className = 'actions';
      if (busy) {
        const note = document.createElement('span');
        note.className = 'muted';
        note.textContent = '下載中…（中斷後可續傳）';
        actions.append(note);
      } else if (downloaded) {
        const remove = document.createElement('button');
        remove.type = 'button';
        remove.textContent = '刪除';
        remove.disabled = running;
        remove.addEventListener('click', () => {
          void (async () => {
            if (
              !window.confirm(
                `刪除已下載的 ${model.name}（${formatByteSize(entry.size)}）？\n之後要再測需重新下載。`,
              )
            ) {
              return;
            }
            await manager.deleteModel(model.id);
            await refreshStatuses();
            renderModels();
            updateRunButton();
            log(`deleted ${model.name}`);
          })();
        });
        actions.append(remove);
      } else {
        const download = document.createElement('button');
        download.type = 'button';
        download.textContent = `下載（${formatByteSize(entry.size)}）`;
        download.disabled = running;
        download.addEventListener('click', () => {
          void (async () => {
            // Confirm first, showing the size: these are 397 MB–1.1 GB
            // downloads on a phone, so the data cost is the user's call.
            if (
              !window.confirm(
                `下載 ${model.name}（${formatByteSize(entry.size)}）？\n下載會使用行動數據，檔案會保存在本機，可在下方刪除。`,
              )
            ) {
              return;
            }
            const controller = new AbortController();
            try {
              await manager.download(model.id, {
                signal: controller.signal,
                onProgress: (progress) => {
                  statuses = {
                    ...statuses,
                    [model.id]: {
                      id: model.id,
                      state: progress.phase === 'verify' ? 'verifying' : 'downloading',
                      received: progress.received,
                      total: entry.size,
                      progress: progress.ratio,
                    },
                  };
                  renderModels();
                },
              });
              log(`downloaded ${model.name}`);
            } catch (error) {
              if (controller.signal.aborted) return;
              log(
                `download failed ${model.name}: ${error instanceof Error ? error.message : String(error)}`,
              );
            } finally {
              await refreshStatuses();
              renderModels();
              updateRunButton();
            }
          })();
        });
        actions.append(download);
      }
      item.append(actions);
      list.append(item);
    }
    modelsEl.append(list);
  };

  await refreshStatuses();
  renderModels();
  updateRunButton();

  runButton.addEventListener('click', () => {
    void (async () => {
      running = true;
      renderModels();
      updateRunButton();
      try {
        const probe = {
          webgpu: capabilities.webgpu === 'ready',
          hardwareConcurrency: capabilities.hardwareConcurrency,
          fastBatches: capabilities.gpuFastBatches === true,
          sharedMemory: capabilities.crossOriginIsolated,
        };
        const configs = buildHarnessConfigs(probe, isAndroidEdgeRuntime());
        const cpuConfig = configs.find((c) => c.id === 'cpu');
        if (!cpuConfig) throw new Error('harness has no CPU reference config');
        const modelId = selectedId;
        const { prompt, stop } = buildHarnessModelPrompt(
          modelId,
          HARNESS_SOURCE_TEXT,
          HARNESS_SOURCE_LANGUAGE,
          HARNESS_TARGET_LANGUAGE,
        );
        log(`model: ${harnessModelName(modelId)}`);
        log(`prompt: ${prompt}`);
        log(`source: ${HARNESS_SOURCE_TEXT}`);
        const results: HarnessRunResult[] = [];
        for (const config of configs) {
          const result = await runConfig(
            manager,
            modelId,
            config,
            prompt,
            stop,
            cpuConfig.selection,
          );
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
      } catch (error) {
        reportFatal(error);
      } finally {
        running = false;
        await refreshStatuses();
        renderModels();
        updateRunButton();
      }
    })();
  });
}

void main().catch(reportFatal);
