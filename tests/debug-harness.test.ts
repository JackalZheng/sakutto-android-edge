import { describe, expect, it } from 'vitest';
import {
  buildHarnessConfigs,
  compareAdapterToReference,
  compareRuns,
  describeErrorChain,
  emptyIsolatedState,
  firstStreamedToken,
  layersOnCpuOverride,
  offloadedRepeatingLayers,
  parseIsolatedState,
  planIsolatedQueue,
  plamo3SwaLayers,
  PLAMO3_FULL_ATTENTION_LAYERS,
  POCO_F5_PRO_REFERENCE,
  recordIsolatedResult,
  sameRun,
  verdictForRun,
  type HarnessRunResult,
  type IsolatedRunState,
} from '../src/engine/debug-harness';
import { EngineError } from '../src/engine/errors';
import type { BackendProbe } from '../src/engine/in-browser/backend';
import { EMPTY_GPU_DIAGNOSTIC, type GpuDiagnostic } from '../src/shared/gpu-diagnostic';

const GPU_PROBE: BackendProbe = {
  webgpu: true,
  hardwareConcurrency: 8,
  fastBatches: false,
  sharedMemory: true,
};

function run(partial: Partial<HarnessRunResult>): HarnessRunResult {
  return {
    configId: 'gpu-default',
    ok: true,
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
    ...partial,
  };
}

describe('buildHarnessConfigs', () => {
  it('always starts with the CPU reference', () => {
    const configs = buildHarnessConfigs(GPU_PROBE, false);
    expect(configs[0]?.id).toBe('cpu');
    expect(configs[0]?.selection.backend).toBe('cpu-wasm');
  });

  it('adds the GPU configurations when WebGPU is available', () => {
    const ids = buildHarnessConfigs(GPU_PROBE, false).map((c) => c.id);
    expect(ids).toEqual([
      'cpu',
      'gpu-default',
      'gpu-release',
      'gpu-1-slot',
      'gpu-nctx-2048',
      'gpu-no-f16',
      'gpu-no-subgroups',
      'gpu-adreno-upstream',
      'gpu-adreno-matvec',
      'gpu-adreno-sg-tiled',
      'gpu-adreno-tiled',
      'gpu-adreno-submit-64',
      'gpu-matvec-1-slot',
      'gpu-matvec-no-fa',
      'gpu-matvec-small-batch-64',
      'gpu-matvec-small-batch-32',
      'gpu-layers-1',
      'gpu-layers-2',
      'gpu-layers-3',
      'gpu-layers-4',
      'gpu-layers-8',
      'gpu-no-flash-attn',
      'gpu-kv-on-cpu',
      'gpu-swa-layers-only',
      'gpu-full-attn-layers-only',
    ]);
  });

  it('omits the GPU configurations when WebGPU is unavailable', () => {
    const ids = buildHarnessConfigs({ ...GPU_PROBE, webgpu: false }, false).map((c) => c.id);
    expect(ids).toEqual(['cpu']);
  });

  it('forces one slot and a 1024 context for the 1-slot config', () => {
    const config = buildHarnessConfigs(GPU_PROBE, false).find((c) => c.id === 'gpu-1-slot');
    expect(config?.selection.nParallel).toBe(1);
    expect(config?.selection.nCtx).toBe(1024);
  });

  it('sets n_ctx 2048 for the context config and keeps the default slot count', () => {
    const configs = buildHarnessConfigs(GPU_PROBE, false);
    const def = configs.find((c) => c.id === 'gpu-default');
    const ctx = configs.find((c) => c.id === 'gpu-nctx-2048');
    expect(ctx?.selection.nCtx).toBe(2048);
    expect(ctx?.selection.nParallel).toBe(def?.selection.nParallel);
  });

  it('marks only the profile configs with a debug profile', () => {
    const configs = buildHarnessConfigs(GPU_PROBE, false);
    for (const config of configs) {
      const expected: Record<string, string> = {
        'gpu-no-f16': 'no-f16',
        'gpu-no-subgroups': 'no-subgroups',
        'gpu-adreno-upstream': 'adreno-subgroups',
        'gpu-adreno-matvec': 'adreno-matvec',
        'gpu-adreno-sg-tiled': 'adreno-sg-tiled',
        'gpu-adreno-tiled': 'no-matvec',
        'gpu-adreno-submit-64': 'adreno-submit-64',
        'gpu-matvec-1-slot': 'adreno-matvec',
        'gpu-matvec-no-fa': 'adreno-matvec',
        'gpu-matvec-small-batch-64': 'adreno-matvec',
        'gpu-matvec-small-batch-32': 'adreno-matvec',
      };
      expect(config.debugProfile).toBe(expected[config.id] ?? null);
    }
  });

  it('offloads 1 and 8 layers for the partial-offload configs', () => {
    const configs = buildHarnessConfigs(GPU_PROBE, false);
    const def = configs.find((c) => c.id === 'gpu-default');
    const one = configs.find((c) => c.id === 'gpu-layers-1');
    const eight = configs.find((c) => c.id === 'gpu-layers-8');
    expect(one?.selection.nGpuLayers).toBe(1);
    expect(eight?.selection.nGpuLayers).toBe(8);
    expect(one?.selection.nParallel).toBe(def?.selection.nParallel);
    expect(one?.selection.nCtx).toBe(def?.selection.nCtx);
    expect(eight?.selection.nParallel).toBe(def?.selection.nParallel);
    expect(eight?.selection.nCtx).toBe(def?.selection.nCtx);
  });

  it('falls back to CPU only for the no-f16 config', () => {
    const configs = buildHarnessConfigs(GPU_PROBE, false);
    for (const config of configs) {
      expect(config.fallbackToCpu ?? false).toBe(config.id === 'gpu-no-f16');
    }
  });

  it('applies the mobile limits on Android', () => {
    const desktop = buildHarnessConfigs(GPU_PROBE, false).find((c) => c.id === 'gpu-default');
    const android = buildHarnessConfigs(GPU_PROBE, true).find((c) => c.id === 'gpu-default');
    expect(android?.selection.nBatch).toBeDefined();
    expect(desktop?.selection.nBatch).toBeUndefined();
    expect(android?.selection.nThreads).toBeLessThanOrEqual(desktop?.selection.nThreads ?? 0);
  });
});

describe('firstStreamedToken', () => {
  it('returns the first non-empty chunk', () => {
    expect(firstStreamedToken(['', '早', '安'])).toBe('早');
  });

  it('returns an empty string when nothing streamed', () => {
    expect(firstStreamedToken([])).toBe('');
    expect(firstStreamedToken([''])).toBe('');
  });
});

describe('compareAdapterToReference', () => {
  const pocoDiagnostic: GpuDiagnostic = {
    ...EMPTY_GPU_DIAGNOSTIC,
    vendor: POCO_F5_PRO_REFERENCE.vendor,
    architecture: POCO_F5_PRO_REFERENCE.architecture,
    fallbackAdapter: POCO_F5_PRO_REFERENCE.fallbackAdapter,
    shaderF16: POCO_F5_PRO_REFERENCE.shaderF16,
    subgroups: POCO_F5_PRO_REFERENCE.subgroups,
    subgroupMinSize: POCO_F5_PRO_REFERENCE.subgroupMinSize,
    subgroupMaxSize: POCO_F5_PRO_REFERENCE.subgroupMaxSize,
    maxBufferSize: POCO_F5_PRO_REFERENCE.maxBufferSize,
    maxStorageBufferBindingSize: POCO_F5_PRO_REFERENCE.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: POCO_F5_PRO_REFERENCE.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: POCO_F5_PRO_REFERENCE.maxComputeInvocationsPerWorkgroup,
    adapterAttempts: ['high-performance'],
  };

  it('matches every field for the reference device', () => {
    const rows = compareAdapterToReference(pocoDiagnostic);
    expect(rows).toHaveLength(12);
    expect(rows.every((row) => row.matches)).toBe(true);
  });

  it('flags a different subgroup width', () => {
    const rows = compareAdapterToReference({ ...pocoDiagnostic, subgroupMaxSize: 32 });
    const row = rows.find((r) => r.field === 'subgroupMaxSize');
    expect(row?.matches).toBe(false);
    expect(row?.reference).toBe('128');
    expect(row?.actual).toBe('32');
    expect(rows.filter((r) => !r.matches)).toHaveLength(1);
  });

  it('treats missing values as differences, not matches', () => {
    const rows = compareAdapterToReference(EMPTY_GPU_DIAGNOSTIC);
    expect(rows.every((row) => !row.matches)).toBe(true);
    expect(rows.find((r) => r.field === 'vendor')?.actual).toBe('n/a');
  });
});

describe('verdictForRun', () => {
  const goodOutput = '早班列車離開安靜的車站，麵包師傅在轉角開了一間小店。';
  const cpu = run({ configId: 'cpu', output: goodOutput, firstToken: '早' });

  it('passes when the GPU output agrees with the CPU reference', () => {
    const verdict = verdictForRun(cpu, run({ output: goodOutput, firstToken: '早' }));
    expect(verdict.pass).toBe(true);
    expect(verdict.reasons).toEqual([]);
  });

  it('fails on a differing first token even if the text is not flagged broken', () => {
    const verdict = verdictForRun(
      cpu,
      run({ output: '晚班列車離開安靜的車站，麵包師傅在轉角開了一間小店。', firstToken: '晚' }),
    );
    expect(verdict.pass).toBe(false);
    expect(verdict.reasons.join(' ')).toContain('first token');
  });

  it('fails on garbled output', () => {
    const verdict = verdictForRun(
      cpu,
      run({ output: '難ら일까지郷この難ら일까지郷この', firstToken: '難' }),
    );
    expect(verdict.pass).toBe(false);
  });

  it('fails when the run itself failed', () => {
    const verdict = verdictForRun(cpu, run({ ok: false, error: 'Error: load failed' }));
    expect(verdict.pass).toBe(false);
    expect(verdict.reasons.join(' ')).toContain('load failed');
  });

  it('ignores the built-in probe status either way', () => {
    const probeOk = verdictForRun(
      cpu,
      run({ output: '難ら일까지郷この難ら일까지郷この', firstToken: '難', correctnessProbe: 'ok' }),
    );
    expect(probeOk.pass).toBe(false);
    const probeFailed = verdictForRun(
      cpu,
      run({ output: goodOutput, firstToken: '早', correctnessProbe: 'failed' }),
    );
    expect(probeFailed.pass).toBe(true);
  });
});

describe('compareRuns', () => {
  it('reports a match for identical runs', () => {
    const output = '早班列車離開安靜的車站，麵包師傅在轉角開了一間小店。';
    const cpu = run({ configId: 'cpu', output, firstToken: '早' });
    const gpu = run({ output, firstToken: '早' });
    const comparison = compareRuns(cpu, gpu);
    expect(comparison.firstTokenMatches).toBe(true);
    expect(comparison.outputsIdentical).toBe(true);
    expect(comparison.gpuLooksBroken).toBe(false);
  });

  it('flags a differing first token', () => {
    const cpu = run({ configId: 'cpu', output: '早安列車', firstToken: '早' });
    const gpu = run({ output: '難ら列車', firstToken: '難' });
    expect(compareRuns(cpu, gpu).firstTokenMatches).toBe(false);
  });

  it('flags mixed-script garbage as broken for a zh-hant target', () => {
    const cpu = run({
      configId: 'cpu',
      output: '早班列車離開安靜的車站，麵包師傅在轉角開了一間小店。',
      firstToken: '早',
    });
    const gpu = run({ output: '難ら일까지郷この難ら일까지郷この', firstToken: '難' });
    const comparison = compareRuns(cpu, gpu);
    expect(comparison.cpuLooksBroken).toBe(false);
    expect(comparison.gpuLooksBroken).toBe(true);
  });
});

describe('PLaMo 3 layer layout helpers', () => {
  it('lists layers 7 and 15 as full attention and the other 14 as SWA', () => {
    expect(PLAMO3_FULL_ATTENTION_LAYERS).toEqual([7, 15]);
    const swa = plamo3SwaLayers();
    expect(swa).toHaveLength(14);
    expect(swa).not.toContain(7);
    expect(swa).not.toContain(15);
  });

  it('offloads repeating layers from the end, after the output layer', () => {
    expect(offloadedRepeatingLayers(1)).toEqual([]);
    expect(offloadedRepeatingLayers(2)).toEqual([15]);
    expect(offloadedRepeatingLayers(4)).toEqual([13, 14, 15]);
    expect(offloadedRepeatingLayers(17)).toHaveLength(16);
    expect(offloadedRepeatingLayers(99)).toHaveLength(16);
  });

  it('builds an -ot rule that matches only the listed layers', () => {
    const rule = layersOnCpuOverride([7, 15]);
    expect(rule).toBe('blk\\.(7|15)\\.=CPU');
    const pattern = new RegExp(rule.slice(0, rule.lastIndexOf('=')));
    expect(pattern.test('blk.7.attn_qkv.weight')).toBe(true);
    expect(pattern.test('blk.15.ffn_up.weight')).toBe(true);
    expect(pattern.test('blk.1.attn_qkv.weight')).toBe(false);
    expect(pattern.test('blk.17.attn_qkv.weight')).toBe(false);
  });
});

describe('debug load configs', () => {
  const configs = buildHarnessConfigs(GPU_PROBE, false);
  const byId = (id: string) => configs.find((c) => c.id === id);

  it('offloads 2, 3 and 4 layers for the extra partial-offload configs', () => {
    expect(byId('gpu-layers-2')?.selection.nGpuLayers).toBe(2);
    expect(byId('gpu-layers-3')?.selection.nGpuLayers).toBe(3);
    expect(byId('gpu-layers-4')?.selection.nGpuLayers).toBe(4);
    expect(byId('gpu-layers-2')?.label).toContain('layer 15');
  });

  it('turns flash attention off only for the flash-attention configs', () => {
    const off = new Set(['gpu-no-flash-attn', 'gpu-matvec-no-fa']);
    for (const config of configs) {
      expect(config.selection.debugLoad?.flashAttn).toBe(off.has(config.id) ? false : undefined);
    }
  });

  it('keeps the KV cache on the CPU for the kv-on-cpu config', () => {
    expect(byId('gpu-kv-on-cpu')?.selection.debugLoad?.noKvOffload).toBe(true);
    expect(byId('gpu-default')?.selection.debugLoad).toBeUndefined();
  });

  it('moves the full-attention layers to the CPU for the SWA-only config and vice versa', () => {
    expect(byId('gpu-swa-layers-only')?.selection.debugLoad?.tensorOverrides).toBe(
      layersOnCpuOverride([7, 15]),
    );
    expect(byId('gpu-full-attn-layers-only')?.selection.debugLoad?.tensorOverrides).toBe(
      layersOnCpuOverride(plamo3SwaLayers()),
    );
  });

  it('never sets debug load options on the CPU reference', () => {
    expect(byId('cpu')?.selection.debugLoad).toBeUndefined();
  });
});

describe('isolated run state', () => {
  const ids = ['cpu', 'gpu-default', 'gpu-layers-2'];

  it('puts the CPU reference first when none is stored', () => {
    expect(planIsolatedQueue(ids, ['gpu-layers-2'], emptyIsolatedState())).toEqual([
      'cpu',
      'gpu-layers-2',
    ]);
  });

  it('reuses a stored CPU result for a single config', () => {
    const state = recordIsolatedResult(
      { ...emptyIsolatedState(), queue: ['cpu'] },
      run({ configId: 'cpu' }),
      'log\n',
    );
    expect(planIsolatedQueue(ids, ['gpu-layers-2'], state)).toEqual(['gpu-layers-2']);
    expect(planIsolatedQueue(ids, ['cpu', 'gpu-default'], state)).toEqual(['cpu', 'gpu-default']);
  });

  it('drops unknown and duplicate ids', () => {
    expect(
      planIsolatedQueue(ids, ['nope', 'gpu-default', 'gpu-default'], emptyIsolatedState()),
    ).toEqual(['cpu', 'gpu-default']);
  });

  it('records a result, pops the queue head and replaces older results', () => {
    let state = { ...emptyIsolatedState(), queue: ['gpu-default', 'gpu-layers-2'] };
    state = recordIsolatedResult(state, run({ configId: 'gpu-default', output: 'a' }), 'one\n');
    expect(state.queue).toEqual(['gpu-layers-2']);
    state = { ...state, queue: ['gpu-default'] };
    state = recordIsolatedResult(state, run({ configId: 'gpu-default', output: 'b' }), 'two\n');
    expect(state.results.filter((r) => r.configId === 'gpu-default')).toHaveLength(1);
    expect(state.results[0]?.output).toBe('b');
    expect(state.log).toBe('one\ntwo\n');
  });

  it('round-trips through JSON and rejects malformed state', () => {
    const state = recordIsolatedResult(emptyIsolatedState(), run({ configId: 'cpu' }), 'x');
    expect(parseIsolatedState(JSON.stringify(state))).toEqual(state);
    expect(parseIsolatedState(null)).toBeNull();
    expect(parseIsolatedState('{')).toBeNull();
    expect(parseIsolatedState('{"version":2,"queue":[],"results":[],"log":""}')).toBeNull();
  });
});

describe('describeErrorChain', () => {
  it('includes the cause and the detail', () => {
    const cause = new Error('abort(ggml_webgpu: Device lost! Reason: 2)');
    cause.name = 'RuntimeError';
    const error = new EngineError('gpu-lost', 'The GPU stopped working.', { cause });
    const text = describeErrorChain(error);
    expect(text).toContain('EngineError: The GPU stopped working.');
    expect(text).toContain('RuntimeError: abort(ggml_webgpu: Device lost! Reason: 2)');
  });

  it('handles plain strings', () => {
    expect(describeErrorChain('worker gone')).toBe('worker gone');
  });
});

describe('model-aware isolated state', () => {
  const run = (configId: string, modelId?: string): HarnessRunResult => ({
    configId,
    ...(modelId ? { modelId } : {}),
    ok: true,
    output: 'x',
    firstToken: 'x',
    ttftMs: 1,
    totalMs: 2,
    error: null,
    gpuEvidence: null,
    correctnessProbe: null,
    correctnessProbeSubgroupSize: null,
    correctnessProbeMs: null,
    fellBackToCpu: false,
  });

  it('needs a CPU reference per model', () => {
    const state = { ...emptyIsolatedState(), results: [run('cpu', 'a')] };
    expect(planIsolatedQueue(['cpu', 'gpu-default'], ['gpu-default'], state, 'a')).toEqual([
      'gpu-default',
    ]);
    expect(planIsolatedQueue(['cpu', 'gpu-default'], ['gpu-default'], state, 'b')).toEqual([
      'cpu',
      'gpu-default',
    ]);
  });

  it('keeps results of the same config for different models', () => {
    let state: IsolatedRunState = { ...emptyIsolatedState(), queue: ['cpu'], modelId: 'b' };
    state = recordIsolatedResult({ ...state, results: [run('cpu', 'a')] }, run('cpu', 'b'), '');
    expect(state.results.map((r) => r.modelId)).toEqual(['a', 'b']);
    expect(state.modelId).toBe('b');
    expect(sameRun(run('cpu', 'a'), 'cpu', 'a')).toBe(true);
    expect(sameRun(run('cpu', 'a'), 'cpu', 'b')).toBe(false);
  });

  it('round-trips the queued model id', () => {
    const raw = JSON.stringify({ ...emptyIsolatedState(), modelId: 'qwen3-0.6b-q4_k_m' });
    expect(parseIsolatedState(raw)?.modelId).toBe('qwen3-0.6b-q4_k_m');
  });

  it('marks the layer-index override configs as PLaMo-only', () => {
    const configs = buildHarnessConfigs(
      { webgpu: true, hardwareConcurrency: 8, fastBatches: true, sharedMemory: true },
      false,
    );
    expect(configs.filter((c) => c.plamoOnly).map((c) => c.id)).toEqual([
      'gpu-swa-layers-only',
      'gpu-full-attn-layers-only',
    ]);
  });
});

describe('Adreno mat-vec profiles (patch 0017)', () => {
  it('runs each Adreno profile once, on the WebGPU selection', () => {
    const configs = buildHarnessConfigs(
      { webgpu: true, hardwareConcurrency: 8, fastBatches: true, sharedMemory: true },
      true,
    );
    const byId = new Map(configs.map((c) => [c.id, c]));
    expect(byId.get('gpu-adreno-upstream')?.debugProfile).toBe('adreno-subgroups');
    expect(byId.get('gpu-adreno-matvec')?.debugProfile).toBe('adreno-matvec');
    expect(byId.get('gpu-adreno-sg-tiled')?.debugProfile).toBe('adreno-sg-tiled');
    expect(byId.get('gpu-adreno-tiled')?.debugProfile).toBe('no-matvec');
    expect(byId.get('gpu-adreno-submit-64')?.debugProfile).toBe('adreno-submit-64');
    for (const id of [
      'gpu-adreno-upstream',
      'gpu-adreno-matvec',
      'gpu-adreno-sg-tiled',
      'gpu-adreno-tiled',
      'gpu-adreno-submit-64',
    ]) {
      expect(byId.get(id)?.selection.backend).toBe('webgpu');
    }
  });

  it('varies one load option per device-lost bisection config, on the mat-vec default', () => {
    const configs = buildHarnessConfigs(
      { webgpu: true, hardwareConcurrency: 8, fastBatches: true, sharedMemory: true },
      true,
    );
    const byId = new Map(configs.map((c) => [c.id, c]));
    const def = byId.get('gpu-default')!;
    const oneSlot = byId.get('gpu-matvec-1-slot')!;
    expect(oneSlot.selection.nParallel).toBe(1);
    expect(oneSlot.selection.nCtx).toBe(1024);
    expect(byId.get('gpu-matvec-no-fa')?.selection.debugLoad).toEqual({ flashAttn: false });
    expect(byId.get('gpu-matvec-small-batch-64')?.selection.nBatch).toBe(64);
    expect(byId.get('gpu-matvec-small-batch-32')?.selection.nBatch).toBe(32);
    for (const id of [
      'gpu-matvec-no-fa',
      'gpu-matvec-small-batch-64',
      'gpu-matvec-small-batch-32',
    ]) {
      expect(byId.get(id)?.selection.nParallel).toBe(def.selection.nParallel);
      expect(byId.get(id)?.selection.nCtx).toBe(def.selection.nCtx);
      expect(byId.get(id)?.debugProfile).toBe('adreno-matvec');
    }
  });
});
