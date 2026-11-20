import { describe, expect, it } from 'vitest';
import {
  buildHarnessConfigs,
  compareAdapterToReference,
  compareRuns,
  firstStreamedToken,
  POCO_F5_PRO_REFERENCE,
  verdictForRun,
  type HarnessRunResult,
} from '../src/engine/debug-harness';
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
    expect(ids).toEqual(['cpu', 'gpu-default', 'gpu-1-slot', 'gpu-nctx-2048', 'gpu-no-f16']);
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

  it('marks only the no-f16 config with the debug profile', () => {
    const configs = buildHarnessConfigs(GPU_PROBE, false);
    for (const config of configs) {
      expect(config.debugProfile).toBe(config.id === 'gpu-no-f16' ? 'no-f16' : null);
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
