import { describe, expect, it } from 'vitest';
import {
  buildHarnessConfigs,
  compareRuns,
  firstStreamedToken,
  type HarnessRunResult,
} from '../src/engine/debug-harness';
import type { BackendProbe } from '../src/engine/in-browser/backend';

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
