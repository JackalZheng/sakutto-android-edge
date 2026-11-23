import { describe, expect, it } from 'vitest';
import {
  buildPerfConfigs,
  crashedPerfResult,
  emptyPerfState,
  formatPerfReport,
  parseGpuOpsLine,
  parseGpuStatLine,
  parsePerfState,
  perfBreakdownQueue,
  perfDebugString,
  type PerfResult,
  PerfTimeoutError,
  perfSweepQueue,
  recordPerfResult,
  resolvePerfConfig,
  summarizeGpuOps,
  summarizeGpuStats,
  withPerfTimeout,
} from '@/engine/debug-perf';

const probe = { webgpu: true, hardwareConcurrency: 8, fastBatches: false, sharedMemory: true };

const stat = (g: number, n: number, extra = ''): string =>
  `SAKUTTO_GPUSTAT g=${g} n=${n} nodes=662 kernels=355 submits=23 encode_ms=2.000 cycle_ms=${n > 4 ? 0 : 100}.000 readback_ms=60.000 readbacks=1${extra}`;

describe('perf configs', () => {
  it('builds a CPU baseline and round-4 variants with the perf token', () => {
    const configs = buildPerfConfigs(probe, true, 16);
    expect(configs[0]?.id).toBe('cpu');
    expect(configs[0]?.label).toContain('threads');
    const ids = configs.map((c) => c.id);
    expect(ids).toEqual(
      expect.arrayContaining([
        'gpu-mv',
        'gpu-r3',
        'gpu-old-softmax',
        'gpu-rwg128',
        'gpu-ctx2048-np2',
        'gpu-ctx1024-np1',
        'gpu-sg-fa',
        'gpu-half-layers',
        'gpu-half-ctx2048-np2',
      ]),
    );
    expect(ids.at(-1)).toBe('gpu-mv-repeat');
    for (const config of configs.slice(1)) {
      expect(config.tuning).toContain('perf');
      expect(config.debugProfile).toBe(
        config.id === 'gpu-sg-fa' ? 'adreno-sg-tiled' : 'adreno-matvec',
      );
    }
    expect(perfDebugString(configs.find((c) => c.id === 'gpu-r3')!)).toBe(
      'adreno-matvec+mvwg256+smup+rwg1024+perf',
    );
    const np1 = configs.find((c) => c.id === 'gpu-ctx1024-np1')!;
    expect(np1.selection.nParallel).toBe(1);
    expect(np1.selection.nCtx).toBe(1024);
    expect(perfDebugString(np1)).toBe('adreno-matvec+perf');
    expect(configs.find((c) => c.id === 'gpu-half-layers')?.selection.nGpuLayers).toBe(9);
  });

  it('skips the hybrid config for an unknown layer count and GPU configs without WebGPU', () => {
    expect(buildPerfConfigs(probe, true, null).some((c) => c.id.startsWith('gpu-half'))).toBe(
      false,
    );
    expect(buildPerfConfigs({ ...probe, webgpu: false }, true, 16).map((c) => c.id)).toEqual([
      'cpu',
    ]);
  });

  it('resolves the per-op variant and plans the queues', () => {
    const configs = buildPerfConfigs(probe, true, 16);
    const ops = resolvePerfConfig(configs, 'gpu-mv#ops')!;
    expect(ops.tuning).toEqual(['perfops']);
    expect(resolvePerfConfig(configs, 'cpu#ops')).toBeNull();
    expect(resolvePerfConfig(configs, 'nope')).toBeNull();
    const sweep = perfSweepQueue(configs);
    expect(sweep[0]).toBe('cpu');
    expect(sweep.at(-1)).toBe('gpu-mv#ops');
    expect(perfBreakdownQueue(configs, 'gpu-rwg128')).toEqual(['gpu-rwg128', 'gpu-rwg128#ops']);
    expect(perfBreakdownQueue(configs, 'cpu')).toEqual(['cpu']);
  });
});

describe('ggml-webgpu stat lines', () => {
  it('parses GPUSTAT and GPUOPS lines', () => {
    expect(parseGpuStatLine(`0.08.9 I ${stat(3, 1)}`)).toEqual({
      graph: 3,
      n: 1,
      nodes: 662,
      kernels: 355,
      submits: 23,
      encodeMs: 2,
      cycleMs: 100,
      readbackMs: 60,
      readbacks: 1,
    });
    expect(parseGpuStatLine('ggml_webgpu: device ready')).toBeNull();
    const ops = parseGpuOpsLine(
      'SAKUTTO_GPUOPS g=1 n=1 ops_ms=749.122 sync_ms=2.138 RMS_NORM:f32+fused=49.064,MUL_MAT:q4_K:n<=4=419.078',
    )!;
    expect(ops.syncMs).toBeCloseTo(2.138);
    expect(ops.ops).toEqual([
      { key: 'RMS_NORM:f32+fused', ms: 49.064 },
      { key: 'MUL_MAT:q4_K:n<=4', ms: 419.078 },
    ]);
  });

  it('summarizes decode tokens from consecutive decode graphs', () => {
    const lines = [stat(0, 41), stat(1, 1), stat(2, 1), stat(3, 1)].map((line) =>
      parseGpuStatLine(line)!,
    );
    const summary = summarizeGpuStats(lines)!;
    expect(summary.promptGraphs).toBe(1);
    expect(summary.decodeGraphs).toBe(3);
    expect(summary.decodeTokens).toBe(2);
    expect(summary.cycleMs).toBe(100);
    expect(summary.readbackWaitMs).toBe(60);
    expect(summary.otherMs).toBe(38);
    expect(summary.submitsPerToken).toBe(23);
    expect(summarizeGpuStats([])).toBeNull();
  });

  it('averages per-op times over decode graphs', () => {
    const lines = [
      'SAKUTTO_GPUOPS g=0 n=7 ops_ms=10 sync_ms=1 MUL_MAT:q4_K:n>4=10',
      'SAKUTTO_GPUOPS g=1 n=1 ops_ms=4 sync_ms=0.5 MUL_MAT:q4_K:n<=4=3,ADD:f32=1',
      'SAKUTTO_GPUOPS g=2 n=1 ops_ms=6 sync_ms=0.5 MUL_MAT:q4_K:n<=4=5,ADD:f32=1',
    ].map((line) => parseGpuOpsLine(line)!);
    const summary = summarizeGpuOps(lines)!;
    expect(summary.decodeGraphs).toBe(2);
    expect(summary.opsMs).toBe(5);
    expect(summary.ops[0]).toEqual({ key: 'MUL_MAT:q4_K:n<=4', ms: 4, share: 0.8 });
    expect(summary.promptOps[0]?.key).toBe('MUL_MAT:q4_K:n>4');
  });
});

describe('perf report and sweep state', () => {
  const result: PerfResult = {
    configId: 'gpu-mv',
    modelId: 'm',
    label: 'adreno-matvec',
    selectionText: 'backend=webgpu',
    debugString: 'adreno-matvec+perf',
    ok: true,
    error: null,
    loadMs: 1234,
    gpuEvidence: 'engaged',
    runs: [
      {
        ttftMs: 900,
        totalMs: 5000,
        promptTokens: 40,
        promptMs: 800,
        predictedTokens: 20,
        predictedMs: 4000,
        output: 'x',
      },
      {
        ttftMs: 400,
        totalMs: 3000,
        promptTokens: 40,
        promptMs: 400,
        predictedTokens: 20,
        predictedMs: 2500,
        output: 'x',
      },
    ],
    gpuStats: null,
    gpuOps: null,
    notes: [],
  };

  it('warns about a run without output', () => {
    const empty = {
      ...result,
      runs: [result.runs[0]!, { ...result.runs[1]!, output: '', predictedTokens: 1 }],
    };
    expect(formatPerfReport([], [empty])).toContain('WARNING: run 2 produced no text (1 tokens)');
  });

  it('formats a copy-pasteable report', () => {
    const text = formatPerfReport(['model: m'], [result]);
    expect(text).toContain('=== Sakutto GPU perf report ===');
    expect(text).toMatch(/gpu-mv\s+1234\s+400\s+3000\s+100\.00\s+8\.00\s+125/u);
    expect(text).toContain('first-run cost');
    expect(text).toContain('ttft 500 ms, total 2000 ms');
  });

  it('round-trips state and records crashes', () => {
    const state = {
      ...emptyPerfState('m'),
      queue: ['cpu', 'gpu-mv'],
      running: 'cpu',
      partial: { ...result, configId: 'cpu' },
    };
    expect(parsePerfState(JSON.stringify(state))).toEqual(state);
    expect(parsePerfState('{')).toBeNull();
    const next = recordPerfResult(state, crashedPerfResult('cpu', 'm', 'crashed'));
    expect(next.queue).toEqual(['gpu-mv']);
    expect(next.running).toBeNull();
    expect(next.results[0]?.ok).toBe(false);
  });

  it('times out a stuck config', async () => {
    await expect(withPerfTimeout(new Promise(() => undefined), 5)).rejects.toBeInstanceOf(
      PerfTimeoutError,
    );
    await expect(withPerfTimeout(Promise.resolve(1), 50)).resolves.toBe(1);
  });
});
