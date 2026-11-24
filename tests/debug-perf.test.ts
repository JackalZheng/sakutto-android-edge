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
  PERF_DEFAULT_ID,
  perfSweepQueue,
  recordPerfResult,
  resolvePerfConfig,
  summarizeGpuOps,
  summarizeGpuStats,
  withPerfTimeout,
} from '@/engine/debug-perf';
import { WEBGPU_TUNING_TOKEN_PATTERN } from '@/shared/gpu-diagnostic';

const probe = { webgpu: true, hardwareConcurrency: 8, fastBatches: false, sharedMemory: true };

const stat = (g: number, n: number, extra = ''): string =>
  `SAKUTTO_GPUSTAT g=${g} n=${n} nodes=662 kernels=355 submits=23 encode_ms=2.000 cycle_ms=${n > 4 ? 0 : 100}.000 readback_ms=60.000 readbacks=1${extra}`;

describe('perf configs', () => {
  it('builds a CPU baseline, the 1.3.1 default, round 5 and the round-6 variants with perf', () => {
    const configs = buildPerfConfigs(probe, true, 16);
    expect(configs[0]?.id).toBe('cpu');
    expect(configs[0]?.label).toContain('threads');
    const ids = configs.map((c) => c.id);
    expect(ids).toEqual([
      'cpu',
      'cpu-2-slot',
      'gpu-r6-r4',
      'gpu-r5',
      'gpu-q4v2',
      'gpu-q4v2-r2',
      'gpu-q4v2-r4',
      'gpu-q4v2-t32',
      'gpu-mvglu',
      'gpu-fna',
      'gpu-r6',
      'gpu-r6-r4-1-slot',
      'gpu-r4',
      'gpu-r5-half',
      'gpu-r6-r4-half',
      'gpu-r6-r4-repeat',
      'gpu-r6-r4-4-slot',
    ]);
    for (const config of configs.filter((c) => c.selection.backend === 'webgpu')) {
      expect(config.tuning).toContain('perf');
      expect(config.debugProfile).toBe('adreno-matvec');
      for (const token of config.tuning) expect(WEBGPU_TUNING_TOKEN_PATTERN.test(token)).toBe(true);
    }
    expect(perfDebugString(configs.find((c) => c.id === 'gpu-r5')!)).toBe('adreno-matvec+perf');
    expect(perfDebugString(configs.find((c) => c.id === 'gpu-r6')!)).toBe(
      'adreno-matvec+q4v2r2+mvglu+fna+perf',
    );
    expect(perfDebugString(configs.find((c) => c.id === 'gpu-q4v2-t32')!)).toBe(
      'adreno-matvec+q4v2t32+q4v2r2+perf',
    );
    expect(perfDebugString(configs.find((c) => c.id === 'gpu-r6-r4')!)).toBe(
      'adreno-matvec+q4v2r4+mvglu+fna+perf',
    );
    expect(PERF_DEFAULT_ID).toBe('gpu-r6-r4');
    expect(configs.find((c) => c.id === 'gpu-r5-half')?.selection.nGpuLayers).toBe(9);
    expect(configs.find((c) => c.id === 'gpu-r6-r4-half')?.selection.nGpuLayers).toBe(9);
    expect(configs.find((c) => c.id === 'gpu-r6-r4-half')?.tuning).toEqual([
      'q4v2r4',
      'mvglu',
      'fna',
      'perf',
    ]);
  });

  it('accepts the round-6 tokens and rejects malformed ones', () => {
    for (const token of [
      'q4v2',
      'q4v2r1',
      'q4v2r2',
      'q4v2r4',
      'q4v2t8',
      'q4v2t64',
      'mvglu',
      'fna',
    ]) {
      expect(WEBGPU_TUNING_TOKEN_PATTERN.test(token)).toBe(true);
    }
    for (const token of ['q4v2r3', 'q4v2t4', 'q4v2x', 'fnax', 'mvglu2']) {
      expect(WEBGPU_TUNING_TOKEN_PATTERN.test(token)).toBe(false);
    }
  });

  it('skips the hybrid config for an unknown layer count and GPU configs without WebGPU', () => {
    expect(buildPerfConfigs(probe, true, null).some((c) => c.id.startsWith('gpu-half'))).toBe(
      false,
    );
    expect(buildPerfConfigs({ ...probe, webgpu: false }, true, 16).map((c) => c.id)).toEqual([
      'cpu',
      'cpu-2-slot',
    ]);
  });

  it('resolves the per-op variant and plans the queues', () => {
    const configs = buildPerfConfigs(probe, true, 16);
    const ops = resolvePerfConfig(configs, 'gpu-r5#ops')!;
    expect(ops.tuning).toEqual(['perfops']);
    expect(resolvePerfConfig(configs, 'cpu#ops')).toBeNull();
    expect(resolvePerfConfig(configs, 'nope')).toBeNull();
    const sweep = perfSweepQueue(configs);
    expect(sweep[0]).toBe('cpu');
    expect(sweep[1]).toBe('gpu-r6-r4');
    // No per-op runs (they crashed the page on the phone) and no hybrid configs in the sweep.
    expect(sweep.some((id) => id.endsWith('#ops'))).toBe(false);
    expect(sweep.some((id) => id.endsWith('-half'))).toBe(false);
    expect(sweep.at(-1)).toBe('gpu-r6-r4-repeat');
    expect(perfBreakdownQueue(configs, 'gpu-fna')).toEqual(['gpu-fna']);
    expect(perfBreakdownQueue(configs, 'gpu-fna', true)).toEqual(['gpu-fna', 'gpu-fna#ops']);
    expect(perfBreakdownQueue(configs, 'gpu-r5-half')).toEqual(['gpu-r5-half']);
    expect(perfBreakdownQueue(configs, 'cpu', true)).toEqual(['cpu']);
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
