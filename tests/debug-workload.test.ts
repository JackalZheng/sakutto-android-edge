import { describe, expect, it } from 'vitest';
import {
  formatWorkloadRuns,
  formatWorkloadTable,
  runWorkload,
  summarizeWorkload,
  WORKLOAD_CONFIG_IDS,
  WORKLOAD_KINDS,
  WORKLOAD_TITLES_B,
} from '@/engine/debug-workload';
import { buildPerfConfigs, formatPerfReport, workloadQueue } from '@/engine/debug-perf';
import type { CompletionRequest, LlamaRuntime } from '@/engine/in-browser/runtime';

// GPU was much faster than the CPU for titles on the POCO F5 Pro ("gpu在翻標題速度快很多"):
// the harness compares short titles and a paragraph on CPU and WebGPU slot counts.

/** Answers every input line of a PLaMo prompt with one line; a few ms per request. */
function fakeRuntime(parallel: number, backend: 'webgpu' | 'cpu-wasm'): LlamaRuntime {
  return {
    backend,
    threads: 4,
    parallel,
    dispose: async () => undefined,
    async complete(request: CompletionRequest) {
      const input = /<\|plamo:msg\|>([^]*?)<\|plamo:tag\|>output/u.exec(request.prompt)?.[1] ?? 'x';
      await new Promise((resolve) => setTimeout(resolve, 2));
      const lines = input.split('\n').map((_line, index) => `譯${index}`);
      request.onToken(lines.join('\n'));
      request.onTimings?.({
        prompt_n: 20,
        prompt_ms: 10,
        predicted_n: lines.length * 4,
        predicted_ms: 20,
      } as never);
    },
  };
}

const OPTIONS = { format: 'plamo3' as const, contextTokens: 1024 };

describe('title vs paragraph workload', () => {
  it('packs PLaMo titles like the page (eager first, then up to 8 per request)', async () => {
    const run = await runWorkload(fakeRuntime(1, 'cpu-wasm'), 'titles-packed', 'B', OPTIONS);
    expect(run.error).toBeNull();
    expect(run.segments).toBe(WORKLOAD_TITLES_B.length);
    expect(run.requests).toHaveLength(3);
    expect(run.outputs.every((o) => o.length > 0)).toBe(true);
    expect(run.maxConcurrent).toBe(1);
    expect(summarizeWorkload(run).promptTokens).toBe(60);
  });

  it('sends headings one request each: 2 at once on 2 GPU slots, 1 at a time on one slot', async () => {
    const gpu = await runWorkload(fakeRuntime(2, 'webgpu'), 'titles-each', 'B', OPTIONS);
    expect(gpu.requests).toHaveLength(12);
    expect(gpu.maxConcurrent).toBe(2);
    const cpu = await runWorkload(fakeRuntime(1, 'cpu-wasm'), 'titles-each', 'B', OPTIONS);
    expect(cpu.requests).toHaveLength(12);
    expect(cpu.maxConcurrent).toBe(1);
    const four = await runWorkload(fakeRuntime(4, 'webgpu'), 'titles-each', 'B', OPTIONS);
    expect(four.maxConcurrent).toBeLessThanOrEqual(4);
  });

  it('does not pack titles for a chat model', async () => {
    const run = await runWorkload(fakeRuntime(1, 'cpu-wasm'), 'titles-packed', 'B', {
      format: 'hy-mt',
      contextTokens: 1024,
    });
    expect(run.requests).toHaveLength(12);
  });

  it('runs the paragraph and reports a table and per-request lines', async () => {
    const runs = [];
    for (const kind of WORKLOAD_KINDS) {
      runs.push(await runWorkload(fakeRuntime(2, 'webgpu'), kind, 'B', OPTIONS));
    }
    expect(runs[2]!.requests.length).toBeGreaterThanOrEqual(1);
    const table = formatWorkloadTable([{ configId: 'gpu-r6-r4', runs }]).join('\n');
    expect(table).toContain('gpu-r6-r4');
    expect(table).toContain('titles-packed');
    const lines = formatWorkloadRuns(runs).join('\n');
    expect(lines).toContain('req @');
    expect(lines).toContain('ttft avg');
  });

  it('has the comparison configs: CPU 1 and 2 slots, Adreno WebGPU 1 (default), 2 and 4', () => {
    const probe = { webgpu: true, hardwareConcurrency: 8, sharedMemory: true, adreno: true };
    const configs = buildPerfConfigs(probe, true, 26);
    expect(workloadQueue(configs)).toEqual([...WORKLOAD_CONFIG_IDS]);
    const slots = Object.fromEntries(
      configs
        .filter((c) => WORKLOAD_CONFIG_IDS.includes(c.id))
        .map((c) => [c.id, [c.selection.backend, c.selection.nParallel, c.selection.nCtx]]),
    );
    expect(slots).toEqual({
      cpu: ['cpu-wasm', 1, 2048],
      'cpu-2-slot': ['cpu-wasm', 2, 2048],
      'gpu-r6-r4': ['webgpu', 1, 2048],
      'gpu-r6-r4-2-slot': ['webgpu', 2, 2048],
      'gpu-r6-r4-4-slot': ['webgpu', 4, 4096],
    });
    // Not in the GPU perf sweep.
    expect(configs.find((c) => c.id === 'cpu-2-slot')?.sweep).toBe(false);
    expect(configs.find((c) => c.id === 'gpu-r6-r4-4-slot')?.sweep).toBe(false);
    // Without WebGPU only the CPU ones.
    expect(workloadQueue(buildPerfConfigs({ ...probe, webgpu: false }, true, 26))).toEqual([
      'cpu',
      'cpu-2-slot',
    ]);
  });

  it('puts the comparison table at the top of the perf report', async () => {
    const run = await runWorkload(fakeRuntime(1, 'cpu-wasm'), 'titles-packed', 'B', OPTIONS);
    const report = formatPerfReport(
      ['model: test'],
      [
        {
          configId: 'cpu',
          modelId: 'm',
          label: 'CPU',
          selectionText: 'backend=cpu-wasm',
          debugString: null,
          ok: true,
          error: null,
          loadMs: 10,
          gpuEvidence: null,
          runs: [],
          gpuStats: null,
          gpuOps: null,
          notes: [],
          workload: [run],
        },
      ],
    );
    expect(report.indexOf('title vs paragraph')).toBeLessThan(report.indexOf('[cpu]'));
  });
});
