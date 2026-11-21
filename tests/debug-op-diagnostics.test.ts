import { describe, expect, it } from 'vitest';
import {
  ADRENO_OP_PROFILES,
  formatNumber,
  MUL_MAT_GROUP_IDS,
  OpGroupTimeoutError,
  withOpGroupTimeout,
  isOpDiagnosticContextLine,
  formatOpRow,
  opGroupArgs,
  parseOpDiagnosticLines,
  PLAMO3_OP_GROUPS,
  summarizeOpGroup,
} from '../src/engine/debug-op-diagnostics';

const MM_VARS = 'type_a=q4_K,type_b=f32,m=2304,n=41,k=1536,bs=[1,1],nr=[1,1],per=[0,1,2,3]';

function stat(op: string, vars: string, nmse: string, extra = ''): string {
  return `SAKUTTO_OPSTAT\t${op}\t${op}\t${vars}\tn=94464\tnmse=${nmse}\tthr=5.000e-04\tmaxabs=1.250000e-02\tnan_gpu=0\tnan_cpu=0\tinf_gpu=0\tinf_cpu=0${extra}`;
}

describe('opGroupArgs', () => {
  it('selects the PLaMo 3 suite, the op and the params filter', () => {
    const group = PLAMO3_OP_GROUPS.find((g) => g.id === 'mul_mat-q4_k');
    expect(group).toBeDefined();
    expect(opGroupArgs(group!)).toEqual([
      'test',
      '--sakutto-plamo3',
      '-o',
      'MUL_MAT',
      '-p',
      'type_a=q4_K',
    ]);
    const rope = PLAMO3_OP_GROUPS.find((g) => g.id === 'rope');
    expect(opGroupArgs(rope!)).toEqual(['test', '--sakutto-plamo3', '-o', 'ROPE']);
    expect(opGroupArgs(rope!, 'CPU')).toEqual([
      'test',
      '--sakutto-plamo3',
      '-o',
      'ROPE',
      '-b',
      'CPU',
    ]);
  });

  it('has unique group ids', () => {
    const ids = PLAMO3_OP_GROUPS.map((g) => g.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('parseOpDiagnosticLines', () => {
  it('pairs stat lines with their result line', () => {
    const rows = parseOpDiagnosticLines([
      'Testing 1 devices',
      stat('MUL_MAT', MM_VARS, '1.000000e-06'),
      '  MUL_MAT(' + MM_VARS + '): OK',
      `SAKUTTO_OPRESULT\tOK\tMUL_MAT\t${MM_VARS}`,
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]?.status).toBe('OK');
    expect(rows[0]?.worst?.nmse).toBeCloseTo(1e-6);
    expect(rows[0]?.worst?.threshold).toBeCloseTo(5e-4);
    expect(rows[0]?.worst?.maxAbs).toBeCloseTo(0.0125);
  });

  it('finds a stat line that shares a line with earlier error text', () => {
    const rows = parseOpDiagnosticLines([
      `[MUL_MAT] ERR = 0.9 > 0.0005 ${stat('MUL_MAT', MM_VARS, '9.000000e-01')}`,
      `SAKUTTO_OPRESULT\tFAIL\tMUL_MAT\t${MM_VARS}`,
    ]);
    expect(rows[0]?.status).toBe('FAIL');
    expect(rows[0]?.worst?.nmse).toBeCloseTo(0.9);
  });

  it('keeps the worst node of a fused case and counts NaNs as worst', () => {
    const vars = 'type=f32,ne=[1536,41,1,1]';
    const rows = parseOpDiagnosticLines([
      stat('ADD', vars, '1.000000e-03'),
      `SAKUTTO_OPSTAT\tADD\tRMS_NORM\t${vars}\tn=10\tnmse=0.000000e+00\tthr=1.000e-07\tmaxabs=0\tnan_gpu=3\tnan_cpu=0\tinf_gpu=0\tinf_cpu=0`,
      `SAKUTTO_OPRESULT\tFAIL\tADD\t${vars}`,
    ]);
    expect(rows[0]?.nodes).toHaveLength(2);
    expect(rows[0]?.worst?.node).toBe('RMS_NORM');
    expect(rows[0]?.worst?.nanGpu).toBe(3);
  });

  it('reports a case without a result line (device lost) as INCOMPLETE', () => {
    const rows = parseOpDiagnosticLines([
      `SAKUTTO_OPRESULT\tOK\tROPE\ta`,
      stat('ROPE', 'b', '1.0e-09'),
      'ggml_webgpu: Device lost! Reason: 2, Message: ...',
    ]);
    expect(rows.map((r) => r.status)).toEqual(['OK', 'INCOMPLETE']);
  });

  it('records unsupported cases without stats', () => {
    const rows = parseOpDiagnosticLines([
      `SAKUTTO_OPRESULT\tNOT_SUPPORTED\tFLASH_ATTN_EXT\thsk=128`,
    ]);
    expect(rows[0]?.status).toBe('NOT_SUPPORTED');
    expect(rows[0]?.worst).toBeNull();
  });

  it('parses inf and nan numbers', () => {
    const rows = parseOpDiagnosticLines([
      `SAKUTTO_OPSTAT\tX\tX\tv\tn=1\tnmse=inf\tthr=1e-7\tmaxabs=nan\tnan_gpu=0\tnan_cpu=0\tinf_gpu=1\tinf_cpu=0`,
      `SAKUTTO_OPRESULT\tFAIL\tX\tv`,
    ]);
    expect(rows[0]?.worst?.nmse).toBe(Number.POSITIVE_INFINITY);
    expect(Number.isNaN(rows[0]?.worst?.maxAbs)).toBe(true);
  });
});

describe('summarizeOpGroup', () => {
  const group = PLAMO3_OP_GROUPS[0]!;
  const ok = parseOpDiagnosticLines([`SAKUTTO_OPRESULT\tOK\tMUL_MAT\ta`]);
  const fail = parseOpDiagnosticLines([`SAKUTTO_OPRESULT\tFAIL\tMUL_MAT\tb`]);

  it('passes when every case is OK', () => {
    expect(summarizeOpGroup({ group, rows: ok, error: null, retcode: 0 }).verdict).toBe('PASS');
  });

  it('fails on any FAIL case', () => {
    expect(
      summarizeOpGroup({ group, rows: [...ok, ...fail], error: null, retcode: 1 }).verdict,
    ).toBe('FAIL');
  });

  it('reports a crash when the worker died without a failing case', () => {
    expect(
      summarizeOpGroup({ group, rows: ok, error: 'RuntimeError: abort', retcode: null }).verdict,
    ).toBe('CRASH');
  });

  it('reports no data when nothing ran', () => {
    expect(summarizeOpGroup({ group, rows: [], error: null, retcode: 1 }).verdict).toBe('NO-DATA');
  });
});

describe('formatting', () => {
  it('formats numbers compactly', () => {
    expect(formatNumber(0)).toBe('0');
    expect(formatNumber(Number.NaN)).toBe('n/a');
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe('inf');
    expect(formatNumber(1.5e-6)).toBe('1.50e-6');
    expect(formatNumber(0.0125)).toBe('0.01250');
  });

  it('formats a row with its worst stats', () => {
    const rows = parseOpDiagnosticLines([
      stat('MUL_MAT', MM_VARS, '9.0e-01'),
      `SAKUTTO_OPRESULT\tFAIL\tMUL_MAT\t${MM_VARS}`,
    ]);
    expect(formatOpRow(rows[0]!)).toContain('FAIL');
    expect(formatOpRow(rows[0]!)).toContain('nmse=0.9000');
  });
});

describe('isOpDiagnosticContextLine', () => {
  it('keeps device and failure lines, drops per-case noise', () => {
    expect(isOpDiagnosticContextLine('Backend 2/2: WebGPU')).toBe(true);
    expect(isOpDiagnosticContextLine('  Skipping CPU backend')).toBe(true);
    expect(isOpDiagnosticContextLine('ggml_webgpu: Device lost! Reason: 2')).toBe(true);
    expect(isOpDiagnosticContextLine('  ROPE(type=f32): OK')).toBe(false);
  });
});

describe('withOpGroupTimeout', () => {
  function manualTimers() {
    const pending = new Map<number, () => void>();
    let next = 0;
    return {
      timers: {
        set: (fn: () => void) => {
          next += 1;
          pending.set(next, fn);
          return next;
        },
        clear: (id: unknown) => {
          pending.delete(id as number);
        },
      },
      fire: () => {
        for (const fn of pending.values()) fn();
      },
      pending,
    };
  }

  it('passes a result through and clears its timer', async () => {
    const clock = manualTimers();
    await expect(withOpGroupTimeout(Promise.resolve(7), 1000, clock.timers)).resolves.toBe(7);
    expect(clock.pending.size).toBe(0);
  });

  it('rejects a group that never answers, so the runner can move on', async () => {
    const clock = manualTimers();
    const hung = withOpGroupTimeout(new Promise<never>(() => undefined), 180_000, clock.timers);
    clock.fire();
    await expect(hung).rejects.toBeInstanceOf(OpGroupTimeoutError);
    await expect(hung).rejects.toThrow(/180 s/u);
  });

  it('keeps the rejection of a group that crashed', async () => {
    const clock = manualTimers();
    await expect(
      withOpGroupTimeout(Promise.reject(new Error('Device lost')), 1000, clock.timers),
    ).rejects.toThrow('Device lost');
  });

  it('lists every MUL_MAT group and the Adreno profiles for the matrix run', () => {
    const ids = PLAMO3_OP_GROUPS.map((group) => group.id);
    for (const id of MUL_MAT_GROUP_IDS) expect(ids).toContain(id);
    expect(ADRENO_OP_PROFILES).toEqual([
      null,
      'adreno-subgroups',
      'adreno-matvec',
      'adreno-sg-tiled',
    ]);
  });
});
