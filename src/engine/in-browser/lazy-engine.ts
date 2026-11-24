import { abortError, EngineError, isEngineError, isGarbledOutputError } from '../errors';
import type {
  IdentifiedEngine,
  TranslateOptions,
  TranslateSegment,
  TranslationChunk,
} from '../types';
import { modelPromptFormat } from '../model/catalog';
import type { BackendSelection } from './backend';
import { InBrowserEngine, type InBrowserEngineOptions } from './engine';
import type { LlamaRuntime } from './runtime';

export interface LoadedRuntime {
  runtime: LlamaRuntime;
  selection: BackendSelection;
  /** The catalog model id; picks the prompt format (unset: PLaMo 3). */
  id?: string;
}

export interface LazyInBrowserEngineOptions {
  loadRuntime: (signal: AbortSignal) => Promise<LoadedRuntime>;
  /** Runs once per lost GPU, before anything loads again. */
  onGpuLost?: () => Promise<void>;
  /**
   * Checks finished WebGPU translations. When `isBroken` says one cannot be right, the engine
   * is dropped, `onBroken` moves the runtime to the CPU, and the block is translated again
   * there instead of being shown.
   */
  outputGuard?: OutputGuard;
  config?: InBrowserEngineOptions['config'];
}

export interface OutputGuard {
  isBroken: (source: string, output: string, target: string) => boolean;
  onBroken: () => Promise<void>;
}

/** Thrown inside `translate` when the guard rejects a WebGPU result; never leaves this file. */
class BrokenOutputError extends Error {
  override readonly name = 'BrokenOutputError';
}

/** Enough to walk reload, four slots, CPU, even if every step loses the GPU again. */
const MAX_GPU_RECOVERIES = 3;

export class LazyInBrowserEngine implements IdentifiedEngine {
  readonly kind = 'plamo' as const;

  private engine: InBrowserEngine | null = null;
  private readonly backends = new WeakMap<InBrowserEngine, BackendSelection['backend']>();
  private engineTask: Promise<InBrowserEngine> | null = null;
  private loadController: AbortController | null = null;
  private generation = 0;
  private waiting = 0;
  private recovery: Promise<void> | null = null;

  constructor(private readonly options: LazyInBrowserEngineOptions) {}

  /** Drops the engine after the runtime is released so the next request loads a fresh one. */
  reset(): void {
    this.generation += 1;
    this.loadController?.abort();
    this.loadController = null;
    this.engine = null;
    this.engineTask = null;
  }

  async *translate(
    segments: readonly TranslateSegment[],
    options: TranslateOptions,
  ): AsyncIterable<TranslationChunk> {
    let pending = segments;
    const streamed = new Set<string>();

    for (let recoveries = 0; ; recoveries += 1) {
      const engine = await this.getEngine(options);
      const restarting = new Set(streamed);
      try {
        const guarded = this.guards(engine);
        for await (const chunk of engine.translate(pending, options)) {
          if (chunk.final !== undefined && guarded) {
            const source = pending.find((segment) => segment.id === chunk.id)?.text;
            if (source !== undefined && guarded.isBroken(source, chunk.final, options.tgt)) {
              throw new BrokenOutputError(chunk.id);
            }
          }
          if (chunk.final !== undefined) {
            pending = pending.filter((segment) => segment.id !== chunk.id);
          } else if (restarting.delete(chunk.id)) {
            yield { ...chunk, restart: true };
            continue;
          } else {
            streamed.add(chunk.id);
          }
          yield chunk;
        }
        return;
      } catch (error) {
        // Garbled output that made the worker fail (invalid UTF-8) is broken output, not a
        // lost device: with an output guard it takes the same recovery as flagged output.
        const brokenOutput =
          error instanceof BrokenOutputError ||
          (isGarbledOutputError(error) && this.guards(engine) !== null);
        if (brokenOutput && !options.signal.aborted) {
          if (recoveries >= MAX_GPU_RECOVERIES) throw new EngineError('gpu-lost', error.message);
          await this.recoverBroken(engine);
          continue;
        }
        const lost = isEngineError(error) && error.code === 'gpu-lost';
        if (!lost || options.signal.aborted || recoveries >= MAX_GPU_RECOVERIES) throw error;
        await this.recover(engine);
        if (pending.length === 0) return;
      }
    }
  }

  private guards(engine: InBrowserEngine): OutputGuard | null {
    const guard = this.options.outputGuard;
    return guard && this.backends.get(engine) === 'webgpu' ? guard : null;
  }

  /** Like `recover`, for a GPU that answers but computes wrong results. */
  private recoverBroken(broken: InBrowserEngine): Promise<void> {
    const guard = this.options.outputGuard;
    if (this.engine === broken && guard) {
      this.reset();
      this.recovery = guard.onBroken().finally(() => {
        this.recovery = null;
      });
    }
    return this.recovery ?? Promise.resolve();
  }

  /** Every request running on the lost GPU lands here; only the first one does the work. */
  private recover(lost: InBrowserEngine): Promise<void> {
    if (this.engine === lost && this.options.onGpuLost) {
      this.reset();
      this.recovery = this.options.onGpuLost().finally(() => {
        this.recovery = null;
      });
    }
    return this.recovery ?? Promise.resolve();
  }

  private getEngine(options: TranslateOptions): Promise<InBrowserEngine> {
    if (this.engine) return Promise.resolve(this.engine);

    if (!this.engineTask) {
      const generation = this.generation;
      const controller = new AbortController();
      this.loadController = controller;

      const task = this.createEngine(controller.signal)
        .then((engine) => {
          if (this.generation === generation) this.engine = engine;
          return engine;
        })
        .finally(() => {
          if (this.engineTask === task) this.engineTask = null;
          if (this.loadController === controller) this.loadController = null;
        });

      this.engineTask = task;
    }

    return this.waitFor(this.engineTask, options.signal);
  }

  /**
   * Races the shared load against the caller's signal so one caller giving up does not tear
   * down a load that later callers still need.
   */
  private async waitFor(
    task: Promise<InBrowserEngine>,
    signal: AbortSignal,
  ): Promise<InBrowserEngine> {
    if (signal.aborted) throw abortError();

    this.waiting += 1;
    try {
      return await new Promise<InBrowserEngine>((resolve, reject) => {
        const onAbort = (): void => reject(abortError());
        signal.addEventListener('abort', onAbort, { once: true });

        task.then(
          (engine) => {
            signal.removeEventListener('abort', onAbort);
            resolve(engine);
          },
          (error: unknown) => {
            signal.removeEventListener('abort', onAbort);
            reject(error);
          },
        );
      });
    } finally {
      this.waiting -= 1;
      if (this.waiting === 0 && this.engineTask) this.loadController?.abort();
    }
  }

  private async createEngine(signal: AbortSignal): Promise<InBrowserEngine> {
    await this.recovery;
    const loaded = await this.options.loadRuntime(signal);

    const engine = new InBrowserEngine(loaded.runtime, {
      contextTokens: Math.floor(loaded.selection.nCtx / Math.max(1, loaded.selection.nParallel)),
      format: modelPromptFormat(loaded.id ?? ''),
      ...(this.options.config ? { config: this.options.config } : {}),
    });
    this.backends.set(engine, loaded.selection.backend);
    return engine;
  }
}
