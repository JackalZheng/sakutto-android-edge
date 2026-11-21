export type EngineErrorCode =
  | 'invalid-input'
  | 'unsupported-language'
  | 'model-missing'
  | 'model-integrity'
  | 'model-load-failed'
  | 'gpu-unavailable'
  | 'gpu-lost'
  | 'download-failed'
  | 'storage-insufficient'
  | 'chrome-unavailable'
  | 'chrome-not-ready'
  | 'offscreen-unavailable'
  | 'aborted'
  | 'unknown';

export class EngineError extends Error {
  override readonly name = 'EngineError';
  readonly retryable: boolean;

  constructor(
    readonly code: EngineErrorCode,
    message: string,
    options?: { cause?: unknown; retryable?: boolean; detail?: string },
  ) {
    super(message);
    if (options && 'cause' in options) this.cause = options.cause;
    this.retryable = options?.retryable ?? true;
    if (typeof options?.detail === 'string') this.detail = options.detail;
  }

  readonly detail?: string;
}

export function isEngineError(value: unknown): value is EngineError {
  return value instanceof EngineError;
}

/**
 * `EngineError.detail` of a 'gpu-lost' error whose real cause is garbled GPU output: the
 * tokens decoded to invalid UTF-8 and the worker failed on them. Recovery treats it like
 * output the guard flagged as broken, not like a lost device.
 */
export const GARBLED_OUTPUT_DETAIL = 'garbled-output';

export function isGarbledOutputError(value: unknown): value is EngineError {
  return isEngineError(value) && value.detail === GARBLED_OUTPUT_DETAIL;
}

const ENGINE_ERROR_CODES: ReadonlySet<string> = new Set<EngineErrorCode>([
  'invalid-input',
  'unsupported-language',
  'model-missing',
  'model-integrity',
  'model-load-failed',
  'gpu-unavailable',
  'gpu-lost',
  'download-failed',
  'storage-insufficient',
  'chrome-unavailable',
  'chrome-not-ready',
  'offscreen-unavailable',
  'aborted',
  'unknown',
]);

export function isEngineErrorCode(value: unknown): value is EngineErrorCode {
  return typeof value === 'string' && ENGINE_ERROR_CODES.has(value);
}

export function isAbortError(value: unknown): boolean {
  if (value instanceof EngineError) return value.code === 'aborted';
  if (typeof value !== 'object' || value === null) return false;
  const name = 'name' in value && typeof value.name === 'string' ? value.name : '';
  return name === 'AbortError' || name === 'WllamaAbortError';
}

export function abortError(message = 'The operation was aborted.'): EngineError {
  return new EngineError('aborted', message);
}

export function toEngineError(value: unknown): EngineError {
  if (isEngineError(value)) return value;
  if (isAbortError(value)) return abortError();
  const message = value instanceof Error ? value.message : String(value);
  return new EngineError('unknown', message, { cause: value });
}
