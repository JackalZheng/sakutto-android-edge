import { describe, expect, it } from 'vitest';
import { isAllowedRequest } from '../e2e/extension';
import { MODEL_LIST } from '@/engine/model/catalog';

describe('extension network allowlist', () => {
  it('allows the Hugging Face redirect used by the pinned model', () => {
    expect(
      isAllowedRequest(
        'https://huggingface.co/pfnet/plamo-3-translate-0.8b-GGUF/resolve/dbeb0856cb0b004b484f190bb71dd65876297013/plamo-3-translate-0.8b-Q4_K_M.gguf',
      ),
    ).toBe(true);
    expect(isAllowedRequest('https://us.aws.cdn.hf.co/xet-bridge-us/file')).toBe(true);
    expect(isAllowedRequest('https://cdn-lfs-us-1.hf.co/repos/file')).toBe(true);
    expect(isAllowedRequest('https://cdn-lfs.huggingface.co/repos/file')).toBe(true);
    expect(isAllowedRequest('https://example.com/model.gguf')).toBe(false);
  });

  it('allows every pinned catalog model URL (all six models)', () => {
    expect(MODEL_LIST).toHaveLength(6);
    for (const model of MODEL_LIST) {
      expect(model.url.startsWith('https://huggingface.co/'), model.id).toBe(true);
      expect(isAllowedRequest(model.url), model.id).toBe(true);
    }
  });
});
