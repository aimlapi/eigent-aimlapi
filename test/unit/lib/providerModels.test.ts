// ========= Copyright 2025-2026 @ Eigent.ai All Rights Reserved. =========
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ========= Copyright 2025-2026 @ Eigent.ai All Rights Reserved. =========

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  attributionHeadersForUrl,
  fetchProviderModels,
} from '@/lib/providerModels';

function mockModelsResponse(data: unknown[]) {
  const fetchMock = vi.fn(async () => ({
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => ({ object: 'list', data }),
  }));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('attributionHeadersForUrl', () => {
  it('sends a partner id the aimlapi.com gateway can parse', () => {
    const headers = attributionHeadersForUrl('https://api.aimlapi.com/models');

    // A malformed partner id is dropped silently by the gateway and earns
    // nothing, so the shape is asserted rather than trusted.
    expect(headers['X-AIMLAPI-Partner-ID']).toMatch(/^part_[A-Za-z0-9]{1,64}$/);
    expect(headers['X-AIMLAPI-Source']).toMatch(
      /^(web|agent|mcp)\/[a-z0-9-]{1,32}$/
    );
    // HTTP-Referer / X-Title identify Eigent, not the vendor.
    expect(headers['HTTP-Referer']).toBe('https://github.com/eigent-ai/eigent');
    expect(headers['X-Title']).toBe('Eigent');
  });

  it('never attaches attribution to another vendor or a look-alike host', () => {
    expect(
      attributionHeadersForUrl('https://openrouter.ai/api/v1/models')
    ).toEqual({});
    expect(
      attributionHeadersForUrl('https://api.aimlapi.com.evil.test/models')
    ).toEqual({});
    expect(
      attributionHeadersForUrl('https://proxy.example.com/api.aimlapi.com')
    ).toEqual({});
    expect(attributionHeadersForUrl('not a url')).toEqual({});
  });

  it('returns a fresh object so the shared table cannot be mutated', () => {
    const first = attributionHeadersForUrl('https://api.aimlapi.com/models');
    first['X-AIMLAPI-Partner-ID'] = 'mutated';

    const second = attributionHeadersForUrl('https://api.aimlapi.com/models');
    expect(second['X-AIMLAPI-Partner-ID']).not.toBe('mutated');
  });
});

describe('fetchProviderModels', () => {
  it('attaches attribution alongside the caller headers for aimlapi.com', async () => {
    const fetchMock = mockModelsResponse([]);

    await fetchProviderModels('https://api.aimlapi.com/v1', '/models', 'k');

    const headers = (fetchMock.mock.calls[0] as any)[1].headers;
    expect(headers.Authorization).toBe('Bearer k');
    expect(headers.Accept).toBe('application/json');
    expect(headers['X-AIMLAPI-Source']).toBe('agent/eigent');
  });

  it('leaves other providers request headers untouched', async () => {
    const fetchMock = mockModelsResponse([]);

    await fetchProviderModels('https://openrouter.ai/api/v1', '/models', 'k');

    const headers = (fetchMock.mock.calls[0] as any)[1].headers;
    expect(Object.keys(headers).sort()).toEqual(['Accept', 'Authorization']);
  });

  it('keeps text-in / text-out models from a `modalities` listing', async () => {
    mockModelsResponse([
      {
        id: 'openai/gpt-4o-mini',
        modalities: { input: ['image', 'text'], output: ['text'] },
      },
      // Image generation: text in, image out.
      {
        id: 'flux/schnell',
        modalities: { input: ['text'], output: ['image'] },
      },
      // Speech to text: audio in, text out.
      {
        id: 'deepgram/nova-3',
        modalities: { input: ['audio'], output: ['text'] },
      },
    ]);

    const groups = await fetchProviderModels(
      'https://api.aimlapi.com/v1',
      '/models',
      'k'
    );

    expect(groups).toEqual([
      { provider: 'openai', models: [{ id: 'openai/gpt-4o-mini' }] },
    ]);
  });

  it('lists an id once when it appears under several endpoint surfaces', async () => {
    mockModelsResponse([
      {
        id: 'anthropic/claude-sonnet-4.5',
        type: 'openai/chat-completions',
        modalities: { input: ['text'], output: ['text'] },
      },
      {
        id: 'anthropic/claude-sonnet-4.5',
        type: 'anthropic/messages',
        modalities: { input: ['text'], output: ['text'] },
      },
    ]);

    const groups = await fetchProviderModels(
      'https://api.aimlapi.com/v1',
      '/models',
      'k'
    );

    expect(groups).toHaveLength(1);
    expect(groups[0].models).toHaveLength(1);
  });

  it('still keeps listings that publish no modality metadata at all', async () => {
    mockModelsResponse([{ id: 'deepseek-reasoner' }]);

    const groups = await fetchProviderModels(
      'https://api.tokenfactory.nebius.com/v1',
      '/models',
      'k'
    );

    expect(groups).toEqual([
      { provider: 'other', models: [{ id: 'deepseek-reasoner' }] },
    ]);
  });
});
