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

import { fetchProviderModels } from '@/lib/providerModels';

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

describe('fetchProviderModels', () => {
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
