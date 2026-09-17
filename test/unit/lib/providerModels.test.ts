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

import { fetchProviderModels } from '@/lib/providerModels';
import { afterEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({ fetchPost: vi.fn() }));

vi.mock('@/api/http', () => ({ fetchPost: mocks.fetchPost }));

afterEach(() => mocks.fetchPost.mockReset());

describe('fetchProviderModels errors', () => {
  it.each([
    [401, 'Invalid API key. Check your API key and click Refresh again.'],
    [
      403,
      'Access denied. Check your API key permissions and account access, then click Refresh again.',
    ],
    [
      500,
      'Could not load models. Check your connection and API host, then click Refresh again.',
    ],
  ])(
    'explains HTTP %s without exposing credentials',
    async (status, message) => {
      mocks.fetchPost.mockRejectedValue(
        Object.assign(new Error('backend error'), { status })
      );
      await expect(
        fetchProviderModels('https://example.com/v1', '/models', 'bad-key')
      ).rejects.toThrow(message);
    }
  );

  it('can fetch models after a rejected key is corrected', async () => {
    mocks.fetchPost
      .mockRejectedValueOnce(
        Object.assign(new Error('backend error'), { status: 401 })
      )
      .mockResolvedValueOnce({ data: [{ id: 'ling-chat' }] });
    await expect(
      fetchProviderModels('https://example.com/v1', '/models', 'bad-key')
    ).rejects.toThrow('Invalid API key');
    await expect(
      fetchProviderModels('https://example.com/v1', '/models', 'new-key')
    ).resolves.toMatchObject([{ models: [{ id: 'ling-chat' }] }]);
    expect(mocks.fetchPost).toHaveBeenLastCalledWith('/model/list', {
      api_host: 'https://example.com/v1',
      models_endpoint: '/models',
      api_key: 'new-key',
    });
  });
});

describe('fetchProviderModels filtering', () => {
  it('keeps text-in / text-out models from a `modalities` listing', async () => {
    mocks.fetchPost.mockResolvedValue({ data: [
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
    ] });

    const groups = await fetchProviderModels(
      'https://api.aimlapi.com/v1',
      '/models',
      'k'
    );

    expect(groups).toEqual([
      { provider: 'openai', models: [{ id: 'openai/gpt-4o-mini' }] },
    ]);
  });

  it('lists an id once when the listing repeats it', async () => {
    mocks.fetchPost.mockResolvedValue({ data: [
      {
        id: 'anthropic/claude-sonnet-4.5',
        type: 'openai/chat-completions',
        modalities: { input: ['text'], output: ['text'] },
      },
      {
        id: 'anthropic/claude-sonnet-4.5',
        type: 'openai/chat-completions',
        modalities: { input: ['text'], output: ['text'] },
      },
    ] });

    const groups = await fetchProviderModels(
      'https://api.aimlapi.com/v1',
      '/models',
      'k'
    );

    expect(groups).toHaveLength(1);
    expect(groups[0].models).toHaveLength(1);
  });

  it('drops a model published only behind a non-chat endpoint surface', async () => {
    mocks.fetchPost.mockResolvedValue({ data: [
      // Responses-API only. Verified live 2026-09-03: a /chat/completions
      // call for this id answers 404 "Model not found", so offering it in
      // the dropdown hands the user an id that cannot work.
      {
        id: 'openai/gpt-5-2-pro',
        type: 'openai/responses/submit',
        modalities: { input: ['document', 'text'], output: ['text'] },
      },
      // Published on both surfaces, so it is reachable and must stay.
      {
        id: 'openai/gpt-5-5',
        type: 'openai/chat-completions',
        modalities: { input: ['image', 'text'], output: ['text'] },
      },
      {
        id: 'openai/gpt-5-5',
        type: 'openai/responses/submit',
        modalities: { input: ['document', 'image', 'text'], output: ['text'] },
      },
      // Anthropic models list a messages row first; the chat-completions row
      // is what keeps them. Verified live: this id answers 200 on
      // /chat/completions despite advertising only `streaming`.
      {
        id: 'anthropic/claude-opus-5',
        type: 'anthropic/messages',
        modalities: { input: ['text'], output: ['text'] },
      },
      {
        id: 'anthropic/claude-opus-5',
        type: 'openai/chat-completions',
        modalities: { input: ['text'], output: ['text'] },
      },
      {
        id: 'openai/text-embedding-3-small',
        type: 'openai/embeddings',
        modalities: { input: ['text'], output: ['text'] },
      },
    ] });

    const groups = await fetchProviderModels(
      'https://api.aimlapi.com/v1',
      '/models',
      'k'
    );

    expect(groups).toEqual([
      {
        provider: 'anthropic',
        models: [{ id: 'anthropic/claude-opus-5' }],
      },
      { provider: 'openai', models: [{ id: 'openai/gpt-5-5' }] },
    ]);
  });

  it('ignores a `type` that is not an endpoint surface name', async () => {
    // A single-word `type` is not the `<family>/<endpoint>` surface shape,
    // so it must not be read as one and must not filter anything out.
    mocks.fetchPost.mockResolvedValue({ data: [{ id: 'some-model', type: 'model' }] });

    const groups = await fetchProviderModels(
      'https://api.tokenfactory.nebius.com/v1',
      '/models',
      'k'
    );

    expect(groups).toEqual([
      { provider: 'other', models: [{ id: 'some-model' }] },
    ]);
  });

  it('still keeps listings that publish no modality metadata at all', async () => {
    mocks.fetchPost.mockResolvedValue({ data: [{ id: 'deepseek-reasoner' }] });

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
