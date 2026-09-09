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

import i18next from 'i18next';

/**
 * Fetch + parse helper for cloud providers that expose an OpenAI-compatible
 * `/v1/models` listing endpoint (e.g. OrcaRouter). Returns chat-capable
 * models grouped by their `<provider>/<model>` prefix so the UI can render
 * provider tabs.
 */

/** Single model entry as returned by an OpenAI-compatible /v1/models call. */
type RawModel = {
  id: string;
  architecture?: {
    input_modalities?: string[] | null;
    output_modalities?: string[] | null;
  };
  /**
   * Alternative modality shape used by listings that do not publish
   * OpenRouter's `architecture` object (e.g. aimlapi.com).
   */
  modalities?: {
    input?: string[] | null;
    output?: string[] | null;
  };
  /**
   * Endpoint surface this row describes, on listings that publish one
   * row per surface (e.g. aimlapi.com). Absent on OpenRouter-shaped and
   * plain OpenAI-shaped listings.
   */
  type?: string;
  context_length?: number;
  max_completion_tokens?: number;
};

export type ProviderModelInfo = {
  id: string;
  contextLength?: number;
  maxCompletionTokens?: number;
};

export type ProviderModelGroup = {
  provider: string;
  models: ProviderModelInfo[];
};

/**
 * Decide whether a model is chat-capable enough to surface in the dropdown.
 * Keeps models that explicitly emit text, plus models that declare no
 * modality metadata at all (some upstream listings — e.g. deepseek-reasoner
 * — leave it null even though they are usable for chat).
 *
 * Filters out: TTS / image-only / video-only outputs, and — for listings that
 * use the `modalities` shape — transcription / OCR entries that emit text but
 * cannot accept a text prompt.
 */
function isChatCapable(model: RawModel): boolean {
  const arch = model.architecture;
  if (arch) {
    const out = arch.output_modalities;
    if (out == null) return true;
    return out.includes('text');
  }

  const modalities = model.modalities;
  if (!modalities) return true;
  const { input, output } = modalities;
  if (output != null && !output.includes('text')) return false;
  if (input != null && !input.includes('text')) return false;
  return true;
}

/**
 * Attribution headers keyed by request origin. `HTTP-Referer` / `X-Title`
 * follow the OpenRouter convention and identify Eigent as the calling
 * application; the `X-AIMLAPI-*` pair is read by aimlapi.com to attribute
 * traffic to this integration. Keying on the resolved origin — rather than on
 * the configured provider id — keeps one vendor's headers off another vendor's
 * request, including a proxy that merely fronts the same API.
 */
const ATTRIBUTION_HEADERS_BY_ORIGIN: Record<string, Record<string, string>> = {
  'https://api.aimlapi.com': {
    'HTTP-Referer': 'https://github.com/eigent-ai/eigent',
    'X-Title': 'Eigent',
    'X-AIMLAPI-Partner-ID': 'part_kK5bWvwrYl5A9aWdwLFoIBQV',
    'X-AIMLAPI-Source': 'agent/eigent',
  },
};

/** Attribution headers for `url`, or an empty object for unknown origins. */
export function attributionHeadersForUrl(url: string): Record<string, string> {
  let origin: string;
  try {
    origin = new URL(url).origin;
  } catch {
    return {};
  }
  // Spread so the shared table is never handed out by reference.
  return { ...(ATTRIBUTION_HEADERS_BY_ORIGIN[origin] ?? {}) };
}

/**
 * The one endpoint surface this client speaks. Everything below goes through
 * `POST /chat/completions`.
 */
const CHAT_COMPLETIONS_SURFACE = 'openai/chat-completions';

/**
 * Decide whether a listing row describes an endpoint this client can call.
 *
 * A listing that publishes one row per endpoint surface names it in `type`
 * (`openai/chat-completions`, `openai/responses/submit`, `anthropic/messages`,
 * `openai/embeddings`, …). Only the chat-completions surface can serve us: a
 * model published solely behind `openai/responses/submit` answers
 * `404 Model not found` on `/chat/completions`, so offering it in the dropdown
 * hands the user an id that cannot work. Verified live against aimlapi.com on
 * 2026-09-03: `openai/gpt-5-2-pro` (responses-only) 404s, while
 * `anthropic/claude-opus-5` — which also publishes a chat-completions row —
 * answers 200.
 *
 * A surface name is recognised by its `<family>/<endpoint>` shape. Listings
 * that do not describe surfaces at all (OpenRouter's, and the plain OpenAI
 * `/v1/models` shape used by the other providers here) carry no `type`, or
 * carry an unrelated single-word value, and are left untouched.
 */
function declaresNonChatEndpoint(model: RawModel): boolean {
  const type = model.type;
  if (typeof type !== 'string' || !type.includes('/')) return false;
  return type !== CHAT_COMPLETIONS_SURFACE;
}

/** Split `anthropic/claude-opus-4.6` into `["anthropic", "claude-opus-4.6"]`. */
function splitProviderPrefix(id: string): [string, string] {
  const idx = id.indexOf('/');
  if (idx <= 0) return ['', id];
  return [id.slice(0, idx), id.slice(idx + 1)];
}

/**
 * Hit `${apiHost}${modelsEndpoint}` with a Bearer token and return chat-capable
 * models grouped by provider prefix, sorted alphabetically by provider, with
 * models within each group sorted alphabetically by id.
 *
 * Throws on network failure or non-2xx response with a user-readable message.
 */
export async function fetchProviderModels(
  apiHost: string,
  modelsEndpoint: string,
  apiKey: string
): Promise<ProviderModelGroup[]> {
  if (!apiKey) {
    throw new Error(
      i18next.t('setting.models-api-key-required', {
        defaultValue: 'API key is required to fetch the model list.',
      })
    );
  }
  const trimmedHost = apiHost.replace(/\/+$/, '');
  const url = `${trimmedHost}${modelsEndpoint}`;

  const response = await fetch(url, {
    method: 'GET',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: 'application/json',
      ...attributionHeadersForUrl(url),
    },
  });

  if (!response.ok) {
    throw new Error(
      i18next.t('setting.models-fetch-failed', {
        defaultValue: 'Failed to fetch models: {{status}} {{statusText}}',
        status: response.status,
        statusText: response.statusText,
      })
    );
  }
  const payload = await response.json();
  const data: RawModel[] = Array.isArray(payload?.data) ? payload.data : [];

  const grouped = new Map<string, ProviderModelInfo[]>();
  // One id can be listed several times when a provider publishes the same
  // model under more than one endpoint surface; the dropdown must show it once.
  const seen = new Set<string>();
  for (const model of data) {
    if (!model?.id || !isChatCapable(model)) continue;
    if (declaresNonChatEndpoint(model)) continue;
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    const [provider] = splitProviderPrefix(model.id);
    const bucket = provider || 'other';
    const info: ProviderModelInfo = {
      id: model.id,
      contextLength: model.context_length,
      maxCompletionTokens: model.max_completion_tokens,
    };
    const arr = grouped.get(bucket);
    if (arr) arr.push(info);
    else grouped.set(bucket, [info]);
  }

  const groups: ProviderModelGroup[] = Array.from(grouped.entries())
    .map(([provider, models]) => ({
      provider,
      models: models.sort((a, b) => a.id.localeCompare(b.id)),
    }))
    .sort((a, b) => a.provider.localeCompare(b.provider));

  return groups;
}

/** localStorage cache helpers — keyed per provider id to keep entries small. */
const CACHE_KEY_PREFIX = 'eigent-provider-models-v1:';

export function loadCachedModels(
  providerId: string
): ProviderModelGroup[] | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY_PREFIX + providerId);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return parsed as ProviderModelGroup[];
  } catch {
    return null;
  }
}

export function saveCachedModels(
  providerId: string,
  groups: ProviderModelGroup[]
): void {
  try {
    localStorage.setItem(CACHE_KEY_PREFIX + providerId, JSON.stringify(groups));
  } catch {
    // localStorage may be unavailable (quota / private mode); silently ignore.
  }
}
