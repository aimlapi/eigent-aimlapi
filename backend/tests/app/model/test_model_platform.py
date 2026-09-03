# ========= Copyright 2025-2026 @ Eigent.ai All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2025-2026 @ Eigent.ai All Rights Reserved. =========

import re

import httpx
import pytest
from camel.models import ModelFactory
from camel.models.openai_compatible_model import OpenAICompatibleModel
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from app.model.model_platform import (
    AIMLAPI_ATTRIBUTION_HEADERS,
    NormalizedModelPlatform,
    NormalizedOptionalModelPlatform,
    aimlapi_attribution_headers,
    is_aimlapi_endpoint,
    is_eigent_cloud_model_endpoint,
    normalize_model_platform,
    normalize_optional_model_platform,
    resolve_cloud_model_runtime_platform,
)


def test_normalize_model_platform_maps_known_aliases():
    assert normalize_model_platform("grok") == "openai-compatible-model"
    assert normalize_model_platform("z.ai") == "zhipuai"
    assert normalize_model_platform("ModelArk") == "openai-compatible-model"
    assert normalize_model_platform("ernie") == "qianfan"
    assert normalize_model_platform("llama.cpp") == "openai-compatible-model"
    assert normalize_model_platform("nebius") == "openai-compatible-model"
    assert normalize_model_platform("aimlapi") == "openai-compatible-model"


def test_aimlapi_partner_id_matches_gateway_contract():
    """A malformed partner id is dropped silently and earns nothing."""
    assert re.fullmatch(
        r"part_[A-Za-z0-9]{1,64}",
        AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Partner-ID"],
    )
    assert re.fullmatch(
        r"(web|agent|mcp)/[a-z0-9-]{1,32}",
        AIMLAPI_ATTRIBUTION_HEADERS["X-AIMLAPI-Source"],
    )


def test_aimlapi_referer_and_title_identify_the_calling_app():
    assert (
        AIMLAPI_ATTRIBUTION_HEADERS["HTTP-Referer"]
        == "https://github.com/eigent-ai/eigent"
    )
    assert AIMLAPI_ATTRIBUTION_HEADERS["X-Title"] == "Eigent"


def test_is_aimlapi_endpoint_matches_host_not_substring():
    assert is_aimlapi_endpoint("https://api.aimlapi.com/v1")
    assert is_aimlapi_endpoint("api.aimlapi.com/v1")
    assert not is_aimlapi_endpoint("https://api.aimlapi.com.evil.test/v1")
    assert not is_aimlapi_endpoint("https://proxy.example.com/api.aimlapi.com")
    assert not is_aimlapi_endpoint("https://openrouter.ai/api/v1")
    assert not is_aimlapi_endpoint(None)
    assert not is_aimlapi_endpoint("")


def test_aimlapi_attribution_is_scoped_to_aimlapi_requests():
    assert aimlapi_attribution_headers("https://api.openai.com/v1") is None
    assert aimlapi_attribution_headers("https://openrouter.ai/api/v1") is None

    headers = aimlapi_attribution_headers("https://api.aimlapi.com/v1")
    assert headers == AIMLAPI_ATTRIBUTION_HEADERS


def test_aimlapi_attribution_merges_and_never_mutates_the_constant():
    original = dict(AIMLAPI_ATTRIBUTION_HEADERS)

    headers = aimlapi_attribution_headers(
        "https://api.aimlapi.com/v1",
        {"X-Title": "user override", "X-Custom": "kept"},
    )

    # A caller's own headers survive, and win on a key clash.
    assert headers["X-Custom"] == "kept"
    assert headers["X-Title"] == "user override"
    assert headers["X-AIMLAPI-Partner-ID"] == original["X-AIMLAPI-Partner-ID"]

    headers["X-AIMLAPI-Partner-ID"] = "mutated"
    assert AIMLAPI_ATTRIBUTION_HEADERS == original


def test_normalize_model_platform_keeps_non_alias_unchanged():
    assert normalize_model_platform("openai") == "openai"
    assert normalize_model_platform("mistral") == "mistral"


def test_normalize_optional_model_platform_handles_none():
    assert normalize_optional_model_platform(None) is None


def test_normalized_model_platform_type_applies_in_pydantic_model():
    class _Model(BaseModel):
        model_platform: NormalizedModelPlatform
        optional_model_platform: NormalizedOptionalModelPlatform = None

    item = _Model(
        model_platform="ernie",
        optional_model_platform="ModelArk",
    )

    assert item.model_platform == "qianfan"
    assert item.optional_model_platform == "openai-compatible-model"


def test_eigent_cloud_azure_responses_use_openai_compatible_transport():
    assert is_eigent_cloud_model_endpoint("https://proxy.eigent.ai")
    assert (
        resolve_cloud_model_runtime_platform(
            model_platform="azure",
            api_url="https://proxy.eigent.ai",
            api_mode="responses",
        )
        == "openai-compatible-model"
    )


def test_cloud_chat_and_direct_azure_responses_keep_azure_transport():
    assert (
        resolve_cloud_model_runtime_platform(
            model_platform="azure",
            api_url="https://proxy.eigent.ai",
            api_mode="chat_completions",
        )
        == "azure"
    )
    assert (
        resolve_cloud_model_runtime_platform(
            model_platform="azure",
            api_url="https://customer-resource.openai.azure.com",
            api_mode="responses",
        )
        == "azure"
    )


@pytest.mark.asyncio
async def test_cloud_responses_runtime_calls_standard_responses_route():
    runtime_platform = resolve_cloud_model_runtime_platform(
        model_platform="azure",
        api_url="https://proxy.eigent.ai",
        api_mode="responses",
    )
    requested_paths: list[str] = []

    async def reject_after_recording(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(
            400,
            request=request,
            json={"error": {"message": "test stop", "type": "test_error"}},
        )

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(reject_after_recording)
    )
    responses_client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://proxy.eigent.ai",
        max_retries=0,
        http_client=http_client,
    )

    backend = ModelFactory.create(
        model_platform=runtime_platform,
        model_type="gpt-5.7-future",
        api_key="test-key",
        url="https://proxy.eigent.ai",
        api_mode="responses",
        async_client=responses_client,
    )

    assert isinstance(backend, OpenAICompatibleModel)
    assert backend._api_mode == "responses"
    with pytest.raises(BadRequestError, match="test stop"):
        await backend._async_client.responses.create(
            model="gpt-5.7-future",
            input="hello",
        )

    assert requested_paths == ["/responses"]
    await responses_client.close()
