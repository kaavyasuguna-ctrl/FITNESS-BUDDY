#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import copy
import json
from typing import Any, AsyncIterator, Iterator, Literal, overload

import httpx

from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.gateway.models import Models
from ibm_watsonx_ai.gateway.policies import Policies
from ibm_watsonx_ai.gateway.providers import Providers
from ibm_watsonx_ai.gateway.rate_limits import RateLimits
from ibm_watsonx_ai.wml_client_error import InvalidMultipleArguments, WMLClientError
from ibm_watsonx_ai.wml_resource import WMLResource

# Type aliases for gateway inputs and outputs
PromptInput = str | list[str] | list[int]
BatchedPromptInput = list[str | list[str] | list[int]]


def _streaming_create(api_client: APIClient, url: str, request_json: dict) -> Iterator:
    kw_args: dict = dict(
        method="POST",
        url=url,
        json=request_json,
        headers=api_client._get_headers(include_container_id=True),
    )

    if hasattr(api_client.httpx_client, "post_stream"):
        stream_function = api_client.httpx_client.post_stream
    else:
        stream_function = api_client.httpx_client.stream

    with stream_function(**kw_args) as resp:
        if resp.status_code == 200:
            resp_iter = resp.iter_lines()

            for chunk in resp_iter:
                field_name, _, response = chunk.partition(":")

                if response.strip() == "[DONE]":
                    break

                if field_name == "data" and response:
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse {response} as json")
                    yield parsed_response
        else:
            resp.read()
            raise WMLClientError(
                f"Request failed with: {resp.text} ({resp.status_code})"
            )


async def _streaming_acreate(
    api_client: APIClient, url: str, request_json: dict
) -> AsyncIterator:
    kw_args: dict = dict(
        method="POST",
        url=url,
        json=request_json,
        headers=await api_client._aget_headers(include_container_id=True),
    )

    if hasattr(api_client.async_httpx_client, "post_stream"):
        stream_function = api_client.async_httpx_client.post_stream
    else:
        stream_function = api_client.async_httpx_client.stream

    async with stream_function(**kw_args) as resp:
        if resp.status_code == 200:
            resp_iter = resp.aiter_lines()

            async for chunk in resp_iter:
                field_name, _, response = chunk.partition(":")

                if response.strip() == "[DONE]":
                    break

                if field_name == "data" and response:
                    try:
                        parsed_response = json.loads(response)
                    except json.JSONDecodeError:
                        raise Exception(f"Could not parse {response} as json")
                    yield parsed_response

        else:
            await resp.aread()
            raise WMLClientError(
                f"Request failed with: ({resp.text} {resp.status_code})"
            )


class Gateway(WMLResource):
    """Model Gateway class."""

    def __init__(
        self,
        *,
        credentials: Credentials | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
    ):
        if credentials:
            api_client = APIClient(credentials, verify=verify)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        WMLResource.__init__(self, __name__, api_client)

        if self._client.ICP_PLATFORM_SPACES and self._client.CPD_version < 5.2:
            raise WMLClientError("AI Gateway is not supported for this release.")

        self.providers = Providers(self._client)
        self.models = Models(self._client)
        self.policies = Policies(self._client)
        self.rate_limits = RateLimits(self._client)

        # Chat completions
        class _ChatCompletions(WMLResource):
            def __init__(self, api_client: APIClient):
                WMLResource.__init__(self, __name__, api_client)

            @overload
            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> Iterator: ...

            def create(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | Iterator | httpx.Response:
                """Generate chat completions for given model and messages.

                :param model: name of model for given provider or alias
                :type model: str

                :param messages: messages to be processed during call
                :type messages: list[dict]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | Iterator
                """
                request_json = {"messages": messages, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_chat_completions_href()

                if stream:
                    return _streaming_create(
                        api_client=self._client, url=url, request_json=request_json
                    )

                response = self._client.httpx_client.post(
                    url=url,
                    headers=self._client._get_headers(include_container_id=True),
                    json=request_json,
                )

                return self._handle_response(200, "chat completion creation", response)

            @overload
            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> AsyncIterator: ...

            async def acreate(
                self,
                model: str,
                messages: list[dict],
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | AsyncIterator | httpx.Response:
                """Generate chat completions for given model and messages asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param messages: messages to be processed during call
                :type messages: list[dict]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | AsyncIterator
                """
                request_json = {"messages": messages, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_chat_completions_href()

                if stream:
                    return _streaming_acreate(
                        api_client=self._client, url=url, request_json=request_json
                    )

                response = await self._client.async_httpx_client.post(
                    url=url,
                    headers=await self._client._aget_headers(include_container_id=True),
                    json=request_json,
                )

                return self._handle_response(200, "chat completion creation", response)

        class _Chat:
            def __init__(self, api_client: APIClient):
                self.completions = _ChatCompletions(api_client)

        self.chat = _Chat(self._client)

        # Text completions
        class _Completions(WMLResource):
            def __init__(self, api_client: APIClient):
                WMLResource.__init__(self, __name__, api_client)

            @overload
            def create(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            def create(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> Iterator: ...

            def create(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | Iterator:
                """Generate text completions for given model and prompt.

                :param model: name of model for given provider or alias
                :type model: str

                :param prompt: prompt for processing
                :type prompt: str or list[str] or list[int]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | Iterator
                """
                request_json = {"prompt": prompt, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_text_completions_href()

                if stream:
                    return _streaming_create(
                        api_client=self._client, url=url, request_json=request_json
                    )
                else:
                    response = self._client.httpx_client.post(
                        url=url,
                        headers=self._client._get_headers(include_container_id=True),
                        json=request_json,
                    )

                    return self._handle_response(
                        200, "text completion creation", response
                    )

            @overload
            async def acreate(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[False] = False,
                **kwargs: Any,
            ) -> dict: ...

            @overload
            async def acreate(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: Literal[True],
                **kwargs: Any,
            ) -> AsyncIterator: ...

            async def acreate(
                self,
                model: str,
                prompt: PromptInput,
                *,
                stream: bool = False,
                **kwargs: Any,
            ) -> dict | AsyncIterator:
                """Generate text completions for given model and prompt asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param prompt: prompt for processing
                :type prompt: str or list[str] or list[int]

                :param stream: if True will stream the response, defaults to False
                :type stream: bool, optional

                :returns: model answer
                :rtype: dict | AsyncIterator
                """
                request_json = {"prompt": prompt, "model": model, **kwargs}
                if stream:
                    request_json["stream"] = True

                url = self._client._href_definitions.get_gateway_text_completions_href()

                if stream:
                    return _streaming_acreate(
                        api_client=self._client, url=url, request_json=request_json
                    )
                else:
                    response = await self._client.async_httpx_client.post(
                        url=url,
                        headers=await self._client._aget_headers(
                            include_container_id=True
                        ),
                        json=request_json,
                    )

                    return self._handle_response(
                        200, "text completion creation", response
                    )

        self.completions = _Completions(self._client)

        # Embeddings
        class _Embeddings(WMLResource):
            # Maximum number of inputs allowed per request by Model Gateway
            _MAX_BATCH_SIZE = 1000

            def __init__(self, api_client: APIClient):
                WMLResource.__init__(self, __name__, api_client)

            def _batch_inputs(self, inputs: PromptInput) -> BatchedPromptInput:
                """Split input into batches of maximum size.

                :param inputs: inputs to be batched
                :type inputs: str or list[str] or list[int]

                :returns: list of batched inputs
                :rtype: list
                """
                # If input is a string, return it as-is (single batch)
                if isinstance(inputs, str):
                    return [inputs]

                # Validate empty list inputs
                if isinstance(inputs, list) and len(inputs) == 0:
                    return [inputs]

                # If input is a list and within limit, return as single batch
                if len(inputs) <= self._MAX_BATCH_SIZE:
                    return [inputs]

                # Split into batches of _MAX_BATCH_SIZE
                batches: BatchedPromptInput = []
                for i in range(0, len(inputs), self._MAX_BATCH_SIZE):
                    batches.append(inputs[i : i + self._MAX_BATCH_SIZE])
                return batches

            @staticmethod
            def _merge_responses(responses: list[dict]) -> dict:
                """Merge multiple batch responses into a single response.

                :param responses: list of response dictionaries
                :type responses: list[dict]

                :returns: merged response
                :rtype: dict

                :raises WMLClientError: if responses list is empty
                """
                if not responses:
                    raise WMLClientError("Cannot merge empty responses list")

                if len(responses) == 1:
                    return responses[0]

                # Merge all embeddings data
                merged_data = []
                for response in responses:
                    if "data" in response:
                        merged_data.extend(response["data"])

                # Use the first response as template and update data
                merged_response = copy.deepcopy(responses[0])
                merged_response["data"] = merged_data

                # Update usage statistics if present
                if any("usage" in r for r in responses):
                    total_tokens = sum(
                        r.get("usage", {}).get("total_tokens", 0) for r in responses
                    )
                    if "usage" not in merged_response:
                        merged_response["usage"] = {}
                    merged_response["usage"]["total_tokens"] = total_tokens

                return merged_response

            def create(self, model: str, input: PromptInput, **kwargs: Any) -> dict:
                """Generate embeddings for given model and input.

                :param model: name of model for given provider or alias
                :type model: str

                :param input: prompt for processing
                :type input: str or list[str] or list[int]

                :returns: embeddings for given model and input
                :rtype: dict

                :raises WMLClientError: if any batch fails, includes information about successful batches
                """
                batches = self._batch_inputs(input)
                responses = []

                for batch_index, batch in enumerate(batches):
                    try:
                        request_json = {"input": batch, "model": model, **kwargs}

                        response = self._client.httpx_client.post(
                            self._client._href_definitions.get_gateway_embeddings_href(),
                            headers=self._client._get_headers(
                                include_container_id=True
                            ),
                            json=request_json,
                        )

                        batch_response = self._handle_response(
                            200, "embedding creation", response
                        )
                        responses.append(batch_response)
                    except Exception as e:
                        total_batches = len(batches)
                        successful_batches = len(responses)
                        raise WMLClientError(
                            f"Batch {batch_index + 1} of {total_batches} failed during embedding creation. "
                            f"Successfully processed {successful_batches} batch(es) before failure. "
                            f"Original error: {str(e)}"
                        ) from e

                return self._merge_responses(responses)

            async def acreate(
                self, model: str, input: PromptInput, **kwargs: Any
            ) -> dict:
                """Generate embeddings for given model and input asynchronously.

                :param model: name of model for given provider or alias
                :type model: str

                :param input: prompt for processing
                :type input: str or list[str] or list[int]

                :returns: embeddings for given model and input
                :rtype: dict

                :raises WMLClientError: if any batch fails, includes information about successful batches
                """
                batches = self._batch_inputs(input)
                responses = []

                for batch_index, batch in enumerate(batches):
                    try:
                        request_json = {"input": batch, "model": model, **kwargs}

                        response = await self._client.async_httpx_client.post(
                            self._client._href_definitions.get_gateway_embeddings_href(),
                            headers=await self._client._aget_headers(
                                include_container_id=True
                            ),
                            json=request_json,
                        )

                        batch_response = self._handle_response(
                            200, "embedding creation", response
                        )
                        responses.append(batch_response)
                    except Exception as e:
                        total_batches = len(batches)
                        successful_batches = len(responses)
                        raise WMLClientError(
                            f"Batch {batch_index + 1} of {total_batches} failed during embedding creation. "
                            f"Successfully processed {successful_batches} batch(es) before failure. "
                            f"Original error: {str(e)}"
                        ) from e

                return self._merge_responses(responses)

        self.embeddings = _Embeddings(self._client)
