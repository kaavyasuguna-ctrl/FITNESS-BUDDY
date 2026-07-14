#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import asyncio
import json as js
import os
import queue
import ssl
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from random import random
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, TypeVar

import httpx
from httpx._utils import get_environment_proxies

from ibm_watsonx_ai._wrappers.httpx import (
    GlobalHttpxSettings,
    retry_transport_factory,
)
from ibm_watsonx_ai.wml_client_error import WMLClientError

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient


HTTPX_DEFAULT_TIMEOUT = httpx.Timeout(timeout=30 * 60, connect=10)

HTTPX_KEEPALIVE_EXPIRY = 5
HTTPX_DEFAULT_LIMIT = httpx.Limits(
    max_connections=10,
    max_keepalive_connections=10,
    keepalive_expiry=HTTPX_KEEPALIVE_EXPIRY,
)
DEFAULT_RETRY_STATUS_CODES = [429, 503, 504, 520]
MAX_RETRY_DELAY = 8
DEFAULT_DELAY = 0.5

_MAX_RETRIES = 10  # number of retries after the first failure
REMAINING_LIMIT_HEADER = "x-requests-limit-remaining"

RETRY_CONFIG: dict = {
    "retries": 3,
    "backoff_factor": 0.3,
    "status_forcelist": (401, 500, 502, 503, 504, 520, 521, 524),
}


def set_verify_for_httpx(func: Callable) -> Callable:
    """
    This decorator passes through the function with verify parameter from environment and global verify.
    Priority order: environment variable > global verify > default (True)
    """

    @wraps(func)
    def wrapper(*args: Any, **kw: Any) -> Any:
        # Use the centralized function to get effective verify value
        effective_verify = GlobalHttpxSettings.get_effective_verify()

        if "verify" not in kw:
            kw.update({"verify": effective_verify})

        return func(*args, **kw)

    return wrapper


@GlobalHttpxSettings.inject_settings
def _get_httpx_client_with_config(
    api_client: APIClient,
    httpx_client_cls: type[HClient],
    is_async: bool,
    limits: httpx.Limits,
    timeout: httpx.Timeout,
    proxies: dict[str, str | None] | None = None,
    **kwargs: Any,
) -> HTTPXClient | HTTPXAsyncClient:
    def correct_key(key: str) -> str:
        if key in ["http", "https"]:
            return key + "://"

        return key

    if proxies or (proxies := get_environment_proxies()):
        # if no proxies were passed, check proxies setting from environment
        transport = None
        mounts = {
            correct_key(key): retry_transport_factory(
                is_async, api_client, limits=limits, proxy=value
            )
            for key, value in proxies.items()
        }
    else:
        # no proxies, no problem, usual transport is initialised
        transport = retry_transport_factory(is_async, api_client, limits)
        mounts = None

    # Get verify from transport if it's a RetryTransport
    if transport:
        t = transport
    elif mounts:  # mounts exists and exist at least one in list
        t = list(mounts.items())[0][1]
    else:
        raise WMLClientError("No transport object passed")

    verify = t.effective_verify

    return httpx_client_cls(
        transport=transport, mounts=mounts, timeout=timeout, verify=verify
    )


def _get_httpx_client(
    api_client: APIClient,
    limits: httpx.Limits = HTTPX_DEFAULT_LIMIT,
    timeout: httpx.Timeout = HTTPX_DEFAULT_TIMEOUT,
) -> HTTPXClient:
    return _get_httpx_client_with_config(
        api_client, HTTPXClient, False, limits, timeout
    )


def _get_async_httpx_client(
    api_client: APIClient,
    limits: httpx.Limits = HTTPX_DEFAULT_LIMIT,
    timeout: httpx.Timeout = HTTPX_DEFAULT_TIMEOUT,
) -> HTTPXAsyncClient:
    return _get_httpx_client_with_config(
        api_client, HTTPXAsyncClient, True, limits, timeout
    )


class HTTPXClient(httpx.Client):
    """Wrapper for httpx Sync Client"""

    def __init__(
        self, verify: ssl.SSLContext | str | bool | None = None, **kwargs: Any
    ):
        # Remove proxies from kwargs as they should be handled via transport or mounts
        kwargs.pop("proxies", None)
        super().__init__(
            verify=verify if verify is not None else bool(verify),
            timeout=kwargs.pop("timeout", None) or HTTPX_DEFAULT_TIMEOUT,
            limits=kwargs.pop("limits", None) or HTTPX_DEFAULT_LIMIT,
            **kwargs,
        )

    def post(  # type: ignore[override]
        self,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers is not None and headers.get("Content-Type") is not None:
                headers["Content-Type"] = "application/json"

        response = super().post(
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        )
        return response

    @contextmanager
    def post_stream(
        self,
        method: str,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> Iterator[httpx.Response]:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers is not None and headers.get("Content-Type") is not None:
                headers["Content-Type"] = "application/json"

        with super().stream(
            method=method,
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        ) as response:
            try:
                yield response
            finally:
                response.close()

    def __del__(self) -> None:
        try:
            # Closing the connection pool when the object is deleted
            self.close()
        except Exception:
            pass


class HTTPXAsyncClient(httpx.AsyncClient):
    def __init__(
        self, verify: ssl.SSLContext | str | bool | None = None, **kwargs: Any
    ):
        # Remove proxies from kwargs as they should be handled via transport or mounts
        kwargs.pop("proxies", None)
        super().__init__(
            verify=verify if verify is not None else bool(verify),
            timeout=kwargs.pop("timeout", None) or HTTPX_DEFAULT_TIMEOUT,
            limits=kwargs.pop("limits", None) or HTTPX_DEFAULT_LIMIT,
            **kwargs,
        )

    async def post(  # type: ignore[override]
        self,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers and not headers.get("Content-Type"):
                headers["Content-Type"] = "application/json"

        response = await super().post(
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        )
        return response

    @asynccontextmanager
    async def post_stream(
        self,
        method: str,
        url: str,
        *,
        content: str | bytes | None = None,
        json: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[httpx.Response]:
        if json is not None and content is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            content = js.dumps(json, cls=NumpyTypeEncoder)

            if headers is not None and headers.get("Content-Type") is not None:
                headers["Content-Type"] = "application/json"

        async with super().stream(
            method=method,
            url=url,
            content=content,
            headers=headers,
            params=params,
            **kwargs,
        ) as response:
            try:
                yield response
            finally:
                await response.aclose()

    def __del__(self) -> None:
        try:
            # Closing the connection pool when the object is deleted
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:
            pass


HClient = TypeVar("HClient", HTTPXClient, HTTPXAsyncClient)


def backoff_timeout(wx_delay_time: float, attempt: int) -> float:
    jitter = 1 + 0.25 * random()
    sleep_seconds = min(wx_delay_time * pow(2.0, attempt), MAX_RETRY_DELAY)
    return sleep_seconds * jitter


def _get_max_retries(
    instance_max_retries: int | None, decorator_max_retries: int
) -> int:
    if isinstance(instance_max_retries, int):
        wx_max_retries = instance_max_retries
    elif (env_max_retries := os.environ.get("WATSONX_MAX_RETRIES")) is not None:
        wx_max_retries = int(env_max_retries)
    else:
        wx_max_retries = decorator_max_retries
    return wx_max_retries


def _get_delay_time(
    instance_delay_time: float | None, decorator_delay_time: float
) -> float:
    if isinstance(instance_delay_time, float):
        wx_delay_time = instance_delay_time
    elif (env_delay_time := os.environ.get("WATSONX_DELAY_TIME")) is not None:
        wx_delay_time = float(env_delay_time)
    else:
        wx_delay_time = decorator_delay_time
    return wx_delay_time


def _get_retry_status_codes(
    instance_retry_status_codes: list | None, decorator_retry_status_codes: list
) -> list:
    wx_retry_status_codes = (
        instance_retry_status_codes
        or (
            list(
                map(
                    int,
                    os.environ.get("WATSONX_RETRY_STATUS_CODES", "")
                    .strip("[]")
                    .split(","),
                )
            )
            if os.environ.get("WATSONX_RETRY_STATUS_CODES")
            else []
        )
        or decorator_retry_status_codes
    )
    return wx_retry_status_codes


def _with_retry(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> httpx.Response:
            response: httpx.Response | None = None

            wx_max_retries = _get_max_retries(self.max_retries, max_retries)

            wx_delay_time = _get_delay_time(self.delay_time, delay_time)

            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )

            for attempt in range(max_retries + 1):
                if response is not None:
                    response.close()
                response = function(self, *args, **kwargs)

                if (
                    response is not None
                    and (response.status_code in wx_retry_status_codes)
                    and attempt != wx_max_retries
                ):
                    if self._client.CLOUD_PLATFORM_SPACES:
                        rate_limit_remaining = int(
                            response.headers.get(
                                REMAINING_LIMIT_HEADER,
                                self.rate_limiter.capacity,
                            )
                        )
                        if rate_limit_remaining == 0:
                            self.rate_limiter.adjust_tokens(rate_limit_remaining)
                        else:
                            time.sleep(backoff_timeout(wx_delay_time, attempt))
                        self.rate_limiter.acquire()
                    else:
                        time.sleep(backoff_timeout(wx_delay_time, attempt))
                else:
                    break

            return response  # type:ignore[return-value]

        return wrapper

    return decorator


def _with_retry_stream(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    """Decorator to retry the function if it encounters a 429 HTTP status."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @contextmanager  # Ensure the wrapped function remains a context manager
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Iterator[httpx.Response]:
            _exception = None
            response: httpx.Response | None = None

            wx_max_retries = _get_max_retries(self.max_retries, max_retries)

            wx_delay_time = _get_delay_time(self.delay_time, delay_time)

            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )

            for attempt in range(max_retries + 1):
                if response is not None:
                    response.close()
                with func(
                    self, *args, **kwargs
                ) as response:  # Call the original context manager
                    if (
                        response is not None
                        and (response.status_code in wx_retry_status_codes)
                        and attempt != wx_max_retries
                    ):
                        #  If the environment is set to cloud, the Token Bucket (rate_limiter here) is used to control traffic flow.
                        if self._client.CLOUD_PLATFORM_SPACES:
                            rate_limit_remaining = int(
                                response.headers.get(
                                    REMAINING_LIMIT_HEADER,
                                    self.rate_limiter.capacity,
                                )
                            )
                            if rate_limit_remaining == 0:
                                self.rate_limiter.adjust_tokens(rate_limit_remaining)
                            else:
                                time.sleep(backoff_timeout(wx_delay_time, attempt))
                            self.rate_limiter.acquire()
                        else:  # If CDP, don't use Token Bucket
                            time.sleep(backoff_timeout(wx_delay_time, attempt))
                        continue  # Retry the request
                    if response is not None:
                        yield response
                    return  # Ensure exit the loop after yielding

        return wrapper

    return decorator


def _with_async_retry(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> httpx.Response:
            response: httpx.Response | None = None

            wx_max_retries = _get_max_retries(self.max_retries, max_retries)
            wx_delay_time = _get_delay_time(self.delay_time, delay_time)
            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )
            for attempt in range(wx_max_retries + 1):
                if response is not None:
                    await response.aclose()
                response = await function(self, *args, **kwargs)

                if (
                    response is not None
                    and (response.status_code in wx_retry_status_codes)
                    and attempt != wx_max_retries
                ):
                    if self._client.CLOUD_PLATFORM_SPACES:
                        rate_limit_remaining = int(
                            response.headers.get(
                                REMAINING_LIMIT_HEADER,
                                self.rate_limiter.capacity,
                            )
                        )
                        if rate_limit_remaining == 0:
                            await self.rate_limiter.async_adjust_tokens(
                                rate_limit_remaining
                            )
                        else:
                            await asyncio.sleep(backoff_timeout(wx_delay_time, attempt))
                        await self.rate_limiter.acquire_async()
                    else:
                        await asyncio.sleep(backoff_timeout(wx_delay_time, attempt))
                else:
                    break

            return response  # type:ignore[return-value]

        return wrapper

    return decorator


def _with_async_retry_stream(
    max_retries: int = _MAX_RETRIES,
    delay_time: float = DEFAULT_DELAY,
    retry_status_codes: list[int] = DEFAULT_RETRY_STATUS_CODES,
) -> Callable:
    """Async decorator to retry the streaming function if it encounters a HTTP status code from `retry_status_codes` or env variable"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @asynccontextmanager
        async def wrapper(
            self: Any, *args: Any, **kwargs: Any
        ) -> AsyncIterator[httpx.Response]:
            wx_max_retries = _get_max_retries(self.max_retries, max_retries)
            wx_delay_time = _get_delay_time(self.delay_time, delay_time)
            wx_retry_status_codes = _get_retry_status_codes(
                self.retry_status_codes, retry_status_codes
            )

            response: httpx.Response | None = None
            for attempt in range(wx_max_retries + 1):
                if response is not None:
                    await response.aclose()

                async with func(self, *args, **kwargs) as response:
                    if response is not None and (
                        response.status_code in wx_retry_status_codes
                        and attempt != wx_max_retries
                    ):
                        #  If the environment is set to cloud, the Token Bucket (rate_limiter here) is used to control traffic flow.

                        if self._client.CLOUD_PLATFORM_SPACES:
                            rate_limit_remaining = int(
                                response.headers.get(
                                    REMAINING_LIMIT_HEADER,
                                    self.rate_limiter.capacity,
                                )
                            )
                            if rate_limit_remaining == 0:
                                await self.rate_limiter.async_adjust_tokens(
                                    rate_limit_remaining
                                )
                            else:
                                await asyncio.sleep(
                                    backoff_timeout(wx_delay_time, attempt)
                                )
                            await self.rate_limiter.acquire_async()
                        else:  # If CDP, don't use Token Bucket
                            await asyncio.sleep(backoff_timeout(wx_delay_time, attempt))
                        continue
                    if response is not None:
                        yield response
                    break

        return wrapper

    return decorator


class TokenBucket:
    """Thread-safe rate limiter with dynamic token adjustments."""

    def __init__(self, rate: float, capacity: int) -> None:
        self.capacity = capacity  # Max tokens
        self.rate = rate  # Tokens per second
        self.tokens: float = capacity  # Start full
        self.lock = threading.Lock()
        self.last_refill = time.time()
        self.condition_lock = threading.Condition(self.lock)
        self.async_lock = asyncio.Lock()
        self.waiting_threads: queue.Queue[int] = queue.Queue()

    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        if new_tokens >= 1:  # Only update if at least one token is added
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now

    def acquire(self) -> None:
        """Wait for a token and process threads in correct order."""
        thread_id = threading.get_ident()

        with self.condition_lock:
            # Add to queue if not already in front
            if (
                self.waiting_threads.empty()
                or self.waiting_threads.queue[-1] != thread_id
            ):
                self.waiting_threads.put(thread_id)

            while True:
                self.refill()

                # Allow thread to proceed only if it's at the front of the queue and tokens are available
                if self.tokens >= 1 and self.waiting_threads.queue[0] == thread_id:
                    self.waiting_threads.get()  # Remove from queue
                    self.tokens -= 1  # Consume token
                    self.condition_lock.notify()  # Wake next in line
                    return

                # Wait only until the next expected refill time
                next_refill = self.last_refill + (1 / self.rate)
                wait_time_float = max(0.0, next_refill - time.time())
                self.condition_lock.wait(wait_time_float)

    async def acquire_async(self) -> None:
        """Asynchronous acquire: Wait until a token is available."""
        async with self.async_lock:
            while self.tokens < 1:
                self.refill()
                wait_time = (1 / self.rate) if self.tokens < 1 else 0
                await asyncio.sleep(wait_time)
            self.tokens -= 1

    def adjust_tokens(self, remaining_tokens: int) -> None:
        """Adjust token count based on RateLimit-Remaining."""
        with self.lock:
            self.tokens = min(self.capacity, remaining_tokens)

    async def async_adjust_tokens(self, remaining_tokens: int) -> None:
        """Adjust token count based on RateLimit-Remaining."""
        async with self.async_lock:
            self.tokens = min(self.capacity, remaining_tokens)
