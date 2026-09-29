"""Offline tests for the Ollama binding's transient-failure retry policy.

The binding declares three attempts with exponential backoff, so a server that
is briefly unavailable -- a model still loading, a restart, a proxy in front of
either -- must be retried rather than failing the whole extraction on the first
try. A request that is wrong on its own terms (bad model name, bad credentials)
must still fail fast: retrying it only re-buys the same failure.

The status matrix runs through the real decorator with a stub client, so it
exercises the retry loop and its predicate without a socket; two local-server
tests cover the HTTP path end to end.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import ollama
import pytest
from tenacity import wait_none

import lightrag.llm.ollama as ollama_binding
from lightrag.llm.ollama import _is_retryable_ollama_error, _ollama_model_if_cache

pytestmark = pytest.mark.offline

# The 5xx family plus 429: an upstream that is not ready yet.
_TRANSIENT_STATUSES = [429, 500, 502, 503, 504]
# A property of the request or the configuration, not of the moment.
_PERMANENT_STATUSES = [400, 401, 403, 404, 422]
# ollama reports "no response was ever received" as a negative code.
_NO_RESPONSE_STATUSES = [-1, None]

_SUCCESS_BODY = (
    b'{"message":{"role":"assistant","content":"recovered"},"done_reason":"stop"}'
)


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """Drop the 4s/8s sleeps; the attempt count is what these tests assert."""
    monkeypatch.setattr(_ollama_model_if_cache.retry, "wait", wait_none())


async def _attempts_against(monkeypatch, status):
    """Drive the real retry loop with a client whose ``chat`` always fails."""
    fake_client = SimpleNamespace(
        chat=AsyncMock(
            side_effect=ollama.ResponseError("upstream unavailable", status)
        ),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )
    monkeypatch.setattr(
        ollama_binding.ollama, "AsyncClient", lambda **kwargs: fake_client
    )

    with pytest.raises(ollama.ResponseError):
        await _ollama_model_if_cache("test-model", "hello")

    return fake_client.chat.await_count


@pytest.mark.parametrize("status", _TRANSIENT_STATUSES + _NO_RESPONSE_STATUSES)
async def test_transient_failure_is_retried_three_times(monkeypatch, status):
    assert await _attempts_against(monkeypatch, status) == 3


@pytest.mark.parametrize("status", _PERMANENT_STATUSES)
async def test_permanent_failure_stops_at_the_first_attempt(monkeypatch, status):
    assert await _attempts_against(monkeypatch, status) == 1


async def test_the_provider_error_survives_the_exhausted_loop(monkeypatch):
    """The pipeline's FAILED summary renders this message.

    An opaque ``tenacity.RetryError`` would hide the actionable half, which is
    why the decorator re-raises rather than wrapping.
    """
    fake_client = SimpleNamespace(
        chat=AsyncMock(side_effect=ollama.ResponseError("upstream unavailable", 503)),
        _client=SimpleNamespace(aclose=AsyncMock()),
    )
    monkeypatch.setattr(
        ollama_binding.ollama, "AsyncClient", lambda **kwargs: fake_client
    )

    with pytest.raises(ollama.ResponseError) as exc_info:
        await _ollama_model_if_cache("test-model", "hello")

    assert exc_info.value.status_code == 503
    assert "upstream unavailable" in str(exc_info.value)


class _OllamaShapedServer:
    """A local endpoint answering a scripted sequence of statuses.

    ``statuses`` is consumed one entry per request; the last entry repeats once
    the script is exhausted. Requests are counted so a test can assert how many
    attempts the binding actually made over HTTP.
    """

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.requests = 0
        server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler's API
                index = min(server.requests, len(server.statuses) - 1)
                status = server.statuses[index]
                server.requests += 1
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                body = _SUCCESS_BODY if status == 200 else b'{"error":"unavailable"}'
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self._server.server_address[1]}"


async def test_a_server_that_comes_back_is_recovered_not_failed():
    """The case the retry exists for: unavailable now, available moments later."""
    with _OllamaShapedServer([503, 503, 200]) as server:
        result = await _ollama_model_if_cache(
            "test-model", "hello", host=server.base_url
        )

        assert result == "recovered"
        assert server.requests == 3


async def test_an_unavailable_server_is_retried_over_http():
    with _OllamaShapedServer([503]) as server:
        with pytest.raises(ollama.ResponseError):
            await _ollama_model_if_cache("test-model", "hello", host=server.base_url)

        assert server.requests == 3


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        *[(status, True) for status in _TRANSIENT_STATUSES],
        *[(status, False) for status in _PERMANENT_STATUSES],
        *[(status, True) for status in _NO_RESPONSE_STATUSES],
    ],
)
def test_predicate_classification(status, expected):
    error = ollama.ResponseError("boom", status_code=status)
    assert _is_retryable_ollama_error(error) is expected


def test_predicate_ignores_unrelated_errors():
    """Only ollama's own error is retryable; anything else is a real bug."""
    assert _is_retryable_ollama_error(ValueError("boom")) is False
    assert _is_retryable_ollama_error(TimeoutError()) is False
