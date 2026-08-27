import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from huggingface_hub import constants
from huggingface_hub.file_download import http_get
from huggingface_hub.utils._throttle import (
    MIN_THROTTLED_CHUNK_SIZE,
    TokenBucket,
    get_download_limiter,
    limit_download_speed,
)


FILE_SIZE = 2_000_000
NB_WORKERS = 4


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Never leak a speed limit to the next test."""
    yield
    limit_download_speed(None)


class TestTokenBucket:
    def test_consume_respects_rate(self):
        bucket = TokenBucket(rate=1_000_000)
        start = time.monotonic()
        for _ in range(10):
            bucket.consume(100_000)  # 1MB in total => 1s at 1MB/s
        # Lower bound is the actual contract; the upper bound only guards against a stall (a busy
        # machine may always take longer than the limit, never less).
        assert 1.0 <= time.monotonic() - start < 3.0

    def test_rate_is_shared_across_threads(self):
        """The *sum* of all workers must not exceed the rate, no matter how many are running."""
        bucket = TokenBucket(rate=1_000_000)
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: bucket.consume(250_000), range(8)))  # 2MB in total => 2s at 1MB/s
        assert 2.0 <= time.monotonic() - start < 5.0

    def test_consume_larger_than_rate(self):
        """A single chunk bigger than one second of budget must not deadlock."""
        bucket = TokenBucket(rate=1_000_000)
        start = time.monotonic()
        bucket.consume(2_000_000)
        assert 2.0 <= time.monotonic() - start < 5.0

    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            (10, MIN_THROTTLED_CHUNK_SIZE),  # tiny rate => floored
            (1_000_000, 100_000),  # ~100ms worth of budget
            (10_000_000_000, constants.DOWNLOAD_CHUNK_SIZE),  # huge rate => capped at the default chunk size
        ],
    )
    def test_chunk_size(self, rate: int, expected: int):
        assert TokenBucket(rate=rate).chunk_size == expected


class TestLimitDownloadSpeed:
    def test_no_limit_by_default(self):
        assert get_download_limiter() is None

    def test_set_and_unset_limit(self):
        limit_download_speed("5MB")
        limiter = get_download_limiter()
        assert limiter is not None and limiter.rate == 5_000_000

        limit_download_speed(None)
        assert get_download_limiter() is None

    def test_accepts_bytes_per_second(self):
        limit_download_speed(1234)
        limiter = get_download_limiter()
        assert limiter is not None and limiter.rate == 1234

    def test_as_context_manager_restores_previous_limit(self):
        limit_download_speed("5MB")
        with limit_download_speed("1MB"):
            limiter = get_download_limiter()
            assert limiter is not None and limiter.rate == 1_000_000
        limiter = get_download_limiter()
        assert limiter is not None and limiter.rate == 5_000_000

    @pytest.mark.parametrize("value", ["not-a-size", "1.5MB", 0, -1])
    def test_invalid_value(self, value):
        with pytest.raises(ValueError):
            limit_download_speed(value)


class _PayloadHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(FILE_SIZE))
        self.end_headers()
        self.wfile.write(b"0" * FILE_SIZE)

    def log_message(self, *args) -> None:
        """Silence the default per-request logging to stderr."""


@pytest.fixture(scope="module")
def payload_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PayloadHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_http_get_is_throttled_globally(payload_server: str):
    """Workers share a single budget: 4 x 2MB capped at 4MB/s takes 2s, not the ~0s the loopback allows."""

    def _download(_) -> None:
        with io.BytesIO() as buffer:
            http_get(payload_server, buffer, expected_size=FILE_SIZE)

    start = time.monotonic()
    with limit_download_speed(4_000_000), ThreadPoolExecutor(max_workers=NB_WORKERS) as pool:
        list(pool.map(_download, range(NB_WORKERS)))
    assert 2.0 <= time.monotonic() - start < 6.0
