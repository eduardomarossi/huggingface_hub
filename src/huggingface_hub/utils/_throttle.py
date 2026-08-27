# Copyright 2026-present, the HuggingFace Inc. team.
#
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
"""Process-wide download speed limiter."""

import threading
import time
import warnings

from .. import constants
from ._parsing import parse_size
from ._runtime import is_xet_available


# Smallest chunk handed out while throttling: below that, the per-chunk overhead (a lock + a sleep)
# starts to weigh more than the transfer itself.
MIN_THROTTLED_CHUNK_SIZE = 16 * 1024

# Chunks are sized to roughly this many seconds of budget: small enough for the pauses to be
# invisible on the progress bar, large enough to keep the loop cheap.
THROTTLED_CHUNK_DURATION = 0.1


class TokenBucket:
    """Token bucket enforcing a transfer rate shared by all callers.

    Tokens (bytes) are produced at a constant `rate` and consumed by whoever asks first, so the
    *sum* of all concurrent downloads never exceeds `rate` bytes per second, no matter how many
    workers are running.
    """

    def __init__(self, rate: int) -> None:
        self.rate = rate

        # Chunks must be much smaller than the per-second budget, otherwise the download alternates
        # between a full-speed burst and a long sleep (a 10MB chunk capped at 1MB/s would stall for
        # 10 seconds), which looks like a frozen download.
        self.chunk_size = max(
            MIN_THROTTLED_CHUNK_SIZE, min(constants.DOWNLOAD_CHUNK_SIZE, int(rate * THROTTLED_CHUNK_DURATION))
        )

        self._tokens = 0.0
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, nbytes: int) -> None:
        """Block until `nbytes` bytes have been granted by the bucket."""
        remaining = float(nbytes)
        while True:
            with self._lock:
                now = time.monotonic()
                # Refill is capped at one second worth of tokens: an idle bucket must not let the
                # workers download several seconds of backlog at full speed when they resume.
                self._tokens = min(float(self.rate), self._tokens + (now - self._last_refill) * self.rate)
                self._last_refill = now

                granted = min(self._tokens, remaining)
                self._tokens -= granted
                remaining -= granted
                if remaining <= 0:
                    return

                # Not enough budget: sleep for the time it takes to produce what's missing. Another
                # worker may grab those tokens in the meantime, in which case we simply wait again.
                delay = min(remaining, float(self.rate)) / self.rate
            time.sleep(delay)


_limiter: TokenBucket | None = None


class limit_download_speed:
    """
    Limit the total download speed, globally.

    The limit is shared by every download worker in the process: the sum of all concurrent
    downloads is capped at `max_speed` bytes per second, whether files are downloaded one by one or
    in parallel (e.g. [`snapshot_download`] with `max_workers=8`).

    Works as both a regular call and a context manager:
        limit_download_speed("5MB")        # limits until `limit_download_speed(None)`
        with limit_download_speed("5MB"):  # limits for the block, restores the previous limit on exit
            ...

    <Tip warning={true}>

    Rate limiting is only implemented for regular HTTP downloads. Since `hf_xet` cannot be rate
    limited, Xet-accelerated downloads are disabled while a limit is set.

    </Tip>

    Args:
        max_speed (`int` or `str`, *optional*):
            Maximum download speed, either in bytes per second (`5_000_000`) or as a string with a
            unit (`"5MB"`). Pass `None` to remove any limit.

    Example:
        ```py
        >>> from huggingface_hub import snapshot_download
        >>> from huggingface_hub.utils import limit_download_speed

        >>> with limit_download_speed("5MB"):  # never exceed 5MB/s in total
        ...     snapshot_download("meta-llama/Llama-3.2-1B-Instruct", max_workers=8)
        ```
    """

    def __init__(self, max_speed: int | str | None) -> None:
        global _limiter

        self._previous = _limiter
        _limiter = _build_limiter(max_speed)

        if _limiter is not None and is_xet_available():
            warnings.warn(
                "Xet-accelerated downloads cannot be rate limited. Falling back to regular HTTP downloads while a "
                "download speed limit is set."
            )

    def __enter__(self) -> "limit_download_speed":
        return self

    def __exit__(self, *exc) -> None:
        global _limiter
        _limiter = self._previous


def get_download_limiter() -> TokenBucket | None:
    """Return the limiter set by [`~utils.limit_download_speed`], or `None` if downloads are not throttled."""
    return _limiter


def _build_limiter(max_speed: int | str | None) -> TokenBucket | None:
    if max_speed is None:
        return None
    rate = parse_size(max_speed) if isinstance(max_speed, str) else int(max_speed)
    if rate <= 0:
        raise ValueError(f"Download speed limit must be strictly positive. Got '{max_speed}'.")
    return TokenBucket(rate)
