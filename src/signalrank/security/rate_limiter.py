from collections import deque
from collections.abc import Callable
from math import isfinite
from threading import Lock
from time import monotonic

from signalrank.security.contracts import RateLimitDecision


def evict_expired(
    timestamps: deque[float],
    now: float,
    window_seconds: float,
) -> None:

    while timestamps and timestamps[0] <= now - window_seconds:
        timestamps.popleft()


class SlidingWindowRateLimiter:
    def __init__(
        self,
        limit: int,
        window_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:

        if limit <= 0:
            raise ValueError("limit must be positive")

        if not isfinite(window_seconds) or window_seconds <= 0.0:
            raise ValueError("window_seconds must be finite and positive")

        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._clock = clock

    def allow(
        self,
        client_id: str,
    ) -> bool:

        return self.decide(client_id).allowed

    def decide(
        self,
        client_id: str,
    ) -> RateLimitDecision:

        with self._lock:
            now = self._clock()

            timestamps = self._requests.setdefault(client_id, deque())

            evict_expired(timestamps, now, self.window_seconds)

            if len(timestamps) < self.limit:
                timestamps.append(now)

                return RateLimitDecision(
                    allowed=True,
                    remaining=(self.limit - len(timestamps)),
                    retry_after_seconds=0.0,
                )

            return RateLimitDecision(
                allowed=False,
                remaining=0,
                retry_after_seconds=((timestamps[0] + self.window_seconds) - now),
            )
