from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest

from signalrank.security.contracts import RateLimitDecision
from signalrank.security.rate_limiter import (
    SlidingWindowRateLimiter,
    evict_expired,
)


def test_evict_expired() -> None:
    timestamps = deque([10.0, 12.0, 16.0, 18.0])

    evict_expired(
        timestamps=timestamps,
        now=20.0,
        window_seconds=10.0,
    )

    assert timestamps == deque([12.0, 16.0, 18.0])


def test_evict_expired_empty_deque() -> None:

    timestamps = deque([])

    evict_expired(
        timestamps=timestamps,
        now=20.0,
        window_seconds=10.0,
    )

    assert timestamps == deque([])


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_multiple_clients() -> None:
    clock = FakeClock(now=1.0)

    limiter = SlidingWindowRateLimiter(
        limit=3,
        window_seconds=10.0,
        clock=clock,
    )

    assert limiter.allow("alice") is True

    clock.now = 2.0
    assert limiter.allow("alice") is True

    clock.now = 3.0
    assert limiter.allow("alice") is True

    clock.now = 4.0
    assert limiter.allow("alice") is False

    clock.now = 4.0
    assert limiter.allow("bob") is True

    clock.now = 5.0
    assert limiter.allow("bob") is True

    clock.now = 6.0
    assert limiter.allow("bob") is True

    clock.now = 7.0
    assert limiter.allow("bob") is False

    clock.now = 11.0
    assert limiter.allow("alice") is True


def test_concurrent_requests() -> None:
    clock = FakeClock(now=5.0)

    limiter = SlidingWindowRateLimiter(limit=10, window_seconds=60.0, clock=clock)

    def make_request(_: int) -> bool:
        return limiter.allow("alice")

    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(make_request, range(100)))

    assert sum(results) == 10
    assert len(results) - sum(results) == 90


def test_rate_limit_decision_fields() -> None:
    decision = RateLimitDecision(
        allowed=True,
        remaining=2,
        retry_after_seconds=0.0,
    )

    assert decision.allowed is True
    assert decision.remaining == 2
    assert decision.retry_after_seconds == 0.0


def test_rate_limit_decision_sequence() -> None:
    clock = FakeClock(now=1.0)
    limiter = SlidingWindowRateLimiter(
        limit=3,
        window_seconds=10.0,
        clock=clock,
    )

    decisions = limiter.decide("alice")

    assert decisions.allowed is True
    assert decisions.remaining == 2
    assert decisions.retry_after_seconds == 0.0

    clock.now = 2.0

    decisions = limiter.decide("alice")

    assert decisions.allowed is True
    assert decisions.remaining == 1
    assert decisions.retry_after_seconds == 0.0

    clock.now = 3.0

    decisions = limiter.decide("alice")

    assert decisions.allowed is True
    assert decisions.remaining == 0
    assert decisions.retry_after_seconds == 0.0

    clock.now = 4.0

    decisions = limiter.decide("alice")

    assert decisions.allowed is False
    assert decisions.remaining == 0
    assert decisions.retry_after_seconds == 7.0

    clock.now = 11.0

    decisions = limiter.decide("alice")

    assert decisions.allowed is True
    assert decisions.remaining == 0
    assert decisions.retry_after_seconds == 0.0


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_rejects_non_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        SlidingWindowRateLimiter(
            limit=limit,
            window_seconds=10.0,
        )


@pytest.mark.parametrize(
    "window_seconds",
    [
        0.0,
        -5.0,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_invalid_window_seconds(
    window_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(
            limit=10,
            window_seconds=window_seconds,
        )
