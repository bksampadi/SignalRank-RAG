from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_seconds: float


class AsyncRateLimiterBackend(Protocol):
    async def decide(self, client_id: str) -> RateLimitDecision:
        """
        Atomically evaluate and record one request attempt.

        Return a decision for accepted or rate-limited requests.
        Raise RateLimitBackendUnavailable if the backend cannot
        provide an authoritative decision.
        """
        ...


class RateLimitBackendUnavailable(RuntimeError):
    """Rate-limit storage cannot provide an authoritative decision."""
