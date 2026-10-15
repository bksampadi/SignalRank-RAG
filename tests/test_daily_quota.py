import pytest
from fastapi import HTTPException

import signalrank.api.main as api


def test_daily_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "_day", None)
    monkeypatch.setattr(api, "_used", 0)

    for _ in range(100):
        api.consume_quota()

    with pytest.raises(HTTPException) as exc:
        api.consume_quota()

    assert exc.value.status_code == 429
    assert api._used == 100
