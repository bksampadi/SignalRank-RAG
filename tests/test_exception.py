import pytest
import sys

from signalrank.exception.exception import SignalRankException

def test_exception():

    with pytest.raises(SignalRankException) as exc_info:
        
        try:

            a = 1/0

        except Exception as e:
            raise SignalRankException(e, sys)
        
    assert "division by zero" in str(exc_info.value)
    assert "line number" in str(exc_info.value)
    assert "python script" in str(exc_info.value)