import pytest
import sys

from src.signalrank.exception.exception import SignalRankException

def test_exception():

    with pytest.raises(SignalRankException):
        
        try:

            a = 1/0

        except Exception as e:
            raise SignalRankException(e, sys)