from typing import Protocol
from negmas import Scenario


class ScenarioMaker(Protocol):
    def __call__(self, index: int) -> Scenario:
        """Creates a scenario"""
        ...
