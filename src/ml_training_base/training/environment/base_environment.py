from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEnvironment(ABC):
    @abstractmethod
    def setup_environment(self, config: Dict[str, Any]) -> None:
        """
        Setup environment for training, e.g., determinism, GPU config, seeds, etc.
        """
        pass
