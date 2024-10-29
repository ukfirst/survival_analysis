from abc import ABC, abstractmethod

class AbstractObjectiveFunction(ABC):

    @abstractmethod
    def compute_loss(self, w):
        """Compute the loss function (negative log-likelihood)."""
        pass
