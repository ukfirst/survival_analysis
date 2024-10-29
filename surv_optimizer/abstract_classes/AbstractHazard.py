from abc import ABC, abstractmethod

class AbstractHazard(ABC):

    def __init__(self, dataset_manager, data_view):
        """
        Initialize with an objective function instance.
        The objective function is used to provide precomputed values, covariates, etc.
        """
        self.dataset_manager = dataset_manager
        self.data_view = data_view
        self.baseline_hazard_ = None
        self.baseline_survival_ = None

    @abstractmethod
    def compute_baseline_hazard(self, w):
        """Compute the baseline hazard function."""
        pass

    @abstractmethod
    def get_cumulative_hazard_function(self):
        """Compute the hazard for input data X."""
        pass

    @abstractmethod
    def get_survival_function(self):
        """Compute the survival function for input data X."""
        pass

    def calculate_risk_contribution(self, param):
        pass
