import numpy as np
import pandas as pd

from surv_optimizer.utils.CoxStepFunction import StepFunction
from surv_optimizer.abstract_classes.AbstractHazard import AbstractHazard

class TwoStateHazard(AbstractHazard):
    def __init__(self, dataset_manager, data_view, verbose=False):
        """
        Initialize the TwoStateHazard.

        Parameters:
        - dataset_manager: Dataset manager instance.
        - data_view: Data view containing the dataset.
        - verbose: If True, print detailed computation steps.
        """
        # Call to the super class to initialize the objective function
        super().__init__(dataset_manager, data_view)

        self.verbose = verbose

        # Extract variables
        self.individuals = self.data_view['individuals'].unique()
        self.n_samples = len(self.individuals)
        self.time = self.data_view['time'].values

        # Extract unique times and event counts from data_view
        self.unique_times = self.dataset_manager.get_unique_times()
        self.event_counts = self.dataset_manager.get_event_counts(self.data_view)

        # Set calculators
        self.covariate_calculator = None
        self.risk_calculator = None

        # Initialize baseline hazard and survival functions
        self.baseline_hazard_ = None
        self.baseline_survival_ = None

    def set_calculators(self, covariate_calculator, risk_calculator):
        """Set the covariate contribution calculator for the survival calculator."""
        self.covariate_calculator = covariate_calculator
        self.risk_calculator = risk_calculator
        if self.verbose:
            print("Calculators have been set.")

    def compute_baseline_hazard(self, w, verbose=False):
        """Compute the cumulative baseline hazard using Breslow's method."""
        # Set coefficients
        self.covariate_calculator.set_coefficients(w)
        if self.verbose:
            print("\nComputing baseline hazard:")
            print(f"Coefficients (w): {w}")

        # Initialize arrays
        hazard = np.zeros(len(self.unique_times))
        cumulative_hazard = np.zeros(len(self.unique_times))

        # Iterate over time bins
        for time_bin_idx, time_point in enumerate(self.unique_times):
            # Number of events at this time
            n_events = self.event_counts.iloc[time_bin_idx]

            # Compute risk set log-sum at this time
            risk_set_log_sum = self.covariate_calculator.compute_log_sum_at_risk(time_bin_idx)

            # Initialize hazard increment to zero
            hazard_increment = 0.0

            # Set risk_set_log_sum to 0 if it is None
            risk_set_log_sum = risk_set_log_sum if risk_set_log_sum is not None else 0
            # Compute hazard increment
            if risk_set_log_sum > 0 and n_events > 0:
                hazard_increment = n_events / np.exp(risk_set_log_sum)
                hazard[time_bin_idx] = hazard_increment

            # Update cumulative hazard
            cumulative_hazard[time_bin_idx] = (
                cumulative_hazard[time_bin_idx - 1] + hazard_increment if time_bin_idx > 0 else hazard_increment
            )

            if self.verbose:
                print(f"\nTime bin index: {time_bin_idx}, Time point: {time_point}")
                print(f"Number of events: {n_events}")
                print(f"Risk set log-sum: {risk_set_log_sum}")
                print(f"Hazard increment: {hazard_increment}")
                print(f"Cumulative hazard: {cumulative_hazard[time_bin_idx]}")

        # Store baseline hazard and survival as step functions
        self.baseline_hazard_ = StepFunction(x=self.unique_times, y=hazard)
        baseline_cumulative_hazard = cumulative_hazard
        baseline_survival = np.exp(-baseline_cumulative_hazard)
        self.baseline_survival_ = StepFunction(x=self.unique_times, y=baseline_survival)

        if self.verbose:
            print("\nBaseline hazard and survival functions have been computed.")

    def compute_baseline_hazard_sojourn(self, verbose=False):
        # Check if baseline hazard is computed
        if self.baseline_hazard_ is None:
            raise ValueError("The baseline hazard must be computed first using 'compute_baseline_hazard'.")
        if verbose:
            print("\nComputing baseline hazard as a function of sojourn time:")

    def get_cumulative_hazard_function(self, verbose=False): # rename to transition hazard
        """Compute the cumulative hazard function for each individual over all time bins."""
        # Check if baseline hazard is computed
        if self.baseline_hazard_ is None:
            raise ValueError("The baseline hazard must be computed first using 'compute_baseline_hazard'.")

        funcs = np.empty(self.n_samples, dtype=object)

        for idx, individual in enumerate(self.individuals):
            cumulative_hazard = np.zeros(len(self.unique_times))

            if self.verbose:
                print(f"\nComputing cumulative hazard for individual {individual}:")

            for time_bin_idx, time_point in enumerate(self.unique_times):
                # Compute individual total contribution
                index = np.where(self.individuals == individual)[0][0]
                total_contribution = self.covariate_calculator.compute_total_contribution(index, time_bin_idx)

                # Compute hazard at this time
                hazard = self.baseline_hazard_.y[time_bin_idx] * np.exp(total_contribution)

                # Update cumulative hazard
                cumulative_hazard[time_bin_idx] = (
                    cumulative_hazard[time_bin_idx - 1] + hazard if time_bin_idx > 0 else hazard
                )

                if self.verbose:
                    print(f"Time bin index: {time_bin_idx}, Time point: {time_point}")
                    print(f"Total contribution: {total_contribution}")
                    #print(f"Baseline hazard: {self.baseline_hazard_.y[time_bin_idx]}")
                    #print(f"Hazard: {hazard}")
                    print(f"Cumulative hazard: {cumulative_hazard[time_bin_idx]}")

            # Store cumulative hazard as a step function
            funcs[idx] = StepFunction(x=self.unique_times, y=cumulative_hazard)

        return funcs

    def get_survival_function(self):
        """Compute the survival function for each individual over all time bins."""
        # Check if baseline survival is computed
        if self.baseline_survival_ is None:
            raise ValueError("The baseline hazard must be computed first using 'compute_baseline_hazard'.")

        funcs = np.empty(self.n_samples, dtype=object)

        for idx, individual in enumerate(self.individuals):
            cumulative_hazard = np.zeros(len(self.unique_times))

            if self.verbose:
                print(f"\nComputing survival function for individual {individual}:")

            for time_bin_idx, time_point in enumerate(self.unique_times):
                # Compute individual total contribution
                index = np.where(self.individuals == individual)[0][0]
                total_contribution = self.covariate_calculator.compute_total_contribution(index, time_bin_idx, verbose=False)

                # Compute hazard at this time
                hazard = self.baseline_hazard_.y[time_bin_idx] * np.exp(total_contribution)

                # Update cumulative hazard
                cumulative_hazard[time_bin_idx] = (
                    cumulative_hazard[time_bin_idx - 1] + hazard if time_bin_idx > 0 else hazard
                )

                if self.verbose:
                    print(f"Time bin index: {time_bin_idx}, Time point: {time_point}")
                    print(f"Total contribution: {total_contribution}")
                    print(f"Hazard: {hazard}")
                    print(f"Cumulative hazard: {cumulative_hazard[time_bin_idx]}")

            # Compute survival function
            survival_probs = np.exp(-cumulative_hazard)

            # Store survival function as a step function
            funcs[idx] = StepFunction(x=self.unique_times, y=survival_probs)

        return funcs

    def compute_hazard_rate(self, index, time_bin_idx):
        """Compute the hazard rate for a specific individual at a given time bin."""
        # Get baseline hazard at the current time bin
        baseline_hazard = self.baseline_hazard_.y[time_bin_idx]

        # Compute total contribution
        total_contribution = self.covariate_calculator.compute_total_contribution(index, time_bin_idx)

        # Compute hazard rate
        hazard_rate = baseline_hazard * np.exp(total_contribution)

        if self.verbose:
            print(f"Computing hazard rate for individual index {index}, time bin {time_bin_idx}")
            print(f"Baseline hazard: {baseline_hazard}")
            print(f"Total contribution: {total_contribution}")
            print(f"Hazard rate: {hazard_rate}")

        return hazard_rate

    def compute_conductance(self, index, time_bin_idx):
        """Compute the conductance, equivalent to the hazard rate."""
        return self.compute_hazard_rate(index, time_bin_idx)

    def compute_effective_resistance(self, index):
        """Compute the effective resistance for an individual over all time bins."""
        effective_resistance = 0.0

        if self.verbose:
            print(f"\nComputing effective resistance for individual index {index}:")

        for time_bin_idx in range(len(self.unique_times)):
            hazard_rate = self.compute_hazard_rate(index, time_bin_idx)
            if hazard_rate > 0:
                effective_resistance += 1.0 / hazard_rate

                if self.verbose:
                    print(f"Time bin index: {time_bin_idx}")
                    print(f"Hazard rate: {hazard_rate}")
                    print(f"Effective resistance: {effective_resistance}")

        return effective_resistance

    import numpy as np

    def calculate_expected_survival_time(self):
        """
        Calculate the expected survival time for each individual based on their survival function.

        Parameters:
        - survival_functions: array of step functions representing the survival probabilities over time for each individual.
        - unique_times: array of time points corresponding to the survival probabilities.

        Returns:
        - expected_times: array of expected survival times for each individual.
        """
        survival_functions = self.get_survival_function()
        expected_times = np.zeros(self.individuals.size)

        # Compute the time intervals between each unique time point
        time_intervals = np.diff(self.unique_times, prepend=0)

        for idx, survival_function in enumerate(survival_functions):
            # Evaluate survival probabilities at each time point for this individual
            survival_probs = survival_function.y  # Access survival probabilities in the StepFunction

            # Expected survival time calculation by integrating survival probabilities
            expected_times[idx] = np.sum(survival_probs * time_intervals)

        return expected_times


# Now, let's create a small trial in the __main__ section
if __name__ == "__main__":
    import numpy as np
    from surv_optimizer.calculators.TwoStateCovariateContribution import TwoStateCovariateContribution
    from surv_optimizer.calculators.TwoStateEventRisk import TwoStateEventRisk
    from surv_optimizer.objective_functions.TwoStateCoxObjectiveFunction import TwoStateCoxObjectiveFunction
    from surv_optimizer.data.DatasetManager import DatasetManager
    # Initialize DatasetManager
    dataset_manager = DatasetManager()

    # Get data view for state_from=1 and sorted=True
    data_view = dataset_manager.get_dataview(state_from=1, sorted=True)

    # Initialize ObjectiveFunction with DatasetManager and data_view
    objective_function = TwoStateCoxObjectiveFunction(dataset_manager, data_view)

    # Initialize RiskCalculator and CovariateContributionCalculator
    risk_calculator = TwoStateEventRisk(dataset_manager, data_view)
    covariate_calculator = TwoStateCovariateContribution(dataset_manager, data_view, risk_calculator)

    # Set calculators in ObjectiveFunction
    objective_function.set_calculators(covariate_calculator, risk_calculator)

    # Initialize SurvivalCalculator
    survival_calculator = TwoStateHazard(dataset_manager, data_view, verbose=True)
    survival_calculator.set_calculators(covariate_calculator, risk_calculator)

    w = np.array([1.05196097, 0.77857355, 0.84338372, 0.34661934])

    # Compute the baseline hazard and survival functions
    survival_calculator.compute_baseline_hazard(w, verbose=False)
    survival_calculator.get_cumulative_hazard_function()

    # Retrieve survival and hazard functions
    # survival_funcs = survival_calculator.get_survival_function()
    # cumulative_hazard_funcs = survival_calculator.get_cumulative_hazard_function()
    #
    # # For individual 0, retrieve the survival and hazard functions over time
    # individual_idx = 0
    # time_points = survival_calculator.uniq_times
    #
    # # Define functions to extract survival and hazard values for individual 0
    # survival_func = lambda times: survival_funcs[individual_idx](times)
    # cumulative_hazard_func = lambda times: cumulative_hazard_funcs[individual_idx](times)
    #
    # # Print survival and cumulative hazard functions
    # print("\nSurvival function for individual 0:")
    # for t in time_points:
    #     s = survival_func(t)
    #     print(f"Time: {t}, Survival: {s}")
    #
    # print("\nCumulative hazard function for individual 0:")
    # for t in time_points:
    #     h = cumulative_hazard_func(t)
    #     print(f"Time: {t}, Cumulative Hazard: {h}")
