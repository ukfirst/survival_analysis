import numpy as np
from surv_optimizer.utils.CoxStepFunction import StepFunction
import matplotlib.pyplot as plt


class NelsonAalenMultiStateEstimator:
    def __init__(self, cumulative_hazard_func, risk_calculator, dataset_manager, data_view):
        """
        Initialize the Nelson-Aalen estimator for a multi-state model.

        Parameters:
        - cumulative_hazard_func: Callable from get_cumulative_hazard_function to get cumulative hazards.
        - risk_calculator: An instance with compute_risk_set_at_risk for each transition.
        - dataset_manager: Provides access to event and time data.
        - data_view: A filtered DataFrame view from DatasetManager for specific transitions.
        """
        self.cumulative_hazard_func = cumulative_hazard_func
        self.risk_calculator = risk_calculator
        self.dataset_manager = dataset_manager
        self.data_view = data_view

        # Extract unique times and event counts from data_view
        self.unique_times = self.dataset_manager.get_unique_times()
        self.event_counts = self.dataset_manager.get_event_counts(self.data_view)
        self.individuals = self.data_view['individuals'].unique()  # Assuming this returns unique individuals

    def compute_transition_cumulative_hazard(self, verbose=False):
        """
        Compute cumulative hazard for a specific transition.

        Parameters:
        - verbose: If True, print detailed computation information.

        Returns:
        - Step function representing cumulative hazard for the given transition.
        """
        # Initialize cumulative hazard array
        cumulative_hazard = np.zeros(len(self.unique_times))
        hazard_increments = np.zeros(len(self.unique_times))

        if verbose:
            print(f"\nComputing Nelson-Aalen cumulative hazard for specified transition:")

        # Loop over each unique time point
        for time_bin_idx, time_point in enumerate(self.unique_times):
            # Get the number of events at the current time bin for the transition
            n_events = self.event_counts.get(time_point, 0)

            # Number of individuals at risk for this transition at time_point
            at_risk_indices = self.risk_calculator.compute_risk_set_at_risk(time_bin_idx)
            n_at_risk = len(at_risk_indices)

            # Initialize hazard increment to zero
            hazard_increment = 0.0

            # Calculate hazard increment only if there are individuals at risk and events occur
            if n_at_risk > 0 and n_events > 0:
                hazard_increment = n_events / n_at_risk
                hazard_increments[time_bin_idx] = hazard_increment

            # Update cumulative hazard (carry forward if no increment)
            cumulative_hazard[time_bin_idx] = (
                cumulative_hazard[time_bin_idx - 1] + hazard_increment if time_bin_idx > 0 else hazard_increment
            )

            if verbose:
                print(f"Time bin index: {time_bin_idx}, Time point: {time_point}")
                print(f"Number of events: {n_events}")
                print(f"Number at risk: {n_at_risk}")
                print(f"Hazard increment: {hazard_increment}")
                print(f"Cumulative hazard: {cumulative_hazard[time_bin_idx]}")

        return StepFunction(x=self.unique_times, y=cumulative_hazard)

    def compare_cumulative_hazard(self, transition_name="Transition", verbose=False):
        """
        Compare Nelson-Aalen cumulative hazard to model-based cumulative hazard, compute residuals, and plot.

        Parameters:
        - transition_name: A label for the plot, e.g., 'Transition 1 -> 2'.
        - verbose: If True, prints detailed residual information.
        """
        # Calculate Nelson-Aalen cumulative hazard
        nelson_aalen_hazard = self.compute_transition_cumulative_hazard(verbose=verbose)
        na_cumulative_hazard_values = nelson_aalen_hazard(self.unique_times)

        # Calculate the average model-based cumulative hazard across individuals
        model_cumulative_hazard_values = np.zeros(len(self.unique_times))

        # Sum cumulative hazards across all individuals
        for idx, individual in enumerate(self.individuals):
            individual_cumulative_hazard_func = self.cumulative_hazard_func[idx]
            individual_cumulative_hazard = [individual_cumulative_hazard_func(t) for t in self.unique_times]
            model_cumulative_hazard_values += np.array(individual_cumulative_hazard)

        # Average cumulative hazard over individuals
        model_cumulative_hazard_values /= len(self.individuals)

        # Compute Cox-Snell residuals
        cox_snell_residuals = na_cumulative_hazard_values - model_cumulative_hazard_values

        # Calculate R-squared measures
        residuals_squared = np.square(cox_snell_residuals)
        cox_snell_r2 = 1 - np.exp(-np.sum(residuals_squared) / len(cox_snell_residuals))
        kent_o_quigley_r2 = np.sum(residuals_squared) / (np.sum(na_cumulative_hazard_values) ** 2)

        # Print R-squared values
        if verbose:
            print(f"\nCox-Snell R-squared: {cox_snell_r2}")
            print(f"Kent-O'Quigley R-squared: {kent_o_quigley_r2}")

        # Plotting comparison
        plt.figure(figsize=(10, 6))
        plt.plot(self.unique_times, na_cumulative_hazard_values, label="Nelson-Aalen Estimator (Mean)", color="blue")
        plt.plot(self.unique_times, model_cumulative_hazard_values, label="Model-based Cumulative Hazard (Mean)",
                 color="red")
        plt.title(f"Comparison of Cumulative Hazard: {transition_name}")
        plt.xlabel("Time")
        plt.ylabel("Cumulative Hazard")
        plt.legend()

        # Add R-squared values as text on the plot
        plt.text(
            0.05, 0.95,
            f"Cox-Snell R² = {cox_snell_r2:.4f}\nKent-O'Quigley R² = {kent_o_quigley_r2:.4f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
        )

        plt.show()