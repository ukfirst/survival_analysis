import numpy as np
from surv_optimizer.abstract_classes.AbstractObjectiveFunction import AbstractObjectiveFunction
from surv_optimizer.calculators.TwoStateEventRisk import TwoStateEventRisk
from surv_optimizer.data.DatasetManager import DatasetManager
from surv_optimizer.Optimizer import Optimizer
from surv_optimizer.calculators.TwoStateCovariateContribution import TwoStateCovariateContribution

class TwoStateCoxObjectiveFunction(AbstractObjectiveFunction):
    def __init__(self, dataset_manager, data_view, alpha_reg=0.1, lambda_reg=0.5):
        """
        Initialize the ObjectiveFunction with a DatasetManager and data view.

        Parameters:
        - dataset_manager: An instance of DatasetManager.
        - data_view: The filtered DataFrame view from DatasetManager.
        """
        self.dataset_manager = dataset_manager
        self.data_view = data_view
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg
        self.time = self.data_view['time'].values

        # Get unique times and number of events at each time
        self.unique_times = self.dataset_manager.get_unique_times()
        self.n_total_covariates = self.dataset_manager.n_total_covariates

        # Placeholder for calculators
        self.covariate_calculator = None
        self.risk_calculator = None

    def set_calculators(self, covariate_calculator, risk_calculator):
        """
        Set the covariate contribution calculator and risk calculator.
        """
        self.covariate_calculator = covariate_calculator
        self.risk_calculator = risk_calculator

    def compute_loss(self, w, verbose=False):
        """Calculate the log-likelihood loss function for the Cox model."""
        # Set coefficients in the covariate calculator
        self.covariate_calculator.set_coefficients(w)

        log_likelihood = 0.0

        if verbose:
            print("=== Computation of Log-Likelihood Loss ===")
            print(f"Initial log-likelihood: {log_likelihood}\n")

        # Iterate over all unique times in the dataset
        for time_bin_idx, time_point in enumerate(self.unique_times):
            # Get event indices at the current time
            individuals = self.risk_calculator.compute_event_indices(time_point)

            if verbose:
                print(f"--- Time Point: {time_point} ---")
                print(f"Individuals with events: {individuals}")

            # Contribution from events at the current time bin
            for idx in individuals:
                # Calculate total contribution using CovariateContributionCalculator
                individual_contribution = self.covariate_calculator.compute_total_contribution(idx, time_bin_idx,
                                                                                               verbose=verbose)
                log_likelihood += individual_contribution

                if verbose:
                    print(f"Individual {idx} contribution: {individual_contribution}")
                    print(f"Updated log-likelihood: {log_likelihood}")

            # Calculate log-sum-exp of the risk set at the current time
            risk_set_log_sum = self.covariate_calculator.compute_log_sum_at_risk(time_bin_idx, verbose=verbose)

            if verbose:
                print(f"Log-sum-exp for risk set: {risk_set_log_sum}")

            # Subtract logarithm of risk set sum, scaled by the number of events at the current time
            log_likelihood -= len(individuals) * risk_set_log_sum

            if verbose:
                print(f"Updated log-likelihood after subtracting risk set log-sum-exp: {log_likelihood}\n")

        if verbose:
            print(f"Final log-likelihood (before negation): {log_likelihood}\n")

        # Apply Elastic Net regularization
        l1_term = self.alpha_reg * self.lambda_reg * np.sum(np.abs(w))  # L1 (Lasso) component
        l2_term = self.alpha_reg * (1 - self.lambda_reg) * np.sum(w ** 2)  # L2 (Ridge) component
        reg_term = l1_term + l2_term

        # Final loss with Elastic Net regularization
        return -(log_likelihood - reg_term)


if __name__ == "__main__":
    dataset_manager = DatasetManager()
    view = dataset_manager.get_dataview(state_from=1,sorted=False)
    risk_calculator = TwoStateEventRisk(dataset_manager, view)
    covariate_calculator = TwoStateCovariateContribution(dataset_manager, view, risk_calculator)
    objective_function = TwoStateCoxObjectiveFunction(dataset_manager, view)
    objective_function.set_calculators(covariate_calculator, risk_calculator)

    def test_loss_function():
        # Set the coefficients to known values
        w = np.array([1.0, 0.5, 1.5, -0.5])  # Known coefficients

        # Assume we have initialized the dataset, risk calculator, and covariate calculator
        loss = objective_function.compute_loss(w, verbose=True)

        print(f"Computed Loss: {loss}")


    # Run the test
    test_loss_function()

    w = np.array([2.0, 1.0, 1.5, -0.5])
    def finite_difference_gradient_check(objective_function, w, epsilon=1e-5):
        """Check if the gradient computed by the loss function is consistent using finite differences."""
        n_params = len(w)
        grad_approx = np.zeros(n_params)

        for i in range(n_params):
            w_eps_plus = np.copy(w)
            w_eps_minus = np.copy(w)

            w_eps_plus[i] += epsilon
            w_eps_minus[i] -= epsilon

            loss_plus = objective_function.compute_loss(w_eps_plus)
            loss_minus = objective_function.compute_loss(w_eps_minus)

            # Approximate gradient
            grad_approx[i] = (loss_plus - loss_minus) / (2 * epsilon)

        print(f"Approximate Gradient: {grad_approx}")
        return grad_approx


    # Test finite difference gradient check
    finite_difference_gradient_check(objective_function, w)




