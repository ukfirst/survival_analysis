import numpy as np
from scipy.special import logsumexp

class TwoStateCovariateContribution:
    def __init__(self, dataset_manager, data_view, risk_calculator):
        """
        Initialize the CovariateContributionCalculator with DatasetManager and data view.
        """
        self.dataset_manager = dataset_manager
        self.data_view = data_view.reset_index(drop=True)
        self.risk_calculator = risk_calculator

        # Extract necessary data
        self.Z_i1 = dataset_manager.Z_i1(data_view)
        self.Z_ij1_filled = dataset_manager.Z_ij1(data_view)
        self.individuals = self.data_view['individuals'].values
        self.unique_times = self.dataset_manager.get_unique_times()
        self.n_static_covariates = dataset_manager.n_static_covariates
        self.n_time_dependent_covariates = dataset_manager.n_time_dependent_covariates

        # Initialize coefficients
        self.beta = None
        self.gamma = None
        self.lambda_ = None

    def set_coefficients(self, w):
        """Extract coefficients for static and time-dependent covariates."""
        n_s = self.n_static_covariates
        n_td = self.n_time_dependent_covariates

        self.beta = w[:n_s]
        self.gamma = w[n_s:n_s + n_td]
        self.lambda_ = w[n_s + n_td:n_s + 2 * n_td]

    def compute_static_contribution(self, index, verbose=False):
        """Calculate the static covariate contribution."""
        Z_i1 = self.Z_i1[index]
        static_contribution = np.dot(Z_i1, self.beta)

        if verbose:
            print("\n--- Static Contribution ---")
            for i, (zi, b) in enumerate(zip(Z_i1, self.beta)):
                print(f"Z_i1[{i}]: {zi} * beta[{i}]: {b} = {zi * b}")
            print(f"Static Contribution (sum): {static_contribution}\n")

        return static_contribution

    def compute_time_contribution(self, index, B_ij, verbose=False):
        """Calculate the time-dependent covariate contribution and fading effect."""
        Z_ij1 = self.Z_ij1_filled[index]
        gamma_contribution = np.dot(self.gamma, Z_ij1)
        fading_effect = 0  # Currently no fading effect applied

        if verbose:
            print("\n--- Time Contribution ---")
            for i, (z_ij, gamma_i) in enumerate(zip(Z_ij1, self.gamma)):
                print(f"Z_ij1[{i}]: {z_ij} * gamma[{i}]: {gamma_i} = {z_ij * gamma_i}")
            print(f"Time Contribution (sum of gamma contribution): {gamma_contribution}")
            print(f"Fading Effect (not applied): {fading_effect}\n")

        time_contribution = gamma_contribution - fading_effect
        return time_contribution

    def compute_total_contribution(self, index, time_bin_idx, verbose=False):
        """Calculate the total covariate contribution for a specific individual at a given time bin."""
        individual = self.individuals[index]
        current_time_bin = self.unique_times[time_bin_idx]
        B_ij_scalar = self.risk_calculator.get_sojourn_time(individual, current_time_bin)

        # Create B_ij array matching the shape of gamma and lambda_
        B_ij = np.full_like(self.gamma, B_ij_scalar)

        # Compute contributions
        static_contribution = self.compute_static_contribution(index, verbose=verbose)
        time_contribution = self.compute_time_contribution(index, B_ij, verbose=verbose)

        total_contribution = static_contribution + time_contribution

        if verbose:
            print(f"\n--- Total Contribution for Individual {individual} at Time Bin {time_bin_idx} ---")
            print(f"Static Contribution: {static_contribution}")
            print(f"Time Contribution: {time_contribution}")
            print(f"Total Contribution: {total_contribution}\n")

        return total_contribution

    def compute_risk_set_contribution(self, indices, time_bin_idx, verbose=False):
        """Calculate contributions for individuals still at risk at a given time bin."""
        contributions = []
        if verbose:
            print(f"\n--- Risk Set at Time Bin {time_bin_idx} ---")
        for idx in indices:
            total_contribution = self.compute_total_contribution(idx, time_bin_idx, verbose=verbose)
            contributions.append(total_contribution)
        return np.array(contributions)

    def compute_log_sum_at_risk(self, time_bin_idx, verbose=False):
        """Compute the log-sum-exp of the risk contributions for the risk set at a specific time bin."""
        at_risk_indices = self.risk_calculator.compute_risk_set_at_risk(time_bin_idx)
        contributions = self.compute_risk_set_contribution(at_risk_indices, time_bin_idx, verbose=verbose)

        if len(contributions) == 0:
            return 0.0

        log_sum_exp = logsumexp(contributions)

        if verbose:
            print(f"\n--- Log-Sum-Exp for Time Bin {time_bin_idx} ---")
            print(f"At Risk Individuals: {at_risk_indices}")
            print(f"Risk Contributions: {contributions}")
            print(f"Log-Sum-Exp: {log_sum_exp}\n")

        return log_sum_exp