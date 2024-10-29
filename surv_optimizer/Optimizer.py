import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted

class Optimizer:
    def __init__(self, objective_function, hazard_survival_calculator, covariate_calculator):
        """
        Initialize the optimizer.

        Parameters:
        - objective_function: Instance of a subclass of AbstractObjectiveFunction.
        - hazard_survival_calculator: Instance of a subclass of AbstractHazardSurvivalCalculator.
        - lambda_reg: Regularization coefficient (default = 10).
        """
        self.objective_function = objective_function
        self.hazard_survival_calculator = hazard_survival_calculator
        self.covariate_calculator = covariate_calculator

        self.is_fitted_ = False
        self.coef_ = None

    def fit(self, w_init=None, n_iter=40, early_stop_m=3, tol=1e-1, verbose=False, show_progress=True):
        if w_init is None:
            w_init = np.random.randn(self.objective_function.n_total_covariates)

        bounds = [(None, None)] * len(w_init)
        last_loss = None
        repeat_count = 0

        with tqdm(total=n_iter, desc="Optimization Progress", ncols=100, disable=not show_progress, leave=True) as pbar:
            for iteration in range(n_iter):
                # Optimization using BFGS for a single iteration
                result = minimize(
                    fun=self.objective_function.compute_loss,
                    x0=w_init,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 1,
                        'ftol': 1e-4,  # Increase this value for larger steps
                        'gtol': 1e-4,  # Increase tolerance to allow larger steps
                        'eps': 1e-2,   # Larger epsilon for bigger step sizes in gradient approximation
                        'maxfun': 500  # Increase max function evaluations to allow more exploration
                    }
                )

                w_init = result.x

                self.callback(iteration + 1, w_init, result.jac, verbose)

                # Early stopping logic based on loss improvement
                current_loss = result.fun
                if last_loss is not None:
                    improvement = last_loss - current_loss
                    if improvement < tol:
                        repeat_count += 1
                        if repeat_count >= early_stop_m:
                            if verbose:
                                print(
                                    f"Early stopping at iteration {iteration + 1}. No significant improvement in loss.")
                            break
                    else:
                        repeat_count = 0  # Reset if there is sufficient improvement
                last_loss = current_loss

                pbar.update(1)

        self.is_fitted_ = True
        self.coef_ = w_init
        return self

    def check_is_fitted_estimator(self, attributes):
        """Check if the estimator is fitted by verifying its required attributes."""
        check_is_fitted(self, attributes)

    def compute_baseline_hazard(self):
        self.check_is_fitted_estimator('coef_')
        return self.hazard_survival_calculator.compute_baseline_hazard(w=self.coef_)

    def compute_hazard(self, X):
        self.check_is_fitted_estimator('coef_')
        return self.hazard_survival_calculator.get_cumulative_hazard_function(w=self.coef_)

    def compute_survival(self, X):
        self.check_is_fitted_estimator('coef_')
        return self.hazard_survival_calculator.get_survival_function(w=self.coef_)

    def callback(self, iteration, w, grad, verbose, early_stop=False):
        """Callback to log progress."""
        if verbose:
            print(f"\nIteration {iteration}: Coefficients = {w}")
            print(f"Gradient Norm: {np.linalg.norm(grad) if grad is not None else 'N/A'}")
            loss = self.objective_function.compute_loss(w)
            print(f"Iteration {iteration}, Loss: {loss}")
            if early_stop:
                print(f"Early stopping at iteration {iteration}.")
