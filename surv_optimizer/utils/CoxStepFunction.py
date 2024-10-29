import numpy as np
from sklearn.utils import check_consistent_length


class StepFunction:
    """Callable step function."""

    def __init__(self, x, y, *, a=1.0, b=0.0, domain=(0, None)):
        check_consistent_length(x, y)
        self.x = x
        self.y = y
        self.a = a
        self.b = b
        domain_lower = self.x[0] if domain[0] is None else domain[0]
        domain_upper = self.x[-1] if domain[1] is None else domain[1]
        self._domain = (float(domain_lower), float(domain_upper))

    def __call__(self, x, all_points=False):
        """
        Evaluate the step function at specific points or return all values at once.

        Parameters:
        - x: Single value or array-like of points at which to evaluate the function.
        - all_points (bool): If True, ignore `x` and return the function evaluated at all predefined points.

        Returns:
        - Function values at `x` if `all_points` is False.
        - Entire function values if `all_points` is True.
        """
        if all_points:
            # Return all values in `y` directly for the predefined `x` values.
            return self.a * self.y + self.b

        # Existing behavior for individual evaluation.
        x = np.atleast_1d(x)
        x = np.clip(x, a_min=self.x[0], a_max=self.x[-1])  # Clip to the function's domain
        i = np.searchsorted(self.x, x, side="left")
        not_exact = self.x[i] != x
        i[not_exact] -= 1
        value = self.a * self.y[i] + self.b
        return value[0] if value.shape[0] == 1 else value