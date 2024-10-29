import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index


from lifelines.utils import concordance_index
import numpy as np

from lifelines.utils import concordance_index
import numpy as np

from lifelines.utils import concordance_index
import numpy as np

def calculate_concordance_index(objective_function, data):
    """
    Calculate the concordance index using the provided objective function and dataset.

    Parameters:
    - objective_function: The model objective function instance (e.g., TwoStateCoxObjectiveFunction).
    - data: The dataset including features, time, and event columns.

    Returns:
    - Concordance index value.
    """
    # Check for the exact column names
    print("Columns in DataFrame:", data.columns)

    # Attempt to identify time and event columns
    time_col_candidates = ['time', 'transition_time']
    event_col_candidates = ['event', 'censorship']

    time_col = next((col for col in time_col_candidates if col in data.columns), None)
    event_col = next((col for col in event_col_candidates if col in data.columns), None)

    if time_col is None or event_col is None:
        raise KeyError(f"Time column and/or event column not found in DataFrame. Available columns: {list(data.columns)}")

    # Extract feature names, excluding time and event
    feature_names = ['x1', 'x2', 'y1', 'y2']

    # Extract feature values
    X = data[feature_names].values

    # Extract survival time and event status
    time = data[time_col].values
    event = data[event_col].values

    # Get model coefficients - use only the first four weights for the four features
    w = objective_function.coef_[:4]  # Assuming the coefficients are stored in objective_function after fitting

    # Calculate the linear predictor
    risk_scores = np.dot(X, w)

    # Calculate concordance index
    c_index = concordance_index(time, -risk_scores, event)

    return c_index



def prepare_time_for_model(data):
    """
    Prepare the time data for use with sksurv CoxPHSurvivalAnalysis model.

    Parameters:
    - data: Pandas DataFrame containing 'event' and 'time' columns.

    Returns:
    - A structured NumPy array suitable for fitting the sksurv CoxPHSurvivalAnalysis model.
    """
    if isinstance(data, pd.DataFrame):
        if 'event' in data.columns and 'time' in data.columns:
            # If 'event' and 'time' columns are present in the DataFrame, prepare them appropriately
            return np.array(list(zip(data['event'].astype(bool), data['time'])), dtype=np.dtype("bool, float"))
        else:
            raise ValueError("Data must contain 'event' and 'time' columns for preparing the time variable.")
    elif isinstance(data, np.ndarray) and data.dtype.names == ('event', 'time'):
        # If the data is already in structured form
        return data
    else:
        raise ValueError(
            "Unsupported time data format. Please provide a DataFrame with 'event' and 'time' or a structured array.")


def create_sksurv_model(data, covariates):
    """
    Creates and fits a sksurv Cox proportional hazards model using the provided data.

    Parameters:
    - data: The DataFrame containing all the information.
    - covariates: List of column names representing covariates for the Cox model.

    Returns:
    - sksurv_model: A fitted Cox proportional hazards model.
    """
    # Prepare the covariate matrix (linear predictor)
    X = data[covariates].values

    # Prepare the structured array with event indicators and time values
    # The 'from_state' column represents whether an event occurred (1 = event, 0 = no event)
    # The 'time' column contains the event or censoring time
    y = np.array(
        list(zip(data['censorship'].astype(bool), data['time'])),
        dtype=[('event', 'bool'), ('time', 'float')])

    # Fit the Cox proportional hazards model using the covariates, event, and time
    sksurv_model = CoxPHSurvivalAnalysis()
    sksurv_model.fit(X, y)

    return sksurv_model

class ModelQualityCheck:

    def __init__(self, data, sksurv_model, covariate_columns, n_to_plot=10):
        """
        Initialize the quality check class.

        Parameters:
        - data: DataFrame containing the covariates, event, and time columns.
        - time_range: The range of time points at which to compare the models.
        - sksurv_model: The fitted sksurv CoxPHSurvivalAnalysis model.
        - n_to_plot: Number of samples to plot for comparison.
        """
        self.data = data.copy()
        self.time_range = None
        self.sksurv_model = sksurv_model
        self.n_to_plot = n_to_plot
        self.covariate_columns = covariate_columns

    def compare_coefficients(self, coefficients):
        """
        Compare the coefficients between the sksurv model and the custom model.

        Parameters:
        - coefficients: Coefficients obtained from the custom model.

        Returns:
        - A DataFrame summarizing the differences between the coefficients.
        """
        sksurv_coefficients = self.sksurv_model.coef_

        # Ensure both coefficients vectors are of the same length for comparison
        # Assuming the first n_static_covariates represent the shared coefficients
        n_common_coefficients = len(sksurv_coefficients)

        # Select the corresponding parts of the coefficient vectors
        sksurv_coefficients_common = sksurv_coefficients[:n_common_coefficients]
        custom_coefficients_common = coefficients[:n_common_coefficients]

        # Calculate element-wise difference
        diff = sksurv_coefficients_common - custom_coefficients_common

        # Create a DataFrame to summarize the comparison
        coefficient_names = [f'x{i + 1}' for i in range(n_common_coefficients)]
        comparison_df = pd.DataFrame({
            'Coefficient': coefficient_names,
            'Custom Coefficient': custom_coefficients_common,
            'skSurv Coefficient': sksurv_coefficients_common,
            'Difference': diff
        })

        print(comparison_df)
        return comparison_df

    def compare_survival_plot(self, optimizer, final_coefficients):
        """Plot survival function comparison between sksurv and custom model."""
        sksurv_funcs = self.sksurv_model.predict_survival_function(self.data[["x1", "x2", "x3", "x4"]])
        custom_surv_funcs = optimizer.get_survival_function(final_coefficients)

        fig, axes = plt.subplots(nrows=self.n_to_plot, ncols=1, figsize=(10, 5 * self.n_to_plot), dpi=120, tight_layout=True)

        if self.n_to_plot == 1:
            axes = [axes]  # Convert to list for consistency

        for i in range(self.n_to_plot):
            axes[i].plot(self.time_range, [sksurv_funcs[i](t) for t in self.time_range], label=f"skSurv Survival {i + 1}", linestyle="--")
            axes[i].plot(self.time_range, [custom_surv_funcs[i](t) for t in self.time_range], label=f"Custom Survival {i + 1}", linestyle="-")
            axes[i].set_title(f"Survival Curve Comparison (Sample {i + 1})")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Survival Probability")
            axes[i].legend()

        plt.show()

    def compare_hazard_plot(self, optimizer, final_coefficients):
        """Plot hazard function comparison between sksurv and custom model."""
        sksurv_hazard_funcs = self.sksurv_model.predict_cumulative_hazard_function(self.data[["x1", "x2", "x3", "x4"]])
        custom_hazard_funcs = optimizer.get_cumulative_hazard_function(final_coefficients)

        fig, axes = plt.subplots(nrows=self.n_to_plot, ncols=1, figsize=(10, 5 * self.n_to_plot), dpi=120, tight_layout=True)

        if self.n_to_plot == 1:
            axes = [axes]  # Convert to list for consistency

        for i in range(self.n_to_plot):
            axes[i].plot(self.time_range, [sksurv_hazard_funcs[i](t) for t in self.time_range], label=f"skSurv Hazard {i + 1}", linestyle="--")
            axes[i].plot(self.time_range, [custom_hazard_funcs[i](t) for t in self.time_range], label=f"Custom Hazard {i + 1}", linestyle="-")
            axes[i].set_title(f"Hazard Curve Comparison (Sample {i + 1})")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Hazard")
            axes[i].legend()

        plt.show()