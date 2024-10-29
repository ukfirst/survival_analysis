import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class CoxPlotHelper:
    def __init__(self, hazard_survival_calculator, beta):
        """Initialize the helper with precomputed beta values and a hazard survival calculator instance."""
        self.hazard_survival_calculator = hazard_survival_calculator
        self.beta = beta

        # Compute baseline hazard if not already done
        if self.hazard_survival_calculator.baseline_hazard_ is None:
            self.hazard_survival_calculator.compute_baseline_hazard(self.beta)

        # Set these as None and lazily compute when needed
        self.survival_funcs = None
        self.hazard_funcs = None
        self.time_range = None

    def generate_survival_dataset(self, example_limit=5):
        """Generate a dataset for survival probabilities over time for each example."""
        # Compute survival functions only if they haven't been computed already
        if self.survival_funcs is None:
            self.survival_funcs = self.hazard_survival_calculator.get_survival_function(self.beta)
            self.time_range = self.hazard_survival_calculator.baseline_survival_.x

        # Limit the number of examples to the specified limit
        n_samples = min(len(self.survival_funcs), example_limit)

        # Create a DataFrame for survival data
        survival_data = pd.DataFrame(index=self.time_range)

        for i in range(n_samples):
            # Evaluate the survival function at each time point
            survival_data[f"Example_{i + 1}"] = [self.survival_funcs[i](t) for t in self.time_range]

        return survival_data

    def generate_hazard_dataset(self, example_limit=5):
        """Generate a dataset for hazard rates over time for each example."""
        # Compute hazard functions only if they haven't been computed already
        if self.hazard_funcs is None:
            self.hazard_funcs = self.hazard_survival_calculator.get_cumulative_hazard_function(self.beta)
            self.time_range = self.hazard_survival_calculator.baseline_hazard_.x

        # Limit the number of examples to the specified limit
        n_samples = min(len(self.hazard_funcs), example_limit)

        # Create a DataFrame for hazard data
        hazard_data = pd.DataFrame(index=self.time_range)

        for i in range(n_samples):
            # Evaluate the cumulative hazard function at each time point
            hazard_data[f"Example_{i + 1}"] = [self.hazard_funcs[i](t) for t in self.time_range]

        return hazard_data

    def plot_survival_and_hazard(self, example_limit=10):
        """Generate and plot survival and hazard curves for the first few examples with scaled values."""
        # Compute survival and hazard functions if not already done
        if self.survival_funcs is None:
            self.survival_funcs = self.hazard_survival_calculator.get_survival_function()
            self.time_range = self.hazard_survival_calculator.baseline_survival_.x

        if self.hazard_funcs is None:
            self.hazard_funcs = self.hazard_survival_calculator.get_cumulative_hazard_function()

        # Prepare data for survival and hazard plots
        n_to_plot = min(example_limit, len(self.survival_funcs))

        # Determine the number of rows and columns for subplots
        n_cols = 3  # 3 plots per row
        n_rows = int(np.ceil(n_to_plot / n_cols))

        # Create figure with subplots for survival and hazard
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows), dpi=120, tight_layout=True)

        # Flatten axes for easier iteration
        axes = axes.flatten()

        # Iterate through the first few examples and plot survival and hazard curves
        for i in range(n_to_plot):
            ax = axes[i]
            survival_data = [self.survival_funcs[i](t) for t in self.time_range]
            hazard_data = [self.hazard_funcs[i](t) for t in self.time_range]

            # Scaling survival and hazard to fit the same range
            max_survival = max(survival_data)
            max_hazard = max(hazard_data)
            if max_survival > 0:
                survival_data = [x / max_survival for x in survival_data]
            if max_hazard > 0:
                hazard_data = [x / max_hazard for x in hazard_data]

            # Plot survival and hazard on the same plot
            individual_index = self.hazard_survival_calculator.individuals[i]  # Get the individual index
            ax.plot(self.time_range, survival_data, label=f"Survival (Individual {individual_index})", linestyle="-")
            ax.plot(self.time_range, hazard_data, label=f"Hazard (Individual {individual_index})", linestyle="--")

            # Set title and labels
            ax.set_title(f"Survival and Hazard for Individual {individual_index}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Scaled Probability / Hazard")
            ax.legend()

        # Hide unused axes
        for j in range(n_to_plot, len(axes)):
            fig.delaxes(axes[j])

        # Show the plot
        plt.show()