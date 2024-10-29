import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

class SurvivalCalibration:
    def __init__(self, model, dataset_manager, data_view):
        """
        Initialize calibration curve for a survival model.

        Parameters:
        - model: An instance of TwoStateHazard or other survival model.
        - dataset_manager: Instance of DatasetManager.
        - data_view: Filtered DataFrame from DatasetManager for transitions.
        """
        self.model = model
        self.dataset_manager = dataset_manager
        self.data_view = data_view

    def compute_calibration_curve(self, time_points, verbose=False):
        """
        Compute and plot calibration curve over specified time points.

        Parameters:
        - time_points: Array of time points to evaluate survival probabilities.
        - verbose: If True, print details of each calculation.

        Returns:
        - Plots the calibration curve at different time points.
        """
        # Extract data
        individuals = np.unique(self.data_view['individuals'].values)
        times = self.data_view['time'].values
        events = self.data_view['event'].values
        Z_i1 = self.dataset_manager.Z_i1(self.data_view)
        Z_ij1 = self.dataset_manager.Z_ij1(self.data_view)

        plt.figure(figsize=(10, 6))
        for time in time_points:
            predicted_probs = []
            observed_probs = []

            for idx, individual in enumerate(individuals):
                Z_i1_indiv = Z_i1[idx]
                Z_ij1_indiv = Z_ij1[idx]

                # Predict survival probability for individual
                survival_func = self.model.get_survival_function()
                predicted_survival = survival_func[idx](time)
                predicted_probs.append(predicted_survival)

                # Compute Kaplan-Meier for observed survival probability
                kmf = KaplanMeierFitter()
                kmf.fit(times, event_observed=events)
                observed_survival = kmf.predict(time)
                observed_probs.append(observed_survival)

                if verbose:
                    print(f"Time: {time}, Predicted: {predicted_survival}, Observed: {observed_survival}")

            # Plot calibration at each time point
            plt.plot(predicted_probs, observed_probs, 'o', label=f'Time {time}')

        plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
        plt.xlabel("Predicted Survival Probability")
        plt.ylabel("Observed Survival Probability")
        plt.title("Calibration Plot for Survival Predictions")
        plt.legend()
        plt.show()
