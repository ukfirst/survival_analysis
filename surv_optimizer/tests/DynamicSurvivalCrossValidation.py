import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

class DynamicSurvivalCrossValidation:
    def __init__(self, model, dataset_manager, data_view, n_folds=5):
        """
        Initialize cross-validation for a survival model.

        Parameters:
        - model: An instance of TwoStateHazard or another model to evaluate.
        - dataset_manager: The DatasetManager instance.
        - data_view: Filtered DataFrame from DatasetManager.
        - n_folds: Number of folds for cross-validation.
        """
        self.model = model
        self.dataset_manager = dataset_manager
        self.data_view = data_view
        self.n_folds = n_folds

    def run_cross_validation(self, verbose=False):
        """
        Perform K-Fold Cross-Validation and evaluate the model on each fold.

        Parameters:
        - verbose: If True, print detailed information about each fold.

        Returns:
        - Dictionary of averaged evaluation metrics.
        """
        metrics = {'c_index': [], 'brier_score': []}

        # Extract data
        individuals = self.data_view['individuals'].values
        times = self.data_view['time'].values
        events = self.data_view['event'].values
        Z_i1 = self.dataset_manager.Z_i1(self.data_view)
        Z_ij1 = self.dataset_manager.Z_ij1(self.data_view)

        data = np.column_stack([individuals, times, events])

        kf = KFold(n_splits=self.n_folds)
        for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
            if verbose:
                print(f"Running fold {fold+1}/{self.n_folds}")

            train_data = data[train_idx]
            test_data = data[test_idx]

            # Fit the model on training data
            self.model.compute_baseline_hazard(train_data)

            # Evaluate on test data
            fold_c_index = []
            fold_brier_score = []
            for idx in test_idx:
                individual = individuals[idx]
                observed_time = times[idx]
                observed_event = events[idx]
                Z_i1_indiv = Z_i1[individual]
                Z_ij1_indiv = Z_ij1[individual]

                # Predict cumulative hazard
                cumulative_hazard = self.model.get_cumulative_hazard_function()

                # Calculate C-index
                c_index = concordance_index(observed_time, -cumulative_hazard, observed_event)
                fold_c_index.append(c_index)

                # Calculate Brier score using KM estimator
                kmf = KaplanMeierFitter()
                kmf.fit(times[train_idx], event_observed=events[train_idx])
                survival_prob = kmf.predict(observed_time)
                brier_score = np.mean((observed_event - survival_prob) ** 2)
                fold_brier_score.append(brier_score)

            # Append fold metrics
            metrics['c_index'].append(np.mean(fold_c_index))
            metrics['brier_score'].append(np.mean(fold_brier_score))

        avg_c_index = np.mean(metrics['c_index'])
        avg_brier_score = np.mean(metrics['brier_score'])

        if verbose:
            print(f"Cross-Validation Results: Average C-index = {avg_c_index:.4f}, Average Brier Score = {avg_brier_score:.4f}")

        return {'avg_c_index': avg_c_index, 'avg_brier_score': avg_brier_score}
