import numpy as np
import pandas as pd
from surv_optimizer.data.SyntheticDataset import get_combined_dataset, bigger_dataset, positive_hazard_transition_dataset
from surv_optimizer.data.TwoStatesDataset import AlternatingStateDatasetGenerator

class DatasetManager:
    def __init__(self):
        # Generate the dataset
        #individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = positive_hazard_transition_dataset()
        # Instantiate dataset generator
        dataset_generator = AlternatingStateDatasetGenerator(n_individuals=50, n_transitions=5)

        # Generate dataset fields
        individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = dataset_generator.generate_dataset()

        # Create a DataFrame for observation-level data
        self.observation_data = pd.DataFrame({
            'individuals': individuals,
            'time': time,
            'censorship': censorship,
            'event_from': event_from,
            'event_to': event_to
        })

        # Create a DataFrame for individual-level data
        individual_df = pd.DataFrame({
            'individuals': np.arange(len(Z_i1)),
            'Z_i1': list(Z_i1),
            'Z_ij1_list': list(Z_ij1)
        })

        # Explode 'Z_ij1_list' to get per-observation 'Z_ij1'
        individual_df = individual_df.explode('Z_ij1_list').reset_index(drop=True)
        individual_df.rename(columns={'Z_ij1_list': 'Z_ij1'}, inplace=True)

        # Create an observation index per individual to align the data
        self.observation_data['obs_index'] = self.observation_data.groupby('individuals').cumcount()
        individual_df['obs_index'] = individual_df.groupby('individuals').cumcount()

        # Merge per-observation covariates into observation data
        self.data = pd.merge(self.observation_data, individual_df, on=['individuals', 'obs_index'], how='left')

        # Drop 'obs_index' as it's no longer needed
        self.data.drop('obs_index', axis=1, inplace=True)

        # Compute unique states
        self.unique_states = set(np.unique(event_from).tolist() + np.unique(event_to).tolist())

        # Now, 'Z_ij1' is per-observation and correctly aligned
        # Proceed to fill forward any missing covariates if needed
        self._fill_forward_Z_ij1()
        self.direction = None

    def get_dataview(self, state_from=1, sorted=False):
        """
        Get a filtered view of the data based on the direction and sorting.

        Parameters:
        - direction: The state from which transitions are considered (e.g., 1 or 2).
        - sorted: Whether to return data sorted by time.

        Returns:
        - A DataFrame view of the data filtered by direction and sorted as specified.
        """
        # Exclude self-transitions
        self.direction = state_from
        possible_event_to_states = self.unique_states - {state_from}

        # Create the condition: event_from equals direction, and event_to is in possible_event_to_states
        condition = (self.data['event_from'] == state_from) & \
                    (self.data['event_to'].isin(possible_event_to_states))

        # Compute the event variable: 1 for transitions, 0 otherwise
        self.data['event'] = np.where(condition, 1, 0)

        # Filter data where event_from equals state_from
        #dataset = self.data[self.data['event_from'] == state_from]
        dataset = self.data
        # Sort the data by time if needed
        if sorted:
            dataset = dataset.sort_values(by='time')

        # Return the filtered data as a view (no data duplication)
        return dataset

    def Z_i1(self, data_view):
        """Return Z_i1 as a 2D numpy array."""
        return np.vstack(data_view['Z_i1'].values)

    def Z_ij1(self, data_view):
        """Return Z_ij1_filled as a 2D numpy array."""
        return np.vstack(data_view['Z_ij1_filled'].values)

    # Properties for dimensions
    @property
    def state_from(self):
        return self.direction
    @property
    def n_samples(self):
        """Return the number of samples (observations)."""
        return len(self.data)

    @property
    def n_static_covariates(self):
        """Return the number of static covariates."""
        return len(self.data['Z_i1'].iloc[0])

    @property
    def n_time_dependent_covariates(self):
        """Determine the number of time-dependent covariates per time point."""
        return len(self.data['Z_ij1'].iloc[0])

    @property
    def n_fading_coefficients(self):
        """Return the number of fading coefficients."""
        return self.n_time_dependent_covariates

    @property
    def n_total_covariates(self):
        """Return the total number of covariates."""
        return self.n_static_covariates + self.n_time_dependent_covariates #+ self.n_fading_coefficients

    # Helper methods
    def get_data(self, keys, data_view):
        """
        Retrieve data based on the keys provided (e.g., ['Z_i1', 'event']).

        Parameters:
        - keys: List of keys of the data to retrieve.
        - sorted: If True, return sorted data. If False, return original data.

        Returns:
        - Pandas DataFrame with the requested data.
        """
        data = data_view
        return data[keys]

    def filter_by_time_individual(self, data_view, time_limit=None, individual_id=None):
        """
        Get filtered data based on time limit and/or individual ID.

        Parameters:
        - time_limit: The time limit for filtering (optional).
        - individual_id: The individual identifier for filtering (optional).
        - sorted: If True, return sorted data. If False, return original data.

        Returns:
        - Filtered DataFrame.
        """
        data = data_view
        mask = pd.Series(True, index=data.index)
        if time_limit is not None:
            mask &= data['time'] <= time_limit
        if individual_id is not None:
            mask &= data['individuals'] == individual_id
        return data[mask]

    def get_individual_mask(self, individual_id, data_view):
        """
        Returns:
        - Boolean mask for filtering by individual ID.
        """
        return data_view['individuals'] == individual_id

    def get_unique_times(self):
        """
        Returns:
        - Numpy array of unique times.
        """
        return np.sort(self.data['time'].unique())

    def get_unique_individuals(self):
        """
        Returns:
        - Numpy array of unique individuals.
        """
        return np.sort(self.data['individuals'].unique())

    def get_all_times(self, data_view):
        """
        Returns:
        - Pandas Series of times.
        """
        return data_view['time']

    def get_all_individuals(self, data_view):
        """
        Return all individuals across the dataset.

        Returns:
        - Pandas Series of individuals.
        """

        return data_view['individuals']

    def get_time_bins_before(self, time_limit, data_view):
        """
        Get data for all time points before or equal to the time limit.

        Parameters:
        - time_limit: The time limit for filtering.
        Returns:
        - Filtered DataFrame.
        """
        data = data_view
        return data[data['time'] <= time_limit]

    def get_time_bins_after(self, time_limit, data_view):
        """
        Get data for all time points after the time limit.

        Parameters:
        - time_limit: The time limit for filtering.

        Returns:
        - Filtered DataFrame.
        """
        data = data_view
        return data[data['time'] > time_limit]

    # Event statistics methods
    def get_event_counts(self, data_view):
        """
        Get the number of events at each unique time.

        Returns:
        - Pandas Series with times as index and event counts as values.
        """
        data = data_view
        event_counts = data.groupby('time')['event'].sum()
        return event_counts

    def get_event_stats(self):
        """
        Get event statistics including event counts at each unique time..

        Returns:
        - DataFrame with columns 'time', 'events'.
        """
        event_counts = self.get_event_counts()
        event_stats = pd.DataFrame({
            'time': event_counts.index,
            'events': event_counts.values,
        })
        return event_stats

    def _fill_forward_Z_ij1(self):
        """Assign time-varying covariate values to observations."""
        # Since 'Z_ij1' is now per-observation and correctly aligned, we can directly assign it
        self.data['Z_ij1_filled'] = self.data['Z_ij1']


if __name__ == '__main__':
    dataset_manager = DatasetManager()
    # Test Case 1: Check if filtered data (direction=1) is correctly handled
    print("Test Case 1: Filtered Data for Direction 1 (sorted=True)")
    filtered_data_direction_3 = dataset_manager.get_dataview(state_from=1, sorted=True)
    print(filtered_data_direction_3)
    filtered_data_direction_1 = dataset_manager.get_dataview(state_from=1, sorted=False)
    print(filtered_data_direction_1)