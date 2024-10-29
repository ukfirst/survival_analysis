import numpy as np
import pandas as pd

class AlternatingStateDatasetGenerator:
    def __init__(self, n_individuals, n_transitions, n_static_covariates=2, n_time_dependent_covariates=2):
        self.n_individuals = n_individuals
        self.n_transitions = n_transitions
        self.n_static_covariates = n_static_covariates
        self.n_time_dependent_covariates = n_time_dependent_covariates

    def generate_dataset(self):
        """
        Generate and return dataset fields (individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1)
        compatible with the expected format of DatasetManager.
        """
        transitions_df = self._generate_simulated_data()
        return self._extract_fields(transitions_df)

    def _generate_simulated_data(self):
        """Simulate alternating state dataset."""
        np.random.seed(42)  # For reproducibility
        individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = [], [], [], [], [], [], []

        # Generate static covariates (Z_i1) for each individual
        Z_i1_data = np.random.uniform(0, 1, size=(self.n_individuals, self.n_static_covariates))

        for i in range(self.n_individuals):
            transitions, current_time, current_state = 0, 0, 1  # Start in state 1 for all individuals
            time_varying_covariates, individual_times, individual_events_from, individual_events_to, individual_censorship = [], [], [], [], []

            while transitions < self.n_transitions:
                # Hazard and sojourn time
                hazard = 1 + np.sum(Z_i1_data[i])  # Simplified hazard based on static covariates
                sojourn_time = np.random.exponential(1 / hazard)
                current_time += sojourn_time

                # Collect data for each transition
                individual_times.append(current_time)
                individual_events_from.append(current_state)
                next_state = 2 if current_state == 1 else 1
                individual_events_to.append(next_state)
                current_state = next_state
                individual_censorship.append(0)  # No censorship for simplicity
                time_varying_covariates.append(np.random.uniform(0, 1, self.n_time_dependent_covariates))
                transitions += 1

            # Append individual data
            individuals.extend([i] * len(individual_times))
            time.extend(individual_times)
            event_from.extend(individual_events_from)
            event_to.extend(individual_events_to)
            censorship.extend(individual_censorship)
            Z_i1.append(Z_i1_data[i])
            Z_ij1.append(time_varying_covariates)

        # Convert lists to a DataFrame using pd.Series for multi-dimensional entries
        transitions_df = pd.DataFrame({
            'individuals': individuals,
            'time': time,
            'event_from': event_from,
            'event_to': event_to,
            'censorship': censorship,
            'Z_i1': pd.Series(Z_i1).repeat(self.n_transitions).reset_index(drop=True),
            'Z_ij1': pd.Series([item for sublist in Z_ij1 for item in sublist])
        })

        return transitions_df

    def _extract_fields(self, transitions_df):
        """Extract individual dataset fields for compatibility with DatasetManager."""
        individuals = transitions_df['individuals'].values
        time = transitions_df['time'].values
        event_from = transitions_df['event_from'].values
        event_to = transitions_df['event_to'].values
        censorship = transitions_df['censorship'].values

        # Individual-level covariates
        unique_individuals = np.unique(individuals)
        Z_i1 = np.array([transitions_df.loc[transitions_df['individuals'] == ind, 'Z_i1'].iloc[0] for ind in unique_individuals])

        # Time-varying covariates
        Z_ij1 = [transitions_df.loc[transitions_df['individuals'] == ind, 'Z_ij1'].tolist() for ind in unique_individuals]

        return individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1

if __name__ == '__main__':
    # Instantiate dataset generator
    dataset_generator = AlternatingStateDatasetGenerator(n_individuals=50, n_transitions=5)

    # Generate dataset fields
    individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = dataset_generator.generate_dataset()
    print(Z_ij1)

    # Pass these fields to DatasetManager
    # dataset_manager = DatasetManager(individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1)
