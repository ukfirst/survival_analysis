import numpy as np
import pandas as pd


def generate_covariates(n_individuals, n_transitions, n_static_covariates=2, n_time_dependent_covariates=2):
    # Static covariates (time-independent) for each individual
    Z_i1 = np.random.uniform(0, 1, (n_individuals, n_static_covariates))

    # Time-dependent covariates for each transition (varied slightly over each transition)
    Z_ij1 = []
    for i in range(n_individuals):
        time_dependent_covariates = []
        for _ in range(n_transitions):
            covariate_shift = np.random.normal(0, 0.1, n_time_dependent_covariates)  # small shifts
            time_dependent_covariates.append(np.clip(Z_i1[i] + covariate_shift, 0, 1))  # Ensure they remain in [0, 1]
        Z_ij1.append(time_dependent_covariates)

    return Z_i1, Z_ij1
def simulate_transition_times(Z_i1, Z_ij1, base_hazard=0.5, max_transitions=5):
    n_individuals = len(Z_i1)
    individuals, time, event_from, event_to, censorship = [], [], [], [], []

    for i in range(n_individuals):
        transitions = 0
        current_time = 0
        current_state = 1  # Start in state 1

        while transitions < max_transitions:
            # Hazard based on time-independent and time-dependent covariates
            static_contrib = np.dot(Z_i1[i], [2, 1])  # Weights for static covariates
            time_dep_contrib = np.dot(Z_ij1[i][transitions], [1.5, 0.5])  # Weights for time-dependent covariates
            hazard_rate = base_hazard * np.exp(static_contrib + time_dep_contrib)

            # Simulate sojourn time
            sojourn_time = np.random.exponential(1 / hazard_rate)
            current_time += sojourn_time

            # Record transition information
            individuals.append(i)
            time.append(current_time)
            event_from.append(current_state)
            event_to.append(2 if current_state == 1 else 1)  # Alternate between states
            censorship.append(0)  # Assuming no censoring for simplicity

            current_state = event_to[-1]
            transitions += 1

    return pd.DataFrame({
        'individuals': individuals,
        'time': time,
        'event_from': event_from,
        'event_to': event_to,
        'censorship': censorship
    })
def generate_simulated_alternating_state_data(n_individuals=50, max_transitions=5):
    # Generate covariates
    Z_i1, Z_ij1 = generate_covariates(n_individuals, max_transitions)

    # Generate transitions
    transitions_df = simulate_transition_times(Z_i1, Z_ij1, max_transitions=max_transitions)

    # Add covariates to DataFrame
    transitions_df['Z_i1'] = [Z_i1[i] for i in transitions_df['individuals']]
    transitions_df['Z_ij1'] = [Z_ij1[i][t] for i, t in zip(transitions_df['individuals'], transitions_df.groupby('individuals').cumcount())]

    return transitions_df

if __name__ == "__main__":
    individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = generate_simulated_alternating_state_data(n_individuals=50, max_transitions=5)
    print(time)
