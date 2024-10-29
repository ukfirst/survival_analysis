# Function to generate synthetic data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def make_simple_data(n_samples=100, n_noise_features=2, base_hazard=0.2, percent_censor=0.3):
    np.random.seed(42)
    """Generates a synthetic survival dataset with linear hazard."""
    x = np.random.standard_normal((n_samples, n_noise_features + 2))
    hazards = x[:, 0] + 2 * x[:, 1]
    event_time = np.random.exponential(1 / (base_hazard * np.exp(hazards)))
    # Generate random censoring times, independent of event times
    censor_time = np.random.exponential(scale=np.percentile(event_time, 75), size=n_samples)

    time = np.minimum(event_time, censor_time)
    event = (event_time < censor_time).astype(int)

    return pd.DataFrame({
        "time": time,
        "event": event,
        **{f"x{i + 1}": x[:, i] for i in range(x.shape[1])}
    })

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def make_transition_data(n_samples=100, n_x_noise_features=0, n_y_noise_features=0,
                         base_hazard=0.2, max_transitions=5, p_transition=0.7):
    """
    Generates a synthetic dataset for a two-state process with probabilistic transitions and censoring.

    Parameters:
    - p_transition: Probability of transitioning to the other state after a sojourn time.
    """
    np.random.seed(42)
    data = []

    # Generate static covariates (with noise features)
    x = np.random.standard_normal((n_samples, 2 + n_x_noise_features))
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    x = scaler_x.fit_transform(x)
    x_hazards = x[:, 0] + 4 * x[:, 1]

    # Generate time-dependent covariates (with noise features)
    y = np.random.standard_normal((n_samples, 2 + n_y_noise_features))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y = scaler_y.fit_transform(y)
    y_hazards = y[:, 0] + y[:, 1]

    for i in range(n_samples):
        n_transitions = np.random.randint(2, max_transitions + 1)
        times = []
        states = []
        sojourn_times = []
        current_state = 1  # Start in state 1
        tt = []  # Time-dependent covariates

        total_time = 0

        for t in range(n_transitions):
            # Sojourn time based on exponential distribution
            sojourn_time = np.random.exponential(1 / (base_hazard * np.exp(x_hazards[i])))
            total_time += sojourn_time
            sojourn_times.append(sojourn_time)
            times.append(total_time)
            states.append(current_state)

            # Generate time-dependent covariates
            time_dependent_covariates = np.random.normal(
                loc=y_hazards[i], scale=0.1, size=2 + n_y_noise_features
            )
            tt.append(time_dependent_covariates)

            # Decide whether to transition to the other state
            if np.random.rand() < p_transition:
                current_state = 3 - current_state  # Transition to the other state
            else:
                pass  # Remain in the same state

        # Generate censoring time and apply it
        censor_time = np.random.exponential(scale=np.percentile(times, 75))

        # Initialize censorship flags
        censorship = [0]*len(times)

        # Apply censoring
        censored_times = []
        censored_states = []
        censored_sojourn_times = []
        censored_tt = []
        for idx, t_time in enumerate(times):
            if t_time <= censor_time:
                censored_times.append(t_time)
                censored_states.append(states[idx])
                censored_sojourn_times.append(sojourn_times[idx])
                censored_tt.append(tt[idx])
            else:
                # Adjust the sojourn time for the last observed state
                if idx == 0:
                    # Censored before first transition
                    censored_times.append(censor_time)
                    censored_states.append(states[idx])
                    censored_sojourn_times.append(censor_time)
                    censored_tt.append(tt[idx])
                    censorship[idx] = 1  # Censored
                else:
                    # Adjust sojourn time of the last included observation
                    censored_sojourn_times[-1] = censor_time - times[idx-1]
                    censorship[idx-1] = 1  # Mark the previous event as censored
                break

        else:
            # No censoring applied
            censored_times = times
            censored_states = states
            censored_sojourn_times = sojourn_times
            censored_tt = tt

        # Adjust the length of lists to match censored data
        censorship = censorship[:len(censored_times)]

        # Collect data
        for j in range(len(censored_times)):
            data.append({
                'individual': i,
                'time': censored_times[j],
                'state': 1 if censored_states[j] == 1 else 0,
                'sojourn_time': censored_sojourn_times[j],
                'censorship': censorship[j],
                **{f"x{k + 1}": x[i, k] for k in range(x.shape[1])},
                **{f"y{k + 1}": censored_tt[j][k] for k in range(len(censored_tt[j]))},
            })

    df = pd.DataFrame(data)
    return df


def get_combined_dataset():
    # Define the dataset components for a larger dataset with final corrections (doubled size)
    individuals = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5,
                            6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 11, 11])
    time = np.array([0, 1, 3, 5, 0, 2, 3, 5, 0, 2, 4, 0, 1, 3, 0, 3, 0, 2,
                     0, 1, 3, 5, 0, 2, 3, 5, 0, 2, 4, 0, 1, 3, 0, 3, 0, 2])
    censorship = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                           0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
    event_from = np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2,
                           1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1,
                           2])  # Corrected to align with actual states
    event_to = np.array([1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                         1, 2, 2, 2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])  # Corrected to align with next states

    # Individual-level covariates (Z_i1)
    Z_i1 = np.array([
        [0.5, -0.04],  # Individual 0
        [1.0, 0.82],  # Individual 1
        [1.3, -0.18],  # Individual 2
        [1.5, 0.43],  # Individual 3
        [1.2, 0.5],  # Individual 4
        [0.8, -0.2],  # Individual 5
        [0.6, 0.12],  # Individual 6 (New)
        [1.4, -0.13],  # Individual 7 (New)
        [1.1, 0.35],  # Individual 8 (New)
        [0.9, -0.08],  # Individual 9 (New)
        [1.2, 0.3],  # Individual 10 (New)
        [0.7, -0.25]  # Individual 11 (New)
    ])

    # Time-varying covariates (Z_ij1) for each individual (2 variables per individual)
    Z_ij1 = np.array([
        [[0.5, -0.04], [0.6, 0.8], [0.7, -0.5], [0.8, 0.9]],  # Individual 0
        [[1.0, 0.82], [0.8, 0.76], [1.1, 0.7], [1.2, 0.6]],  # Individual 1
        [[1.3, -0.18], [0.7, 0.21], [0.9, 0.3]],  # Individual 2
        [[1.5, 0.43], [0.9, 0.64], [1.0, 0.5]],  # Individual 3
        [[1.2, 0.5], [1.3, 0.4]],  # Individual 4
        [[0.8, -0.2], [0.9, 0.1]],  # Individual 5
        [[0.6, 0.12], [0.7, -0.15], [0.8, 0.23], [0.9, -0.33]],  # Individual 6 (New)
        [[1.4, -0.13], [1.2, 0.55], [1.0, 0.45], [1.3, -0.25]],  # Individual 7 (New)
        [[1.1, 0.35], [0.8, -0.08], [0.9, 0.5]],  # Individual 8 (New)
        [[0.9, -0.08], [1.0, 0.4], [1.1, -0.05]],  # Individual 9 (New)
        [[1.2, 0.3], [1.1, -0.1], [1.3, 0.6]],  # Individual 10 (New)
        [[0.7, -0.25], [0.9, 0.15]]  # Individual 11 (New)
    ], dtype=object)  # Keep dtype=object to handle variable-length sequences

    return individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1

def bigger_dataset(n_individuals= 30, max_time= 5, n_static_covariates= 2, n_time_dependent_covariates= 2, seed=42):
    np.random.seed(seed)  # Set the seed for reproducibility

    # Generate individual IDs
    observations_per_individual = np.random.randint(2, 5, size=n_individuals)  # Random between 2 and 4 observations
    individuals = np.repeat(np.arange(n_individuals), 4)  # Each individual has 4 observations

    # Generate random times for each observation, capped by max_time
    time = np.sort(np.random.randint(0, max_time, size=len(individuals)))

    # Random censorship (0 or 1)
    censorship = np.random.choice([0, 1], size=len(individuals), p=[0.85, 0.15])  # 15% chance for censorship

    # Generate event_from and event_to, based on states (1 or 2)
    event_from = np.random.choice([1, 2], size=len(individuals), p=[0.95, 0.05])
    event_to = np.array([np.random.choice([state for state in [1, 2] if state != ef]) for ef in event_from])

    # Generate individual-level covariates (Z_i1) for each individual within the range [-1, 1]
    Z_i1 = 2 * np.random.rand(n_individuals, n_static_covariates) - 1

    Z_ij1 = []
    for i in range(n_individuals):
        Z_ij1_individual = []
        # Create time-varying covariates for each observation per individual
        for _ in range(observations_per_individual[i]):  # Variable number of observations per individual
            Z_ij1_individual.append(2 * np.random.rand(n_time_dependent_covariates) - 1)
        Z_ij1.append(np.array(Z_ij1_individual, dtype=float))

    return individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1


import numpy as np
from sklearn.preprocessing import MinMaxScaler


def gamma_sojourn_time_normalized(scale, shape=2):
    """Generate sojourn time using a Gamma distribution with a specified shape and scale, normalized by the shape."""
    return np.random.gamma(shape=shape, scale=scale)  # Normalize by shape to match the expected mean


def positive_hazard_transition_dataset(n_samples=25, base_hazard=2.0, max_transitions=3, p_transition=1):
    """
    Generates a synthetic dataset for a two-state process with hazard-based transitions.

    Parameters:
    - n_samples: Number of individuals.
    - base_hazard: Baseline hazard rate affecting transitions.
    - max_transitions: Maximum number of transitions allowed for each individual.
    - p_transition: Probability of transitioning to the other state after a sojourn time.

    Returns:
    - individuals: Array of individual indices.
    - time: Array of time points.
    - event_from: State before transition.
    - event_to: State after transition.
    - censorship: Array indicating if the individual was censored (1 for censored, 0 for not).
    - Z_i1: Individual-level covariates.
    - Z_ij1: Time-varying covariates.
    """
    np.random.seed(42)

    # Initialize variables to collect the data
    individuals = []
    time = []
    event_from = []
    event_to = []
    censorship = []
    Z_i1 = []
    Z_ij1 = []

    # Generate individual-level covariates (Z_i1) from a uniform distribution [0, 1]
    Z_i1_data = np.random.uniform(low=0, high=1, size=(n_samples, 2))

    for i in range(n_samples):
        transitions = 0
        current_time = 0
        current_state = 1  # Start in state 1 for all individuals

        # Store individual-level covariates
        Z_i1.append(Z_i1_data[i])

        # Time-varying covariates initialization for the individual
        time_varying_covariates = []
        individual_times = []
        individual_events_from = []
        individual_events_to = []
        individual_censorship = []

        while transitions < max_transitions:
            # Calculate the hazard directly based on covariates Z_i1
            hazard = base_hazard * (2 * Z_i1_data[i, 0] + 1 * Z_i1_data[i, 1])

            # Ensure hazard is positive
            hazard = max(hazard, 1e-3)

            # Sojourn time (time spent in the current state) determined by the hazard function
            sojourn_time = np.random.exponential(1 / hazard)

            # Accumulate the time for the individual
            current_time += sojourn_time
            individual_times.append(current_time)

            # Determine transition based on probability
            next_state = 2 if current_state == 1 else 1

            individual_events_from.append(current_state)
            individual_events_to.append(next_state)
            current_state = next_state  # Update the state

            # Append time-varying covariates for each transition (using the same covariates across all transitions)
            time_varying_covariate = Z_i1_data[i] + np.random.normal(0, 0.1, size=Z_i1_data[i].shape)
            time_varying_covariates.append(time_varying_covariate)

            # No censorship for simplicity
            individual_censorship.append(0)

            transitions += 1

        # Collect individual data
        individuals.extend([i] * len(individual_times))
        time.extend(individual_times)
        event_from.extend(individual_events_from)
        event_to.extend(individual_events_to)
        censorship.extend(individual_censorship)
        Z_ij1.append(time_varying_covariates)

    # Convert lists to numpy arrays
    individuals = np.array(individuals)
    time = np.array(time)
    event_from = np.array(event_from)
    event_to = np.array(event_to)
    censorship = np.array(censorship)
    Z_i1 = np.array(Z_i1)
    Z_ij1 = np.array(Z_ij1, dtype=object)

    return individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    # Generate the synthetic dataset with time-dependent and independent covariates
    individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = positive_hazard_transition_dataset()

    # Flatten the time-dependent covariates (Z_ij1) to match transitions
    Z_ij1_flattened = np.vstack(Z_ij1)  # Flatten time-dependent covariates

    # Repeat the individual-level covariates (Z_i1) for each transition
    Z_i1_repeated = np.repeat(Z_i1, [len(zi) for zi in Z_ij1], axis=0)

    # Combine time-independent and time-dependent covariates into one feature matrix
    X = np.hstack([Z_i1_repeated, Z_ij1_flattened])  # Combine both covariate types

    # Compute the true hazard values based on the individual covariates (to simulate the real hazard function)
    true_hazards = 2 * Z_i1_repeated[:, 0] + 1 * Z_i1_repeated[:, 1] + 1.5 * Z_ij1_flattened[:,
                                                                             0] + 0.5 * Z_ij1_flattened[:, 1]

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, true_hazards)

    # Print the estimated coefficients for both time-independent and time-dependent covariates
    print("True coefficients: [2, 1] for static covariates, [1.5, 0.5] for time-varying covariates")
    print("Estimated coefficients from linear regression:", model.coef_)

    # Make predictions to check the fit
    predictions = model.predict(X)

    # Print some prediction examples to check
    print("\nSome predictions:")
    for i in range(5):
        print(f"Z_i1[{i}] = {Z_i1_repeated[i]}, Z_ij1[{i}] = {Z_ij1_flattened[i]} -> "
              f"Predicted Hazard = {predictions[i]:.4f}, True Hazard = {true_hazards[i]:.4f}")