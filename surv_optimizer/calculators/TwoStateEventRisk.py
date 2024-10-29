import numpy as np

class TwoStateEventRisk:
    def __init__(self, dataset_manager, data_view):
        """
        Initialize the RiskCalculator with DatasetManager and data view.

        Parameters:
        - dataset_manager: An instance of DatasetManager.
        - data_view: The filtered DataFrame view from DatasetManager.
        """
        self.dataset_manager = dataset_manager
        self.data_view = data_view

        # Extract necessary data
        self.individuals = self.data_view['individuals'].values
        self.time = self.data_view['time'].values
        self.event = self.data_view['event'].values
        self.censorship = self.data_view['censorship'].values
        self.event_from = self.data_view['event_from'].values
        self.event_to = self.data_view['event_to'].values
        self.event_counts = self.dataset_manager.get_event_counts(self.data_view)

        self.unique_individuals = np.unique(self.individuals)

        # Create the state tracker instance
        self.state_tracker = IndividualStateTracker(
            self.event_from,
            self.event_to,
            self.time,
            self.individuals
        )
        self.censorship_evaluator = CensorshipEvaluator(
            self.censorship,
            self.time,
            self.individuals
        )
        self.unique_times = self.dataset_manager.get_unique_times()

        self.state_from = dataset_manager.state_from

    def get_sojourn_time(self, individual, current_time_bin):
        """
        Calculate the sojourn time for an individual up to the current time bin.
        """
        _, state_start_time = self.state_tracker.get_current_state(individual, current_time_bin)
        sojourn_time = self.unique_times[current_time_bin] - state_start_time
        return max(sojourn_time, 0)
    def get_event_counts_at_sojourn_time(self, current_time_bin, verbose=False):
        """
        Calculate the number of events that occurred at the given sojourn time for the current time bin.

        Parameters:
        - current_time_bin: The index of the current time bin.
        - verbose: If True, print detailed debug information.

        Returns:
        - n_events: The number of events that occurred at the specified sojourn time.
        """
        n_events = 0
        current_time = self.unique_times[current_time_bin]

        # Iterate over all individuals
        for individual_id in self.unique_individuals:
            # Get the sojourn time for the individual at the current time bin
            sojourn_time = self.get_sojourn_time(individual_id, current_time_bin)
            target_time = current_time - sojourn_time

            # Find the closest time bin corresponding to the target time
            target_time_bin = np.searchsorted(self.unique_times, target_time, side='right') - 1

            if target_time_bin >= 0:
                # Get the number of events at the target time bin
                n_events += self.event_counts.iloc[target_time_bin]

                if verbose:
                    print(f"Individual: {individual_id}, Current Time Bin: {current_time_bin}, Sojourn Time: {sojourn_time}, Target Time: {target_time}, Target Time Bin: {target_time_bin}, Events Counted: {self.event_counts.iloc[target_time_bin]}")

        if verbose:
            print(f"Total number of events at sojourn time for current time bin {current_time_bin}: {n_events}")

        return n_events
    def debug_print(self, current_time_bin, individual, current_state, is_censored, sojourn_time, status):
        """Print debug information for risk calculation if verbose mode is enabled."""
        print(f"\nTime Bin: {current_time_bin}")
        print("=" * 30)
        print(f"Individual: {individual}")
        print(f"  Current State: {current_state}")
        print(f"  Is Censored: {is_censored}")
        print(f"  Sojourn Time: {sojourn_time}")
        print(f"  Status: {status}")

    def compute_risk_set_at_risk(self, time_bin_idx, verbose=False):
        """
        Calculate the indices of data points (rows in data_view) corresponding to individuals
        who are at risk at the given time bin.

        Parameters:
        - time_bin_idx: The index of the current time bin to consider.
        - verbose: If True, print detailed debug information.

        Returns:
        - List of indices of data points who are at risk for the given time bin.
        """
        risk_indices = []
        current_time_bin = self.unique_times[time_bin_idx]

        # Map individuals to their data indices
        individual_indices = {
            individual_id: np.where(self.individuals == individual_id)[0]
            for individual_id in self.unique_individuals
        }

        for individual_id in self.unique_individuals:
            indices = individual_indices[individual_id]
            times = self.time[indices]
            events = self.event[indices]
            censorships = self.censorship[indices]
            event_from = self.event_from[indices]
            event_to = self.event_to[indices]

            # Determine if the individual is censored before or at the current time
            is_censored = self.censorship_evaluator.is_censored_before_time(
                individual_id, current_time_bin
            )

            current_state, state_start_time = self.state_tracker.get_current_state(
                individual_id, current_time_bin
            )
            sojourn_time = self.get_sojourn_time(individual_id, current_time_bin)

            is_in_initial_state = current_state == self.state_from
            is_within_sojourn = sojourn_time >= 0

            if not is_censored and is_in_initial_state and is_within_sojourn:
                # Individual is at risk
                # Get the last data index before or at current_time_bin
                time_mask = times <= current_time_bin
                if np.any(time_mask):
                    # Get indices of times <= current_time_bin
                    valid_indices = indices[time_mask]
                    valid_times = times[time_mask]
                    # Get the index with the maximum time
                    last_idx_in_valid_times = np.argmax(valid_times)
                    last_idx = valid_indices[last_idx_in_valid_times]
                else:
                    # No times before current_time_bin, use earliest index
                    last_idx = indices[0]

                risk_indices.append(last_idx)
                status = "AT RISK"
            else:
                status = "NOT AT RISK"

            if verbose:
                self.debug_print(
                    current_time_bin,
                    individual_id,
                    current_state,
                    is_censored,
                    sojourn_time,
                    status,
                )

        return risk_indices

    def compute_event_indices(self, current_time, verbose=False):
        """
        Compute indices of individuals experiencing an event at the given time,
        excluding individuals censored before or at this time.
        """
        # Condition 1: Event occurs at the current time
        event_mask = (self.time == current_time)

        # Condition 2: Ensure that an event (state transition) happens (not self-transition)
        valid_transition_mask = (self.event_from != self.event_to) & (self.event == 1)

        # Condition 3: Exclude censored individuals at or before current_time
        censorship_mask = self.censorship == 0  # 0 indicates not censored

        # Combine all conditions to create the final mask
        final_mask = event_mask & valid_transition_mask & censorship_mask

        # Debugging prints
        if verbose:
            print(f"Current Time: {current_time}")
            print(f"Event Mask: {event_mask}")
            print(f"Valid Transition Mask: {valid_transition_mask}")
            print(f"Censorship Mask: {censorship_mask}")
            print(f"Final Mask: {final_mask}")

        # Return indices of individuals meeting all conditions
        return np.where(final_mask)[0]


class IndividualStateTracker:
    def __init__(self, event_from, event_to, individual_times, individuals, initial_state=1):
        """
        Initialize the IndividualStateTracker to keep track of each individual's state transitions.
        """
        self.event_from = event_from
        self.event_to = event_to
        self.individual_times = individual_times
        self.individuals = individuals
        self.initial_state = initial_state

    def get_current_state(self, individual, current_time_bin):
        """
        Get the current state of an individual at a given time.

        Parameters:
        - individual: The individual identifier.
        - current_time_bin: The current time to evaluate the state for the individual.

        Returns:
        - The current state of the individual at the given time.
        - The start time of the current state.
        """
        current_time = self.individual_times[current_time_bin]
        individual_indices = np.where(self.individuals == individual)[0]
        times = self.individual_times[individual_indices]
        from_states = self.event_from[individual_indices]
        to_states = self.event_to[individual_indices]

        # Sort events by time
        sorted_indices = np.argsort(times)
        times = times[sorted_indices]
        from_states = from_states[sorted_indices]
        to_states = to_states[sorted_indices]

        # Initialize current state and state start time
        if len(times) == 0 or current_time < times[0]:
            # No events have occurred yet; assume initial state
            current_state = self.initial_state
            state_start_time = 0
            return current_state, state_start_time

        # Start from the initial state
        current_state = from_states[0]
        state_start_time = 0

        for time, to_state in zip(times, to_states):
            if current_time < time:
                break
            current_state = to_state
            state_start_time = time

        return current_state, state_start_time

class CensorshipEvaluator:
    def __init__(self, censorship, event_times, individuals):
        """
        Initialize the CensorshipEvaluator.

        Parameters:
        - censorship: Censorship data for all individuals.
        - event_times: Event times for all individuals.
        - individuals: Array representing individual identifiers for each event.
        """
        self.censorship = censorship
        self.individual_times = event_times
        self.individuals = individuals

    def is_censored_before_time(self, individual, current_time_bin):
        """
        Check if an individual is censored before or at the current time bin.

        Parameters:
        - individual: Identifier of the individual.
        - current_time_bin: The current time bin.

        Returns:
        - Boolean indicating whether the individual is censored before or at the current time bin.
        """
        individual_indices = np.where(self.individuals == individual)[0]
        times = self.individual_times[individual_indices]
        cens = self.censorship[individual_indices]

        valid_indices = np.where(times <= current_time_bin)[0]

        if len(valid_indices) == 0:
            return False

        return np.any(cens[valid_indices] == 1)

if __name__ == '__main__':
    from surv_optimizer.objective_functions.TwoStateCoxObjectiveFunction import TwoStateCoxObjectiveFunction
    from surv_optimizer.data.SyntheticDataset import get_combined_dataset
    from surv_optimizer.data.DatasetManager import DatasetManager

    # Generate dataset
    individuals, time, event_from, event_to, censorship, Z_i1, Z_ij1 = get_combined_dataset()
    dataset = DatasetManager()
    data_view = dataset.get_dataview(state_from=1, sorted=False)
    # Create the TwoStateCoxObjectiveFunction instance
    objective_function = TwoStateCoxObjectiveFunction(dataset, data_view)

    # Instantiate RiskCalculator
    risk_calculator = TwoStateEventRisk(dataset, data_view)
    risk_calculator.get_event_counts_at_sojourn_time(3, verbose=True)
    # for t_idx, t in enumerate(risk_calculator.unique_times):
    #     print(f"\nTime Bin Index: {t_idx}, Time: {t}")
    #     event_indices = risk_calculator.compute_event_indices(t, verbose=True)
    #     print(f"Event Indices at time {t}: {event_indices}")
    #     risk_indices = risk_calculator.compute_risk_set_at_risk(t_idx, verbose=True)
    #     print(f"Risk Set Indices at time {t}: {risk_indices}")

