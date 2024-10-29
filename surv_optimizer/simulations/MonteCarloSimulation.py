import numpy as np
import matplotlib.pyplot as plt

from surv_optimizer.Optimizer import Optimizer
from surv_optimizer.abstract_classes.AbstractHazard import AbstractHazard
from surv_optimizer.calculators.TwoStateHazard import TwoStateHazard
from surv_optimizer.objective_functions.TwoStateCoxObjectiveFunction import TwoStateCoxObjectiveFunction
from surv_optimizer.data.SyntheticDataset import make_transition_data
from surv_optimizer.simulations.EventAnalysisUntils import EventAnalysisUtils


class RecurrentEventMonteCarloSimulator:
    def __init__(
        self,
        hazard_survival_calculator1,
        hazard_survival_calculator2,
        dataset_manager,
        data_view1,
        data_view2,
        risk_calculator,
    ):
        """
        Initialize the RecurrentEventMonteCarloSimulator.

        Parameters:
        - hazard_survival_calculator1: Hazard calculator for transitions from state 1 to 2.
        - hazard_survival_calculator2: Hazard calculator for transitions from state 2 to 1.
        - dataset_manager: DatasetManager instance.
        - data_view1: Data view for state_from=1.
        - data_view2: Data view for state_from=2.
        - risk_calculator: Risk calculator instance.
        """
        self.hazard_calculator1 = hazard_survival_calculator1
        self.hazard_calculator2 = hazard_survival_calculator2
        self.dataset_manager = dataset_manager
        self.data_view1 = data_view1
        self.data_view2 = data_view2
        self.risk_calculator = risk_calculator

        # Prepare the individuals
        self.individuals = self.dataset_manager.get_unique_individuals()
        self.n_individuals = len(self.individuals)

        # Ensure hazard calculators have computed baseline hazards
        if self.hazard_calculator1.baseline_hazard_ is None:
            raise ValueError("Baseline hazard for calculator1 is not computed.")
        if self.hazard_calculator2.baseline_hazard_ is None:
            raise ValueError("Baseline hazard for calculator2 is not computed.")

        # Compute cumulative hazard functions for each individual
        self.cumulative_hazard_funcs1 = self.hazard_calculator1.get_cumulative_hazard_function()
        self.cumulative_hazard_funcs2 = self.hazard_calculator2.get_cumulative_hazard_function()

        # Compute conductance for each individual
        self.conductance1 = self.compute_conductance(self.cumulative_hazard_funcs1)
        self.conductance2 = self.compute_conductance(self.cumulative_hazard_funcs2)

    def compute_conductance(self, cumulative_hazard_funcs):
        """
        Compute the conductance for each individual.

        Parameters:
        - cumulative_hazard_funcs: List of cumulative hazard functions.

        Returns:
        - Array of conductance values per individual.
        """
        conductance = []
        for chf in cumulative_hazard_funcs:
            # Effective resistance is the final value of cumulative hazard
            R = chf.y[-1] if chf.y[-1] > 0 else 1e-6  # Avoid division by zero
            C = 1.0 / R
            conductance.append(C)
        return np.array(conductance)

    def simulate_individual_events(self, individual_idx, max_time, max_events=None, n_simulations=100):
        """
        Simulate the sequence of event times for a single individual.

        Parameters:
        - individual_idx: Index of the individual in the individuals array.
        - max_time: Maximum simulation time.
        - max_events: Maximum number of events to simulate (optional).
        - n_simulations: Number of simulations per event.

        Returns:
        - List of events with their times and state transitions.
        """
        events = []
        current_state = 1
        t = 0.0
        n_events = 0

        # Get the cumulative hazard functions for this individual
        chf1 = self.cumulative_hazard_funcs1[individual_idx]
        chf2 = self.cumulative_hazard_funcs2[individual_idx]

        # Get conductance values for this individual
        C1 = self.conductance1[individual_idx]
        C2 = self.conductance2[individual_idx]

        while t < max_time and (max_events is None or n_events < max_events):
            if current_state == 1:
                # Use chf1 to simulate time to next event
                chf = chf1
                conductance = C1
                next_state = 2
            else:
                # current_state == 2
                # Use chf2 to simulate time to next event
                chf = chf2
                conductance = C2
                next_state = 1

            # Simulate time to next event
            delta_t = self.simulate_time_to_next_event(chf, t, max_time, conductance, n_simulations)
            if delta_t is None:
                # No event occurs before max_time
                break

            t += delta_t
            if t >= max_time:
                break

            # Record the event
            events.append({'time': t, 'from_state': current_state, 'to_state': next_state})
            n_events += 1

            # Update current state
            current_state = next_state

        return events

    def simulate_time_to_next_event(self, chf, t0, max_time, conductance, n_simulations=100):
        """
        Simulate the time to the next event given the cumulative hazard function.

        Parameters:
        - chf: Cumulative hazard function (StepFunction).
        - t0: Current time.
        - max_time: Maximum simulation time.
        - conductance: Conductance value for adjusting the hazard.
        - n_simulations: Number of simulations to perform.

        Returns:
        - Mean time until the next event, or None if no event occurs before max_time.
        """
        simulated_times = []
        for _ in range(n_simulations):
            U = np.random.uniform(0, 1)
            H_t0 = chf(t0)
            H_total = -np.log(U) / conductance
            H_target = H_t0 + H_total

            # Get the times and cumulative hazard values
            times = chf.x
            hazards = chf.y

            # Find the index where t0 would be inserted to keep times sorted
            idx = np.searchsorted(times, t0, side='right') - 1
            if idx < 0:
                idx = 0

            while idx < len(times) - 1:
                t_i = max(t0, times[idx])
                t_i1 = times[idx + 1]
                H_i = chf(t_i)
                H_i1 = chf(t_i1)

                h_i = (H_i1 - H_i) / (t_i1 - t_i) if t_i1 > t_i else 0

                if h_i == 0:
                    # No hazard in this interval, move to next
                    idx += 1
                    continue

                if H_i1 >= H_target:
                    # Event occurs in this interval
                    t_event = t_i + (H_target - H_i) / h_i
                    if t_event < t0:
                        t_event = t0  # Ensure t_event >= t0
                    if t_event > max_time:
                        break  # Event occurs after max_time
                    delta_t = t_event - t0
                    simulated_times.append(delta_t)
                    break
                else:
                    # Move to next interval
                    idx += 1
            else:
                # No event occurs before max_time
                continue

        if len(simulated_times) == 0:
            return None  # No events simulated before max_time

        # Return the mean of simulated times
        mean_delta_t = np.mean(simulated_times)
        return mean_delta_t

    def simulate(self, max_time, max_events=None, n_simulations=100):
        """
        Simulate events for all individuals.

        Parameters:
        - max_time: Maximum simulation time.
        - max_events: Maximum number of events per individual (optional).
        - n_simulations: Number of simulations per event.

        Returns:
        - List of simulated data per individual.
        """
        simulated_data = []
        for individual_idx in range(self.n_individuals):
            events = self.simulate_individual_events(individual_idx, max_time, max_events, n_simulations)
            simulated_data.append({
                'individual_id': self.individuals[individual_idx],
                'events': events
            })
        return simulated_data


if __name__ == "__main__":
    from surv_optimizer.objective_functions.TwoStateCoxObjectiveFunction import TwoStateCoxObjectiveFunction
    from surv_optimizer.calculators.TwoStateCovariateContribution import TwoStateCovariateContribution
    from surv_optimizer.calculators.TwoStateEventRisk import TwoStateEventRisk
    from surv_optimizer.data.DatasetManager import DatasetManager
    from surv_optimizer.Optimizer import Optimizer

    np.set_printoptions(suppress=True)
    # Initialize DatasetManager
    dataset_manager = DatasetManager()

    # Get data view for state_from=1 and sorted=True
    data_view1 = dataset_manager.get_dataview(state_from=1, sorted=True)
    data_view2 = dataset_manager.get_dataview(state_from=2, sorted=True)
    risk_calculator1 = TwoStateEventRisk(dataset_manager, data_view1)
    risk_calculator2 = TwoStateEventRisk(dataset_manager, data_view2)
    covariate_calculator1 = TwoStateCovariateContribution(dataset_manager, data_view1, risk_calculator1)
    covariate_calculator2 = TwoStateCovariateContribution(dataset_manager, data_view2, risk_calculator2)
    survival_calculator1 = TwoStateHazard(dataset_manager, data_view1)
    survival_calculator1.set_calculators(covariate_calculator1, risk_calculator1)
    coef1 = [0.18632779, 0.72250941, 0.41984884, 0.12001864]
    survival_calculator1.compute_baseline_hazard(coef1)
    coef2 = [-0.02164444, -0.33949401, -0.16757074,  0.35347641]
    survival_calculator2 = TwoStateHazard(dataset_manager, data_view2)
    survival_calculator2.set_calculators(covariate_calculator2, risk_calculator2)
    survival_calculator2.compute_baseline_hazard(coef2)


    # After optimization, compute baseline hazard for all time bins
    # Initialize the simulator
    simulator = RecurrentEventMonteCarloSimulator(
        hazard_survival_calculator1=survival_calculator1,
        hazard_survival_calculator2=survival_calculator2,
        dataset_manager=dataset_manager,
        data_view1=data_view1,
        data_view2=data_view2,
        risk_calculator=risk_calculator1
    )

    # Simulate events up to a maximum time of 100 units
    simulated_data = simulator.simulate(max_time=100)

    # Access the simulated events for each individual
    for individual_data in simulated_data:
        individual_id = individual_data['individual_id']
        events = individual_data['events']
        print(f"Individual {individual_id} events:")
        for event in events:
            print(f"  Time: {event['time']}, From State: {event['from_state']}, To State: {event['to_state']}")
    # Initialize the analysis utils
    analysis_utils = EventAnalysisUtils(simulated_data, true_event_data)

    # Plot the number of events per individual
    analysis_utils.plot_event_counts()

    # Plot event times for a specific individual
    individual_id = 9  # Replace with desired individual ID
    analysis_utils.plot_event_times(individual_id)

    # Plot the distribution of time differences
    analysis_utils.plot_time_differences()
