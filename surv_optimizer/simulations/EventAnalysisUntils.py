import matplotlib.pyplot as plt

class EventAnalysisUtils:
    def __init__(self, simulated_data, true_data):
        """
        Initialize the EventAnalysisUtils.

        Parameters:
        - simulated_data: List of simulated data per individual.
        - true_data: List of true event data per individual.
        """
        self.simulated_data = simulated_data
        self.true_data = true_data

    def compute_event_counts(self):
        """
        Compute the number of events per individual for simulated and true data.

        Returns:
        - Tuple of (simulated_counts, true_counts)
        """
        simulated_counts = [len(individual['events']) for individual in self.simulated_data]
        true_counts = [len(individual['events']) for individual in self.true_data]
        return simulated_counts, true_counts

    def plot_event_counts(self):
        """
        Plot a histogram comparing the number of events per individual.
        """
        simulated_counts, true_counts = self.compute_event_counts()
        individuals = range(len(simulated_counts))

        plt.figure(figsize=(12, 6))
        plt.bar(individuals, simulated_counts, alpha=0.6, label='Simulated')
        plt.bar(individuals, true_counts, alpha=0.6, label='True')
        plt.xlabel('Individual')
        plt.ylabel('Number of Events')
        plt.title('Comparison of Number of Events per Individual')
        plt.legend()
        plt.show()

    def plot_event_times(self, individual_id):
        """
        Plot the event times for a specific individual.

        Parameters:
        - individual_id: ID of the individual to plot.
        """
        # Find the data for the specified individual
        simulated = next((ind for ind in self.simulated_data if ind['individual_id'] == individual_id), None)
        true = next((ind for ind in self.true_data if ind['individual_id'] == individual_id), None)

        if simulated is None or true is None:
            print(f"Individual {individual_id} not found in data.")
            return

        simulated_times = [event['time'] for event in simulated['events']]
        true_times = [event['time'] for event in true['events']]

        plt.figure(figsize=(12, 6))
        plt.step(simulated_times, range(1, len(simulated_times) + 1), where='post', label='Simulated')
        plt.step(true_times, range(1, len(true_times) + 1), where='post', label='True')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Number of Events')
        plt.title(f'Event Times for Individual {individual_id}')
        plt.legend()
        plt.show()

    def compute_time_differences(self):
        """
        Compute the time differences between simulated and true events.

        Returns:
        - List of time differences per individual.
        """
        time_differences = []
        for sim_ind, true_ind in zip(self.simulated_data, self.true_data):
            sim_times = [event['time'] for event in sim_ind['events']]
            true_times = [event['time'] for event in true_ind['events']]
            min_length = min(len(sim_times), len(true_times))
            differences = [sim_times[i] - true_times[i] for i in range(min_length)]
            time_differences.append(differences)
        return time_differences

    def plot_time_differences(self):
        """
        Plot the distribution of time differences between simulated and true events.
        """
        time_differences = self.compute_time_differences()
        all_differences = [diff for ind_diffs in time_differences for diff in ind_diffs]

        plt.figure(figsize=(12, 6))
        plt.hist(all_differences, bins=30, alpha=0.7)
        plt.xlabel('Time Difference (Simulated - True)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Time Differences Between Simulated and True Events')
        plt.show()
