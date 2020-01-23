class TrainingStats:

    def __init__(self):

        self.iterations = []
        self.losses = []
        self.times = []

    def add_training_iteration(self, iteration, loss, time):

        self.iterations.append(iteration)
        self.losses.append(loss)
        self.times.append(time)

    def trained_until(self):
        """The number of iterations trained for (the maximum iteration plus
        one)."""

        if not self.iterations:
            return 0
        return max(self.iterations) + 1
