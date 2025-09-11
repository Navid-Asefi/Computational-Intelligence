import bisect
import random

from tools.probability import Prob


class Selection:
    def __init__(self) -> None:
        pass

    @staticmethod
    def roulette_wheel(population, cumulative_probability, k=None):
        if not population:
            return []

        if k is None:
            k = len(population)

        # defensive checks
        if len(cumulative_probability) != len(population):
            raise ValueError("cumulative_probability must match population length")

        # ensure last cumulative is exactly 1.0
        cp = list(cumulative_probability)
        cp[-1] = 1.0

        selected = []
        for _ in range(k):
            r = random.random()  # in [0.0, 1.0)
            idx = bisect.bisect_left(cp, r)
            if idx >= len(population):  # safety clamp
                idx = len(population) - 1
            selected.append(population[idx])
        return selected

    @staticmethod
    def random_selection(population, size_of_group):
        new_gen = []
        while len(new_gen) < size_of_group:
            adam = random.choice(population)
            new_gen.append(adam)
        return new_gen

    @staticmethod
    def tournament_selection(population, size, count):
        best = []
        for _ in range(count):
            selection = random.sample(population, size)
            prob = Prob(selection)
            selection_with_fit = list(zip(selection, prob.fitness()))
            best.append(max(selection_with_fit, key=lambda x: x[1]))
        return best

    @staticmethod
    def rank_selection(community, fitness, k=None):
        if not community:
            return []
        if k is None:
            k = len(community)

        # Pair fitness with chromosomes and sort (descending: best first)
        combined = sorted(zip(fitness, community), key=lambda x: x[0], reverse=True)
        _, community_sorted = zip(*combined)
        n = len(community_sorted)

        # Assign ranks: best gets rank n, worst gets rank 1
        ranks = list(range(n, 0, -1))

        # Build cumulative distribution
        cumulative = []
        running = 0
        for r in ranks:
            running += r
            cumulative.append(running)

        total = cumulative[-1]
        # Normalize cumulative so the last element is 1.0
        cp = [c / total for c in cumulative]
        cp[-1] = 1.0

        # Sample k chromosomes
        selected = []
        for _ in range(k):
            r = random.random()
            idx = bisect.bisect_left(cp, r)
            selected.append(community_sorted[idx])

        return selected

    @staticmethod
    def truncation_selection(population, selection_ratio, size_of_group):
        fitness_values = list(zip(population, Prob(population).fitness()))

        fitness_values.sort(key=lambda x: x[1], reverse=True)

        num_selected = int(len(population) * selection_ratio)

        selected_individuals = [
            community for community, _ in fitness_values[:num_selected]
        ]

        new_gen = []
        while len(new_gen) < size_of_group:
            adam = random.choice(selected_individuals)
            new_gen.append(adam)
        return new_gen
