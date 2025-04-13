import random


class Crossovers:
    def __init__(self) -> None:
        pass

    @staticmethod
    def one_point(parent1, parent2):
        # pointer
        p = random.randint(0, 9)
        child1 = parent1[0:p] + parent2[p:10]
        child2 = parent2[0:p] + parent1[p:10]
        return child1, child2, p

    @staticmethod
    def two_point(parent1, parent2):
        # pointer
        p1 = random.randint(1, 9)
        p2 = random.randint(1, 9)
        while p1 == p2:
            p2 = random.randint(1, 9)
        if p1 > p2:
            temp = p1
            p1 = p2
            p2 = temp

        child1 = parent1[0:p1] + parent2[p1:p2] + parent1[p2:10]
        child2 = parent2[0:p1] + parent1[p1:p2] + parent2[p2:10]
        return child1, child2, p1, p2

    @staticmethod
    def simple_arithmetic(parent1, parent2, alpha, num):
        child1 = []
        child2 = []
        for i in range(10):
            child1.append(parent1[i])
            child2.append(parent2[i])

        child1[num - 1] = round(
            (alpha * parent1[num - 1]) + ((1 - alpha) * parent2[num - 1]), 2
        )
        child2[num - 1] = round(
            (alpha * parent2[num - 1]) + ((1 - alpha) * parent1[num - 1]), 2
        )

        return child1, child2

    @staticmethod
    def simple(parent1, parent2, alpha, num):
        child1 = []
        child2 = []

        for i in range(0, 10 - num):
            child1.append(parent1[i])
            child2.append(parent2[i])

        for i in range(10 - num, 10):
            child1.append(round((alpha * parent1[i]) + ((1 - alpha) * parent2[i]), 2))
            child2.append(round((alpha * parent2[i]) + ((1 - alpha) * parent1[i]), 2))

        return child1, child2

    @staticmethod
    def whole_crossover(parent1, parent2, alpha):
        child = [
            round(alpha * a + (1 - alpha) * b, 2) for a, b in zip(parent1, parent2)
        ]

        child2 = [
            round(alpha * a + (1 - alpha) * b, 2) for a, b in zip(parent2, parent1)
        ]
        return child, child2

    @staticmethod
    def uniform_crossover(parent1, parent2, crossover_prob=0.5):
        offspring1 = parent1[:]
        offspring2 = parent2[:]

        for i in range(len(parent1)):
            if random.random() < crossover_prob:
                offspring1[i], offspring2[i] = offspring2[i], offspring1[i]

        return offspring1, offspring2
