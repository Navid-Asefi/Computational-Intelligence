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

    @staticmethod
    def Order_Recombination(parent1, parent2):
        size = len(parent1)

        a, b = sorted(random.sample(range(size), 2))

        child1 = ["x"] * size
        child2 = ["x"] * size

        child1[a:b] = parent1[a:b]
        child2[a:b] = parent2[a:b]

        def fill(child, donor, start, end):
            idx = end
            for gene in donor[end:] + donor[:end]:
                if gene not in child:
                    if idx >= size:
                        idx = 0
                    child[idx] = gene
                    idx += 1

        fill(child1, parent2, a, b)
        fill(child2, parent1, a, b)

        return child1, child2, a, b

    @staticmethod
    def Cycle_Recombination(parent1, parent2):
        def cx(p1, p2):
            child = [None] * len(p1)
            index = 0
            copy_from_p1 = True

            while None in child:
                if child[index] is not None:
                    index = child.index(None)
                    copy_from_p1 = not copy_from_p1
                    continue

                cycle_start = index
                current_index = index

                while True:
                    child[current_index] = (
                        p1[current_index] if copy_from_p1 else p2[current_index]
                    )
                    value_in_p1 = p1[current_index]
                    current_index = p2.index(value_in_p1)
                    if current_index == cycle_start:
                        break

                index = current_index + 1 if None in child else index

            return child

        child1 = cx(parent1, parent2)
        child2 = cx(parent2, parent1)
        return child1, child2
