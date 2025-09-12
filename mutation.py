import random


class Mutations:
    def __init__(self) -> None:
        pass

    @staticmethod
    def bit_flip(parent):
        """Bit-flip mutation for binary chromosomes"""
        num_mutations = random.randint(1, len(parent))
        child = parent.copy()
        indices_to_mutate = random.sample(range(len(parent)), num_mutations)
        for idx in indices_to_mutate:
            child[idx] = 1 - child[idx]
        return child

    @staticmethod
    def swap_mutation(parent):
        """Swap mutation for permutation chromosomes"""
        child = parent.copy()
        idx1, idx2 = 0, 0
        while idx1 == idx2:
            idx1 = random.randint(0, len(parent) - 1)
            idx2 = random.randint(0, len(parent) - 1)
        child[idx1], child[idx2] = child[idx2], child[idx1]
        return child

    @staticmethod
    def scramble_mutation(parent):
        """Scramble mutation for permutation chromosomes"""
        child = parent.copy()
        point1, point2 = random.sample(range(len(parent)), 2)
        if point1 > point2:
            point1, point2 = point2, point1
        segment = child[point1 : point2 + 1]
        random.shuffle(segment)
        child[point1 : point2 + 1] = segment
        return child

    @staticmethod
    def inversion_mutation(parent):
        """Inversion mutation for permutation chromosomes"""
        child = parent.copy()
        point1, point2 = random.sample(range(len(parent)), 2)
        if point1 > point2:
            point1, point2 = point2, point1
        segment = child[point1 : point2 + 1]
        child[point1 : point2 + 1] = segment[::-1]
        return child

    @staticmethod
    def insert_mutation(parent):
        """Insert mutation for permutation chromosomes"""
        child = parent.copy()
        from_idx, to_idx = random.sample(range(len(parent)), 2)
        gene = child.pop(from_idx)
        child.insert(to_idx, gene)
        return child
