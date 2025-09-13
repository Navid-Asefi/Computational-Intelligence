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
            idx1 = random.randint(0, len(parent[0]) - 1)
            idx2 = random.randint(0, len(parent[0]) - 1)
        child[0][idx1], child[0][idx2] = child[0][idx2], child[0][idx1]
        child[1][idx1], child[1][idx2] = child[1][idx2], child[1][idx1]
        return child

    @staticmethod
    def scramble_mutation(parent):
        """Scramble mutation for permutation chromosomes"""
        child = [p.copy() for p in parent]  # copy each chromosome individually

        # Randomly choose two points
        point1, point2 = random.sample(range(len(parent[0])), 2)
        if point1 > point2:
            point1, point2 = point2, point1

        # Extract the segments
        segment1 = child[0][point1 : point2 + 1]
        segment2 = child[1][point1 : point2 + 1]

        # Print points and original segments
        print(f"\nScramble points: {point1}-{point2}")
        print(f"Original segments: {segment1}, {segment2}")

        # Shuffle the segments
        random.shuffle(segment1)
        random.shuffle(segment2)

        # Assign the shuffled segments back
        child[0][point1 : point2 + 1] = segment1
        child[1][point1 : point2 + 1] = segment2

        # Print the new segments
        print(f"Shuffled segments: {segment1}, {segment2}")

        return child

    @staticmethod
    def inversion_mutation(parent):
        """Inversion mutation for permutation chromosomes"""
        child = parent.copy()
        point1, point2 = random.sample(range(len(parent[0])), 2)
        if point1 > point2:
            point1, point2 = point2, point1
        segment1 = child[0][point1 : point2 + 1]
        segment2 = child[1][point1 : point2 + 1]

        print(f"\nInversion points: {point1}-{point2}")
        print(f"Original segments: {segment1}, {segment2}")

        child[0][point1 : point2 + 1] = segment1[::-1]
        child[1][point1 : point2 + 1] = segment2[::-1]

        print(f"Inverted segments: {segment1}, {segment2}")

        return child

    @staticmethod
    def insert_mutation(parent):
        """Insert mutation for permutation chromosomes"""
        child = parent.copy()
        from_idx, to_idx = random.sample(range(len(parent[0])), 2)
        gene = child[0].pop(from_idx)
        gene1 = child[1].pop(from_idx)
        child[0].insert(to_idx, gene)
        child[1].insert(to_idx, gene1)
        return child

    @staticmethod
    def complement_mutation(parent):
        child = parent.copy()
        length = len(child[0])

        # pick a safe number of genes to mutate
        num_genes_to_mutate = random.randint(1, length)

        # now safe: sample within bounds
        genes_to_mutate = random.sample(range(length), num_genes_to_mutate)

        for idx in genes_to_mutate:
            child[0][idx] = 9 - child[0][idx]  # complement relative to 9
            child[1][idx] = 9 - child[1][idx]  # complement relative to 9

        return child
