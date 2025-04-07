import random


class ChromosomeGen:
    def __init__(self, generations, binary=False):
        self.community = []
        self.generations = generations
        self.binary = binary

    def gene_generator(self):
        """Generates the required Chromosomes and their genes"""
        for _ in range(self.generations):
            if self.binary:
                self.community.append([random.randint(0, 1) for _ in range(10)])
            else:
                self.community.append([random.randint(0, 9) for _ in range(10)])

        return self.community
