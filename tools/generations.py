import random


class ChromosomeGen:
    def __init__(self, generations, binary=False, permute=False):
        self.community = []
        self.generations = generations
        self.binary = binary
        self.permute = permute

    def gene_generator(self):
        """Generates the required Chromosomes and their genes"""
        for _ in range(self.generations):
            if self.binary:
                self.community.append([random.randint(0, 1) for _ in range(10)])

            elif self.permute:
                a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                self.community.append(random.sample(a, 10))

            else:
                self.community.append([random.randint(0, 9) for _ in range(10)])

        return self.community
