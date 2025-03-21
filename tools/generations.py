import random

class ChromosomeGen:

    def __init__(self, generations):
        self.community = []
        self.generations = generations

    def gene_generator(self):
        """Generates the required Chromosomes and their genes"""
        for _ in range(self.generations):
            self.community.append([random.randint(0, 9) for _ in range(10)])
            
        return self.community

