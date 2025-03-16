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
    def __iter__(self):
        """Makes the class iterable"""
        return iter(self.community)
    
    def __len__(self):
        return len(self.community)  
    
    def __str__(self):
        return f"ChromosomeGen with {self.generations} generations:\n{self.community}"
