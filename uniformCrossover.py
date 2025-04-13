from tools.generations import ChromosomeGen
import random

def uniform_crossover(parent1, parent2, crossover_prob=0.5):

    offspring1 = parent1[:]
    offspring2 = parent2[:]
    
    for i in range(len(parent1)):
        if random.random() < crossover_prob:
            offspring1[i], offspring2[i] = offspring2[i], offspring1[i]
    
    return offspring1, offspring2


chromosomes=ChromosomeGen(2, binary=True)
parents=chromosomes.gene_generator()
parent1=parents[0]
parent2=parents[1]

child1, child2 = uniform_crossover(parent1, parent2)
print("Child 1:", child1)
print("Child 2:", child2)
