import random
from tools.generations import ChromosomeGen

chromosomes = ChromosomeGen(1,permute=True)
parents = chromosomes.gene_generator()
parent=parents[0]

def swap_mutation(parent):

    offspring = parent.copy()
    length = 10
        
    # Randomly select two distinct indices
    idx1, idx2 = 0, 0
    while idx1 == idx2:
        idx1 = random.randint(0, 9)
        idx2 = random.randint(0, 9)
             
    # Swap the values at the selected indices
    offspring[idx1], offspring[idx2] = offspring[idx2], offspring[idx1]
    
    return offspring



offspring = swap_mutation(parent)
print("Parent:   ", parent)
print("Offspring:", offspring)