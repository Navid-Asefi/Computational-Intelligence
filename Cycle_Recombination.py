from tools.generations import ChromosomeGen
import random

chromosomes = ChromosomeGen(2, permute=True)
parents = chromosomes.gene_generator()
parent1=parents[0]
parent2=parents[1]



def cycle_crossover(parent1,parent2):
    # Initialize child with placeholders
    child = [None] * 10
    
    # Start with the first index not copied
    index = 0
    # Flag to alternate which parent we copy from for each cycle
    copy_from_p1 = True

    while None in child:
        # If we've reached a copied index, find the next available one
        if child[index] is not None:
            index = child.index(None)
            # Toggle the parent for the new cycle
            copy_from_p1 = not copy_from_p1
            continue

        # Start a new cycle from the current index
        cycle_start = index
        current_index = index
        
        # Trace the entire cycle first (optional, but can be done)
        # ... alternatively, we can copy as we trace ...
        
        # Simpler: copy the value for the current index in the cycle
        if copy_from_p1:
            child[current_index] = parent1[current_index]
        else:
            child[current_index] = parent2[current_index]
            
        # Find the next index in the cycle
        # Find the value in p1 at current_index, then find its position in p2
        value_in_p1 = parent1[current_index]
        current_index = parent2.index(value_in_p1)
        
        # If we've returned to the start of the cycle, break and find next cycle
        if current_index == cycle_start:
            index = cycle_start + 1 # Move to the next index to check
        else:
            # Otherwise, continue the cycle
            index = current_index

    return child

child1 = cycle_crossover(parent1, parent2)
child2= cycle_crossover(parent2,parent1)
# To get child2, simply swap the parents: child2 = cycle_crossover(p2, p1)
print(f"{parent1}\n,{parent2}\n,{child1}\n,{child2}")