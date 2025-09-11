import random
from tools.generations import ChromosomeGen

chromosomes = ChromosomeGen(1,permute=True)
parents = chromosomes.gene_generator()
parent=parents[0]

def insert_mutation(parent):

    offspring = parent.copy()
    length = 10
    
    # Select two distinct positions
    from_idx, to_idx = random.sample(range(10), 2)
    
    # Remove the gene from its original position
    gene = offspring.pop(from_idx)
    
    # Insert it at the target position
    offspring.insert(to_idx, gene)
    
    print(f"Moved gene from index {from_idx} to index {to_idx}")
    return offspring

offspring = insert_mutation(parent)
print(f"{parent}\n,{offspring}")
