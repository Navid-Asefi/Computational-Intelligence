import random
from tools.generations import ChromosomeGen

chromosomes = ChromosomeGen(1, binary=True)
parents = chromosomes.gene_generator()
parent=parents[0]


def bit_flipping_mutation(parent):
    num_mutations=random.randint(1,10)
    child = parent.copy()
    indices_to_mutate = random.sample(range(10), num_mutations)
    for idx in indices_to_mutate:
        child[idx] = 1 - child[idx]
    return child

child,num=bit_flipping_mutation(parent)
print(f"{parent}\n,{child}")
