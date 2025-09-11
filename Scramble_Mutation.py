import random
from tools.generations import ChromosomeGen

chromosomes = ChromosomeGen(1,permute=True)
parents = chromosomes.gene_generator()
parent=parents[0]

def scramble_mutation(parent):
   
    offspring = parent.copy()
    length = 10
    
    point1, point2 = random.sample(range(10), 2)
    
    if point1 > point2:
        point1, point2 = point2, point1
    
    segment = offspring[point1:point2 + 1]
    
    random.shuffle(segment)
    
    offspring[point1:point2 + 1] = segment
    
    return offspring

print("Parent chromosome:", parent)
offspring = scramble_mutation(parent)
print("Offspring chromosome:", offspring)