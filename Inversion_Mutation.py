import random
from tools.generations import ChromosomeGen

chromosomes = ChromosomeGen(1,permute=True)
parents = chromosomes.gene_generator()
parent=parents[0]

def inversion_mutation(parent):
    offspring = parent.copy()
    length = 10
    
    point1, point2 = random.sample(range(10), 2)
    if point1 > point2:
        point1, point2 = point2, point1
        
    segment = offspring[point1:point2 + 1]
    reversed_segment = segment[::-1]  
    offspring[point1:point2 + 1] = reversed_segment
    print(f"Inverted indices {point1} to {point2}")
    print(f"Original segment: {segment}")
    print(f"Reversed segment: {reversed_segment}")
    
    return offspring

print("Parent chromosome:", parent)

offspring = inversion_mutation(parent)
print("Offspring chromosome:", offspring)
