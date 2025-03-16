import random
from generations import chromosomeGen
from probability import Prob

def truncation_selection(population, selection_ratio,size_of_group):   
     
    fitness_values = list(zip(population, pr.fitness()))

    fitness_values.sort(key=lambda x: x[1], reverse=True)
    
    num_selected = int(len(population) * selection_ratio)
    
    selected_individuals = [community for community, _ in fitness_values[:num_selected]]
    
    new_gen = []
    while len(new_gen) < size_of_group :
        adam = random.choice(selected_individuals)
        new_gen.append(adam)
    return new_gen

user_input = int(input("How many chromosomes do you want to be generated: "))
selection_ratio=float(input("enter top percentage population that you want: "))
size_of_group = int(input("How many chromosomes should the group have: "))
gen = chromosomeGen(user_input)
pr = Prob(gen.gene_generator())

print(gen)
print(truncation_selection(gen, selection_ratio ,size_of_group))