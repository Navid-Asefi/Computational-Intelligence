import random
from generations import chromosomeGen

def random_selection(all_chromo,user_input):
    new_gen = []

    while len(new_gen) < (user_input ):
        adam = random.choice(all_chromo)
        if adam not in new_gen :
            new_gen.append(adam)
    return new_gen

user_input = int(input("How many chromosomes do you want to be generated: "))
size_of_group = int(input("How many chromosomes should the group have: "))
gen = chromosomeGen(size_of_group)
all_chromo = gen.gene_generator()

print(f"All of the generated chromosomes: {all_chromo}")
print(f"The selected chromosome: {random_selection(all_chromo,user_input)}")
