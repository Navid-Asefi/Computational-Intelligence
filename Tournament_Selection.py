import random
from generations import chromosomeGen
from probability import Prob

def tournament(population, size, count):
    best = []
    for _ in range(count):
        selection = random.sample(population, size)
        prob = Prob(selection)
        selection_with_fit = list(zip(selection, prob.fitness()))
        best.append(max(selection_with_fit, key=lambda x: x[1]))
    return selection_with_fit, best

user_input = int(input("How many chromosomes do you want to be generated: "))
size_of_group = int(input("How many chromosomes should the group have: "))
run_count = int(input("How many times should the tournament take place: "))
gen = chromosomeGen(user_input)
all_chromo = gen.gene_generator()
all, best = tournament(all_chromo, size_of_group, run_count)

print("All Chromosomes: ")
for xs in all_chromo:
     print(" ".join(map(str, xs)))
print(f"\nSample Chromosomes: ")
for xs in all:
    print(" ".join(map(str, xs)))
print(f"\nBest Chromosome:\n {best} ")

    


