import random
from probability import Prob
from generations import chromosomeGen

def roulette_wheel_selection(population, cumulative_probability):
    rand = random.uniform(0, 1)
    for i, cp in enumerate(cumulative_probability):
        if rand <= cp:
            return population[i]

chromo_count = int(input("Enter the number of wanted chromosomes: "))
chromo_class = chromosomeGen(chromo_count)
chromos = chromo_class.gene_generator()

prob_class = Prob(chromos)
prob_class.fitness()
poss = prob_class.prob()

print(f"All of the generated chromosomes: {chromos}")
print(f"Possibilities of them: {poss}")
print(f"The selected chromosome{roulette_wheel_selection(chromos, prob_class.cumul())}")



