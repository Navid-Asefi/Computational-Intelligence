import os
from selection import Selection
from generations import ChromosomeGen
from probability import Prob


select = Selection()

#Prints the methods available
os.system('cls' if os.name == 'nt' else 'clear')
print('\n'.join(["1. Random Selection", "2. Roulette Wheel", "3. Rank Selection", "4. Tournament Selection", "5. Truncation Selection", " "]))
#Takes user inputs
selection_method = input("What selectoin method do you want to use? (Enter a number): ")
os.system('cls' if os.name == 'nt' else 'clear')
chromosomes_count = int(input("How many chromosomes should your population have? "))
os.system('cls' if os.name == 'nt' else 'clear')

if selection_method == "1":
    wanted_chromosomes = int(input("How many chromosomes does the next generation need? "))
    os.system('cls' if os.name == 'nt' else 'clear')
    pop = ChromosomeGen(chromosomes_count).gene_generator()
    print(f"All of the created population:\n{pop}")
    print(f"\nThe selected chromosomes:\n{select.random_selection(pop, wanted_chromosomes)}")

elif selection_method == "2":
    chromo_class = ChromosomeGen(chromosomes_count)
    chromos = chromo_class.gene_generator()
    prob_class = Prob(chromos)
    prob_class.fitness()
    poss = prob_class.prob()

    print(f"All of the generated chromosomes: \n{chromos}")
    print(f"\nPossibilities of them: \n{poss}")
    print(f"\nThe selected chromosome\n{select.roulette_wheel(chromos, prob_class.cumul())}")

elif selection_method == "3":
    pop = ChromosomeGen(chromosomes_count).gene_generator()
    print(f"Unsorted community:\n{pop}\nTheir fitnesses:\n{Prob(pop).fitness()}\n")
    print(f"\nChosen Chromosome:\n{select.rank_selection(pop, Prob(pop).fitness())}\n")

elif selection_method == "4":
    size_of_group = int(input("How many chromosomes should the group have: "))
    os.system('cls' if os.name == 'nt' else 'clear')
    run_count = int(input("How many times should the tournament take place: "))
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"The selected chromosomes:\n{select.tournament_selection(ChromosomeGen(chromosomes_count).gene_generator(), size_of_group, run_count)}")

elif selection_method == "5": 
    selection_ratio=float(input("Enter top percentage population that you want: "))
    os.system('cls' if os.name == 'nt' else 'clear')
    size_of_group = int(input("How many chromosomes should the group have: "))
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"The selected chromosomes:\n{select.truncation_selection(ChromosomeGen(chromosomes_count).gene_generator(), selection_ratio, size_of_group)}")


