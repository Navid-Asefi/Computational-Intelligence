import os
import sys

from crossovers import Crossovers
from selection import Selection
from tools.generations import ChromosomeGen
from tools.probability import Prob


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Enter a valid integer.")


def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Enter a valid float.")


def choose_method(method_map, label):
    print(f"\nChoose a {label} method:")
    for key, (name, _) in method_map.items():
        print(f"{key}. {name}")
    return input(f"Enter a number for {label}: ").strip()


def main():
    select = Selection()
    crossover = Crossovers()

    selection_methods = {
        "1": ("Random Selection", "random_selection"),
        "2": ("Roulette Wheel", "roulette_wheel"),
        "3": ("Rank Selection", "rank_selection"),
        "4": ("Tournament Selection", "tournament_selection"),
        "5": ("Truncation Selection", "truncation_selection"),
    }

    crossover_methods = {
        "1": ("One Point Crossover", "one_point"),
        "2": ("Two Point Crossover", "two_point"),
        "3": ("Simple Crossover", "simple"),
        "4": ("Simple Arithmetic Crossover", "simple_arithmetic"),
        "5": ("Whole Arithmetic Crossover", "whole_crossover"),
    }

    binary_crossovers = {}

    clear()
    selection_choice = choose_method(selection_methods, "selection")
    clear()
    crossover_choice = choose_method(crossover_methods, "crossover")
    clear()

    crossover_func_name = crossover_methods[crossover_choice][1]
    crossover_func = getattr(crossover, crossover_func_name)
    needs_binary = crossover_func_name in binary_crossovers

    chromosomes_count = get_int("How many chromosomes should your population have? ")
    clear()

    chromo_gen = ChromosomeGen(chromosomes_count, needs_binary)
    population = chromo_gen.gene_generator()
    selected = []

    # Selection Phase
    if selection_choice == "1":
        wanted = get_int("How many chromosomes does the next generation need? ")
        selected = select.random_selection(population, wanted)

    elif selection_choice == "2":
        prob = Prob(population)
        prob.fitness()
        selected = [select.roulette_wheel(population, prob.cumul())]

    elif selection_choice == "3":
        fitnesses = Prob(population).fitness()
        selected = [select.rank_selection(population, fitnesses)]

    elif selection_choice == "4":
        group_size = get_int("Group size for tournament: ")
        clear()
        run_count = get_int("How many tournaments to run: ")
        selected = select.tournament_selection(population, group_size, run_count)

    elif selection_choice == "5":
        ratio = get_float("Top percentage (e.g., 0.4): ")
        clear()
        group_size = get_int("How many chromosomes to select: ")
        selected = select.truncation_selection(population, ratio, group_size)

    else:
        print("Invalid selection method.")
        return

    clear()
    print("Selected Chromosomes:")
    for i, chrom in enumerate(selected):
        print(f"{i + 1}: {chrom}")

    # Crossover Phase
    crossover_func = getattr(crossover, crossover_methods[crossover_choice][1])

    offspring = []
    if crossover_func_name == "simple":
        alpha = get_float("Type the desired alpha (between 0 and 1): ")
        num = get_int("How many genes to blend (0 to 10): ")
        if not (0 <= alpha <= 1 and 0 <= num <= 10):
            print("Invalid alpha or gene count.")
            sys.exit()

    elif crossover_func_name == "simple_arithmetic":
        alpha = get_float("Type the desired alpha (between 0 and 1): ")
        num = get_int("Type the gene number you want to change:  ")
        if not (0 <= alpha <= 1 and 0 <= num <= 10):
            print("Invalid alpha or gene count.")
            sys.exit()

    elif crossover_func_name == "whole_crossover":
        alpha = get_float("Type the desired alpha (between 0 and 1): ")
        if not (0 <= alpha <= 1):
            print("Invalid alpha or gene count.")
            sys.exit()

    for i in range(0, len(selected) - 1, 2):
        parent1 = selected[i]
        parent2 = selected[i + 1]

        if crossover_func_name in ("simple", "simple_arithmetic"):
            child1, child2 = crossover_func(parent1, parent2, alpha, num)
            offspring.extend([child1, child2])

        elif crossover_func_name == "whole_crossover":
            child1, child2 = crossover_func(parent1, parent2, alpha)
            offspring.extend([child1, child2])

        else:
            child = crossover_func(parent1, parent2)
            offspring.append(child)
    print("\nOffspring after crossover:")
    for i, child in enumerate(offspring):
        print(f"{i + 1}: {child}")


if __name__ == "__main__":
    main()
