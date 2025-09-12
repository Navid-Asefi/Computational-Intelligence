import os
import sys

from crossovers import Crossovers
from mutation import Mutations
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
        "3": ("Uniform Crossover", "uniform_crossover"),
        "4": ("Simple Crossover", "simple"),
        "5": ("Simple Arithmetic Crossover", "simple_arithmetic"),
        "6": ("Whole Arithmetic Crossover", "whole_crossover"),
        "7": ("Order Recombination Crossover", "Order_Recombination"),
        "8": ("Cycle Recombination Crossover", "Cycle_Recombination"),
    }

    mutation_methods = {
        "1": ("Bit Flip", "bit_flip"),  # Binary only
        "2": ("Swap Mutation", "swap_mutation"),  # Permutation only
        "3": ("Scramble Mutation", "scramble_mutation"),  # Permutation only
        "4": ("Inversion Mutation", "inversion_mutation"),  # Permutation only
        "5": ("Insert Mutation", "insert_mutation"),  # Permutation only
        "6": ("Complement Mutation", "complement_mutation"),
    }

    binary_crossovers = {}
    permute_crossovers = {"Order_Recombination", "Cycle_Recombination"}
    needs_permute_mut = {
        "swap_mutation",
        "scramble_mutation",
        "inversion_mutation",
        "insert_mutation",
    }

    clear()
    selection_choice = choose_method(selection_methods, "selection")
    clear()
    crossover_choice = choose_method(crossover_methods, "crossover")
    clear()

    crossover_func_name = crossover_methods[crossover_choice][1]
    crossover_func = getattr(crossover, crossover_func_name)
    needs_binary = crossover_func_name in binary_crossovers
    needs_permute = crossover_func_name in permute_crossovers

    chromosomes_count = get_int("How many chromosomes should your population have? ")
    clear()

    for key, (name, _) in mutation_methods.items():
        print(f"{key}. {name}")

    mutation_choice = input(
        "Enter number for mutation (leave empty for none): "
    ).strip()

    mutation_func = None

    if mutation_choice in mutation_methods:
        func_name = mutation_methods[mutation_choice][1]

        # Check compatibility
        if func_name == "bit_flip" and not needs_binary:
            print("Bit Flip mutation requires binary chromosomes. Skipping mutation.")
        elif func_name in needs_permute_mut and not needs_permute:
            print(f"{func_name} requires permutation chromosomes. Skipping mutation.")
        else:
            mutation_func = getattr(Mutations, func_name)

    chromo_gen = ChromosomeGen(chromosomes_count, needs_binary, needs_permute)
    population = chromo_gen.gene_generator()
    selected = []

    # Selection Phase
    if selection_choice == "1":
        wanted = get_int("How many chromosomes does the next generation need? ")
        selected = select.random_selection(population, wanted)

    elif selection_choice == "2":
        prob = Prob(population)
        prob.fitness()
        prob.prob()
        selected = select.roulette_wheel(population, prob.cumul())

    elif selection_choice == "3":
        fitnesses = Prob(population).fitness()
        selected = select.rank_selection(population, fitnesses)

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

    alpha = None
    num = None

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

    if mutation_func is not None:
        for i in range(len(offspring)):
            child = (
                list(offspring[i]) if isinstance(offspring[i], tuple) else offspring[i]
            )
            offspring[i] = mutation_func(child)

    print("\nOffspring after mutation:")
    for i, child in enumerate(offspring):
        print(f"{i + 1}: {child}")


if __name__ == "__main__":
    main()
