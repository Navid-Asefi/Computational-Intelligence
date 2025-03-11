from generations import chromosomeGen
from probability import Prob
import random

def rank_selection(community, fitness):

    # 'community' and 'fitness' are lists of the same length
    # and 'community' is sorted according to 'fitness'.

    # Create a combined list of tuples, pairing fitness values with communities
    combined = list(zip(fitness, community))

    # Sort the combined list by the first element (fitness)
    combined.sort(key=lambda x: x[0])  # Sorts based on fitness in ascending order

    # Unzip the sorted tuples back into separate lists
    fitness_sorted, community_sorted = zip(*combined)

    # Calculate rank
    rank = [(len(community_sorted) + (i + 1)) - len(community_sorted) for i in range(len(community_sorted))]

    # Perform cumulative sum for the rank
    cumulative = []
    sum = 0
    for r in rank:
        sum += r
        cumulative.append(sum)

    # Generate a random number and find the corresponding community
    rand = random.uniform(0, 1)
    rand = rand * cumulative[-1]

    for i in range(len(cumulative)):
        if rand <= cumulative[i]:
            print(f"Sorted community: {community_sorted}\nTheir rank: {rank}\nRandom {rand} generated in {cumulative} cumulative.")
            result = community_sorted[i]  # Assuming you want to return this value
            break

    return result



people = int(input("Enter the number of wanted chromosomes: "))
pop = chromosomeGen(people)
pop = pop.gene_generator()
rank = Prob(pop)
rank_fit = rank.fitness()
# print community list based on their rank from index 0 to n-1 (greatest fitness to lowest fitness)
print(f"Unsorted community: {pop}\nTheir fitnesses: {rank_fit}")
print(f"Chosen chromosome: {rank_selection(pop, rank_fit)}")