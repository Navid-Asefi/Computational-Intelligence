import random
from probability import Prob

class Selection:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def roulette_wheel(population, cumulative_probability):
        rand = random.uniform(0, 1)
        for i, cp in enumerate(cumulative_probability):
            if rand <= cp:
                return population[i]
    
    @staticmethod
    def random_selection(population, size_of_group):
        new_gen = []
        while len(new_gen) < size_of_group :
            adam = random.choice(population)
            new_gen.append(adam)
        return new_gen
    
    @staticmethod
    def tournament_selection(population, size, count): 
        best = []
        for _ in range(count):
            selection = random.sample(population, size)
            prob = Prob(selection)
            selection_with_fit = list(zip(selection, prob.fitness()))
            best.append(max(selection_with_fit, key=lambda x: x[1]))
        return best

    @staticmethod
    def rank_selection(community, fitness):
        combined = list(zip(fitness, community))

        combined.sort(key=lambda x: x[0])

        fitness_sorted, community_sorted = zip(*combined)

        rank = [(len(community_sorted) + (i + 1)) - len(community_sorted) for i in range(len(community_sorted))]

        cumulative = []
        sum = 0
        for r in rank:
            sum += r
            cumulative.append(sum)

        rand = random.uniform(0, 1)
        rand = rand * cumulative[-1]

        for i in range(len(cumulative)):
            if rand <= cumulative[i]:
                print(f"Sorted community: {community_sorted}\nTheir rank: {rank}\nRandom {rand} generated in {cumulative} cumulative.")
                result = community_sorted[i]  # Assuming you want to return this value
                break

        return result

    @staticmethod
    def truncation_selection(population, selection_ratio,size_of_group):   
         
        fitness_values = list(zip(population, Prob(population).fitness()))

        fitness_values.sort(key=lambda x: x[1], reverse=True)
        
        num_selected = int(len(population) * selection_ratio)
        
        selected_individuals = [community for community, _ in fitness_values[:num_selected]]
        
        new_gen = []
        while len(new_gen) < size_of_group :
            adam = random.choice(selected_individuals)
            new_gen.append(adam)
        return new_gen

