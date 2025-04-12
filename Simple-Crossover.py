from tools.generations import ChromosomeGen
import sys

def recombination():

    chromosome = ChromosomeGen(2)
    parents = chromosome.gene_generator()
    child1 = []
    child2 = []

    try:
        alpha = float(input("Type the desired alpha: "))
        if alpha < 0 or alpha > 1:
            raise ValueError
        
        num = int(input("Number of genes you desire to change into the parents' convexed linear combination: "))
        if num > 10:
            raise ValueError
        
        for j in range(0, 10-num):
            child1.append(parents[0][j])
            child2.append(parents[1][j])

        for i in range(10-num, 10):
            child1.append(round((alpha*parents[0][i]) + ((1-alpha)*parents[1][i]), 2))
            child2.append(round((alpha*parents[1][i]) + ((1-alpha)*parents[0][i]), 2))
        
    except ValueError:
        print("You miss indexed genes or entered the wrong number of alpha")
        print("Remember the alpha should be between 0 and 1")
        print("And there are only 10 genes in each chromosome")
        sys.exit()

    return child1, child2, parents


child1, child2, parents = recombination()

print(f"The parents: {parents}\n ")
print(f"The first child: {child1}\n")
print(f"the second child: {child2}")

