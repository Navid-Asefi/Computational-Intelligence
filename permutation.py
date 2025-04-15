from tools.generations import ChromosomeGen
import random

chromosomes = ChromosomeGen(2, permute=True)
parents = chromosomes.gene_generator()

def Order_Recombination(parents):

    a = random.randint(0, 9)
    b = random.randint(0, 9)

    child1 = []
    child2 = []

    for _ in range(10):
        child1.append("x")
        child2.append("x")

    if a < b:
        child1[a:b] = parents[0][a:b]
        child2[a:b] = parents[1][a:b]

        indx = child1.count("x")
        fill(indx, b, child1, child2, parents)

    elif b < a:
        child1[b:a] = parents[0][b:a]
        child2[b:a] = parents[1][b:a]

        indx = child1.count("x")
        fill(indx, a, child1, child2, parents)


    return child1, child2, a, b, indx


def fill(indx, b, child1, child2, parents):

    for j in range(indx):
        print(child1)
        for i in range(10):
            if not parents[1][(b+i)%10] in child1:
                child1[(b+j)%10] = parents[1][(b+i)%10]

            if not parents[0][(b+i)%10] in child2:
                child2[(b+j)%10] = parents[0][(b+i)%10]

            continue

print(parents, Order_Recombination(parents))