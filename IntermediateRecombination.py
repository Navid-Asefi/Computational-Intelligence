from tools.generations import ChromosomeGen


def whole_crossover():
    chromo = ChromosomeGen(2)

    parents = chromo.gene_generator()

    alpha = float(input("Type the desired alpha: "))
    child = [
        round(alpha * a + (1 - alpha) * b, 2) for a, b in zip(parents[0], parents[1])
    ]

    child2 = [
        round(alpha * a + (1 - alpha) * b, 2) for a, b in zip(parents[1], parents[0])
    ]
    return parents, child, child2


parents, child1, child2 = whole_crossover()

print(f"The parents: {parents}\n ")
print(f"The first child: {child1}\n")
print(f"the second child: {child2}")
