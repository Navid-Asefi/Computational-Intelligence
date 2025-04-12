from tools.generations import ChromosomeGen
import random

def one_point(parent1,parent2):
    #pointer
    p=random.randint(0,9)
    child1=parent1[0:p]+parent2[p:10]
    child2=parent2[0:p]+parent1[p:10]
    return child1,child2,p



chromosomes=ChromosomeGen(2, binary=True)
parents=chromosomes.gene_generator()

parent1=parents[0]
parent2=parents[1]
    
a,b,c=one_point(parent1,parent2)
print(parent1,parent2)
print(f"{c}\n{a}\n{b}")