from tools.generations import ChromosomeGen
import random

def two_point(parent1,parent2):
    #pointer
    p1=random.randint(1,9)
    p2=random.randint(1,9)
    while p1==p2:
        p2=random.randint(1,9)
    if p1 > p2:
        temp=p1
        p1=p2
        p2=temp
            
    child1=parent1[0:p1]+parent2[p1:p2]+parent1[p2:10]
    child2=parent2[0:p1]+parent1[p1:p2]+parent2[p2:10]
    return child1,child2,p1,p2



chromosomes=ChromosomeGen(2, binary=True)
parents=chromosomes.gene_generator()

parent1=parents[0]
parent2=parents[1]
 
a,b,c,d=two_point(parent1,parent2)
print(f"pointer1={c} pointer2={d}\n{a}\n{b}")