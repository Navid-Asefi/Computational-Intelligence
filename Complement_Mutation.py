
import random
from tools.generations import ChromosomeGen

chromosomes = ChromosomeGen(1,permute=True)
parents = chromosomes.gene_generator()
parent=parents[0]

def complement_mutation(parent):
    """
    جهش مکمل برای کروموزوم‌های جایگشتی 0 تا 9
    
    Parameters:
    chromosome: کروموزوم ورودی (لیستی از اعداد 0 تا 9)
    num_genes_to_mutate: تعداد ژن‌هایی که باید جهش یابند
    
    Returns:
    کروموزوم جهش یافته
    """
    child = parent.copy()
    num_genes_to_mutate=random.randint(1,10)
    genes_to_mutate = random.sample(range(len(parent)), num_genes_to_mutate)
    

    for gene_index in genes_to_mutate:
        current_value = child[gene_index]
        child[gene_index] = 9 - current_value  
    
    return child

child=complement_mutation(parent)
print(f"{parent}\n,{child}")
 