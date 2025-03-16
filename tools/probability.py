class Prob:

    def __init__(self, vector):
        self.vector = vector
        self.chr_sum = []
        self.probability = []
        self.cumulative = []

    def fitness(self):
        """Calculating the fitness of each chromosome"""

        for i in range(len(self.vector)):
            gen_sum = 0
            for j in range(10):
                gen_sum += self.vector[i][j] ** 2
            gen_sum = 1 / (1 + gen_sum)
            self.chr_sum.append(gen_sum)
        return (self.chr_sum)
    
    def prob(self):

        """Calculating the probability of chromosomes being choosed 
        based on the fitness of each chromosome"""

        sum1 = sum(self.chr_sum)
        for i in range(len(self.chr_sum)):
            self.probability.append(self.chr_sum[i] / sum1)
        return self.probability
    
    def cumul(self):

        """Calculating the cumulative probability for future usage"""

        sum = 0
        for i in range(len(self.probability)):
            sum += self.probability[i]
            self.cumulative.append(sum)
        return self.cumulative


