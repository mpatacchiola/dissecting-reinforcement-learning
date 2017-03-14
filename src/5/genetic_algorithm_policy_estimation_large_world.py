#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io/blog/
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in 
#all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#PEP-8 format: Limit all lines to a maximum of 79 characters ----------------|
#
#The only dependence is Numpy. You need matplotlib if you want to save 
#the fitness plot.
#This is an mplementation of the Genetic Algorithm for policy estimation for 
#the cleaning robot example (4x3 grid-world). This example is part to the 
#'dissecting reinforcement learning' series on my personal blog:
# https://mpatacchiola.github.io/blog/


import numpy as np
from gridworld import GridWorld
import sys

def return_random_population(population_size, chromosome_size, gene_set):
    '''Returns a random initialised population

    This funtion initialise a matrix of integers
    in the interval [0,3] using numpy randint.
    @param chromosome_size 
    @param population_size
    @param gene_set list or array containing the gene values
    @return matrix of integers size:
         population_size x chromosome_size
    '''

    return np.random.choice(gene_set, 
                            size=(population_size,chromosome_size))


def return_best_worst_population(population, fitness_array):
    '''Returns the population sorted in best-worst order

    @param population numpy matrix containing the population chromosomes
    @param fitness_array numpy array containing the fitness score for
        each chromosomes
    @return the new population and the new fitness array
    '''
    new_population = np.zeros(population.shape)
    new_fitness_array = np.zeros(fitness_array.shape)
    worst_best_indeces = np.argsort(fitness_array)
    best_worst_indeces = worst_best_indeces[::-1] #reverse the array
    row_counter = 0
    for index in best_worst_indeces:
         new_population[row_counter,:] = population[index,:]
         new_fitness_array[row_counter] = fitness_array[index]
         row_counter += 1
    return new_population, new_fitness_array

def return_mutated_population(population, gene_set, mutation_rate, elite=0):
    '''Returns a mutated population

    It applies the point-mutation mechanism to each value
    contained in the chromosomes.
    @param population numpy array containing the chromosomes
    @param gene_set a numpy array with the value to pick
    @parma mutation_rate a float repesenting the probaiblity 
        of mutation for each gene (e.g. 0.02=2%)
    @return the mutated population
    '''
    for x in np.nditer(population[elite:,:], op_flags=['readwrite']):
        if(np.random.uniform(0,1) < mutation_rate):
            x[...] = np.random.choice(gene_set, 1)
    return population

def return_truncated_population(population, fitness_array, new_size):
     '''Truncates the input population and returns part of the matrix

     @param population numpy array containing the chromosomes
     @param fitness_array numpy array containing the fitness score for
         each chromosomes
     @param new_size the size of the new population
     @return a population containing new_size chromosomes 
     '''
     chromosome_size = population.shape[1]
     pop_size = population.shape[0]
     new_population = np.resize(population, (new_size,chromosome_size))
     new_fitness_array = np.resize(fitness_array, new_size)
     return new_population, new_fitness_array

def return_roulette_selected_population(population, fitness_array, new_size):
  '''Returns a new population of individuals (roulette wheel).

  Implementation of a roulette wheel mechanism. The population returned
  is obtained through a weighted sampling based on the fitness array.
  @param population numpy matrix containing the population chromosomes
  @param fitness_array numpy array containing the fitness score for
      each chromosomes
  @param new_size the size of the new population
  @return a new population of roulette selected chromosomes, and
          the fitness array reorganised based on the new population.
  '''
  #Softmax to obtain a probability distribution from the fitness array.
  fitness_distribution = np.exp(fitness_array - 
                         np.max(fitness_array))/np.sum(np.exp(fitness_array - 
                                                       np.max(fitness_array)))
  #Selecting the new population indeces through a weighted sampling
  pop_size = population.shape[0]
  chromosome_size = population.shape[1]
  pop_indeces = np.random.choice(pop_size, new_size, p=fitness_distribution)
  #New population initialisation
  new_population = np.zeros((new_size, chromosome_size))
  new_fitness_array = np.zeros(new_size)
  #Assign the chromosomes in population to new_population
  row_counter = 0
  for i in pop_indeces:
      new_population[row_counter,:] = np.copy(population[i,:]) 
      new_fitness_array[row_counter] = np.copy(fitness_array[i])
      row_counter += 1
  return new_population, new_fitness_array

def return_crossed_population(population, new_size, elite=0):
    '''Return a new population based on the crossover of the individuals

    The parents are randomly chosen. Each pair of parents generates
    only one child. The slicing point is randomly chosen.
    @param population numpy matrix containing the population chromosomes
    @param new_size defines the size of the new population
    @param elite defines how many chromosomes remain unchanged
    @return a new population of crossed individuals
    '''
    pop_size = population.shape[0]
    chromosome_size = population.shape[1] 
    if(elite > new_size): 
        ValueError("Error: the elite value cannot " +
                    "be larger than the population size")
    new_population = np.zeros((new_size,chromosome_size))
    #Copy the elite into the new population matrix
    new_population[0:elite] = population[0:elite]
    #Randomly pick the parents to cross
    parents_index = np.random.randint(low=0, 
                                       high=pop_size, 
                                       size=(new_size-elite,2))
    #Generating the remaining individuals through crossover
    for i in range(elite,new_size-elite):
        first_parent = population[parents_index[i,0], :]
        second_parent = population[parents_index[i,1], :]
        slicing_point = np.random.randint(low=0, high=chromosome_size)
        child = np.zeros(chromosome_size)
        child[0:slicing_point] = first_parent[0:slicing_point]
        child[slicing_point:] = second_parent[slicing_point:]
        new_population[i] = np.copy(child)
    return new_population

#def return_chromosome_string(chromosome_array):
#    '''Returns a string where the actions of the chromosome
#    are replaced with symbols.
#
#    Attention, this function only works for the 4x3 gridworld
#    It must be readapted for larger worlds
#    @param chromosome_array
#    @return a string of symbols
#    '''
#    chromosome_string = ""
#    counter=0
#    for gene in chromosome_array:
#        if(counter==3): chromosome_string += ' * '
#        elif(counter==7): chromosome_string += ' * '
#        elif(counter==5): chromosome_string += ' # '
#        else:
#            if(gene == 0): chromosome_string += ' ^ '
#            elif(gene == 1): chromosome_string += ' > '
#            elif(gene == 2): chromosome_string += ' v '
#            elif(gene == 3): chromosome_string += ' < '
#        counter += 1
#    return chromosome_string

def main():
    tot_generations = 300
    tot_episodes = 100
    tot_steps = 80 #a good value is (world_rows + world_cols) * 2
    population_size = 100
    elite_size = 10
    mutation_rate = 0.10
    gene_set = [0, 1, 2, 3]
    chromosome_size = 300 #world_rows * world_cols

    mean_fitness_list = list()
    max_fitness_list = list()
    min_fitness_list = list()

    #Define the world dimension
    world_rows = 10
    world_columns = 30
    env = GridWorld(world_rows,world_columns)

    #Define the state matrix
    state_matrix = np.zeros((10,30))
    state_matrix[0, 29] = 1
    state_matrix[1, 29] = 1
    state_matrix[1, 1] = -1
    state_matrix[5, 5] = -1
    state_matrix[3, 7] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((10,30), -0.04)
    reward_matrix[0, 29] = 1
    reward_matrix[1, 29] = -1
    print("Reward Matrix:")
    print(reward_matrix)

    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])

    env.setStateMatrix(state_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)

    #Init a random population
    population_matrix = return_random_population(population_size, 
                                                 chromosome_size,
                                                 gene_set = gene_set)
    print("Population matrix shape: " + str(population_matrix.shape))

    #Main iteration loop
    for generation in range(tot_generations):
        #The fitness value for each individual is stored in np.array
        fitness_array = np.zeros((population_size))
        for chromosome_index in range(population_size):
          for episode in range(tot_episodes):
            #Reset and return the first observation
            observation = env.reset(exploring_starts=True)
            for step in range(tot_steps):
              #Estimating the action for that state
              col = observation[1] + (observation[0]*world_columns)
              action = population_matrix[chromosome_index,:][col]
              #Taking the action and observing the new state and reward
              observation, reward, done = env.step(action)
              #Accumulating the fitness for this individual
              fitness_array[chromosome_index] += reward
              if done: break

        #Printing and saving Fitness information
        max_fitness_list.append(np.amax(fitness_array))
        mean_fitness_list.append(np.mean(fitness_array))
        min_fitness_list.append(np.amin(fitness_array))
        print("Generation: " + str(generation+1))
        print("Fitness Mean: " + str(np.mean(fitness_array)))
        print("Fitness STD: " + str(np.std(fitness_array)))
        print("Fitness Max: " + str(np.amax(fitness_array))
              + " at index " + str(np.argmax(fitness_array)))
        print("Fitness Min: " + str(np.amin(fitness_array))
              + " at index " + str(np.argmin(fitness_array)))
        for i in range(int(fitness_array.shape[0]/10)):
            print("Fitness " + str(i) + " ..... " + str(fitness_array[i]))    
        print("")

        #Uncomment the following line to enable roulette wheel selection
        #population_matrix, fitness_array = \
            #return_roulette_selected_population(population_matrix,                                                  
                                                #fitness_array,
                                                #population_size)
        population_matrix, fitness_array = \
            return_best_worst_population(population_matrix, fitness_array)
        #Comment the following line if you enable the truncated selection
        population_matrix, fitness_array = \
            return_truncated_population(population_matrix, 
                                        fitness_array, 
                                        new_size=int(population_size/2))
        population_matrix = return_crossed_population(population_matrix, 
                                                      population_size, 
                                                      elite=elite_size)
        population_matrix = return_mutated_population(population_matrix,
                                                      gene_set=gene_set,
                                                      mutation_rate=mutation_rate, 
                                                      elite=elite_size)

    #If you have matplotlib installed it saves an image of
    #the fitness/generation plot
    try:
        import matplotlib.pyplot as plt
        print("Using matplotlib to show the fitness/generation plot...")
        array = np.arange(1, tot_generations+1, dtype='int32')
        plt.plot(array, mean_fitness_list,  color='red', marker='o', markersize=6, markevery=10, label='Mean')
        plt.plot(array, max_fitness_list, color='blue', marker='^', markersize=6, markevery=10, label='Max')
        #plt.plot(array, min_fitness_list, color='black', marker='v', markersize=6, markevery=10, label='Min')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        #plt.xlim((0,tot_generations))
        #plt.ylim((-100,+100))
	plt.ylabel('Fitness', fontsize=15)
	plt.xlabel('Generation', fontsize=15)
        print("Saving the image in './fitness.jpg'...")
        plt.savefig("./fitness.jpg", dpi=500)
	#plt.show()
    except ImportError, e:
        print("Please install matplotlib if you want to see the fitness/generation plot.")
        pass # module doesn't exist, deal with it.


if __name__ == "__main__":
    main()
