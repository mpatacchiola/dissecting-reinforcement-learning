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
#Implementation of the Genetic Algorithm for policy estimation for the
#cleaning robot example (grid-world). This example is part to the 'dissecting
#reinforcement learning' series on my personal blog:
# https://mpatacchiola.github.io/blog/

import numpy as np
from gridworld import GridWorld

def return_random_population(population_size, chromosome_size):
    '''Returns a random initialised population

    This funtion initialise a matrix of integers
    in the interval [0,3] using numpy randint.
    @param chromosome_size 
    @param population_size
    @return matrix of integers size:
         population_size x chromosome_size
    '''
    return np.random.randint(low=0, 
                             high=4, 
                             size=(population_size,chromosome_size))

#def return_best_worst_population(population, fitness_array):
    

def return_mutated_population(population, mutation_rate, elite=0):
    '''Returns a mutated population

    It applies the point-mutation mechanism to each value
    contained in the chromosomes.
    @param population numpy array containing the chromosomes
    @parma mutation_rate a float repesenting the probaiblity 
        of mutation for each gene (e.g. 0.02=2%)
    @return the mutated population
    '''
    for x in np.nditer(population[elite:,:], op_flags=['readwrite']):
        if(np.random.uniform(0,1) < mutation_rate):
            x[...] = np.random.choice(4, 1)
    return population


def return_roulette_selected_population(population, fitness_array):
  '''Returns a new population of individuals (roulette wheel).

  Implementation of a roulette wheel mechanism. The population returned
  is obtained through a weighted sampling based on the fitness array.
  @param population numpy matrix containing the population chromosomes
  @param fitness_array numpy array containing the fitness score for
      each chromosomes
  @return a new population of roulette selected chromosomes.
  '''
  #Softmax to obtain a probability distribution from the fitness array.
  fitness_distribution = np.exp(fitness_array - 
                         np.max(fitness_array))/np.sum(np.exp(fitness_array - 
                                                       np.max(fitness_array)))
  #Selecting the new population indeces through a weighted sampling
  pop_size = population.shape[0]
  pop_indeces = np.random.choice(pop_size, pop_size, p=fitness_distribution)
  #New population initialisation
  new_population = np.zeros(population.shape)
  #Assign the chromosomes in population to new_population
  for i in pop_indeces:
      new_population[i,:] = population[i,:]
  return new_population
    

def main():
    population_size = 100
    elit_size = 10
    mutation_rate = 0.05
    chromosome_size = 12
    tot_generations = 100
    print_generation = 1000
    env = GridWorld(3,4)

    mean_fitness_list = list()
    max_fitness_list = list()
    min_fitness_list = list()

    #Define the state matrix
    state_matrix = np.zeros((3,4))
    state_matrix[0, 3] = 1
    state_matrix[1, 3] = 1
    state_matrix[1, 1] = -1
    print("State Matrix:")
    print(state_matrix)

    #Define the reward matrix
    reward_matrix = np.full((3,4), -0.04)
    reward_matrix[0, 3] = 1
    reward_matrix[1, 3] = -1
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
                                                 chromosome_size)
    print("Population matrix shape: " + str(population_matrix.shape))

    #main iteration loop
    for generation in range(tot_generations):
        #The fitness value for each individual is stored in np.array
        fitness_array = np.zeros((population_size))
        for chromosome_index in range(population_size):
          for episode in range(100):
            #Reset and return the first observation
            observation = env.reset(exploring_starts=True)
            for step in range(30):
              #Estimating the action for that state
              col = observation[1] + (observation[0]*4)
              action = population_matrix[chromosome_index,:][col]
              #Taking the action and observing the new state and reward
              observation, reward, done = env.step(action)
              #Accumulating the fitness for this individual
              fitness_array[chromosome_index] += reward
              if done: break

        population_matrix = return_roulette_selected_population(population_matrix, fitness_array)
        population_matrix = return_mutated_population(population_matrix, mutation_rate)

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
        print("")

    #Time to check
    print("")


if __name__ == "__main__":
    main()
