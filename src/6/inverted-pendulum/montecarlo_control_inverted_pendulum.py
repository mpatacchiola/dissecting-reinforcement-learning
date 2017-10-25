#!/usr/bin/env python

#MIT License
#Copyright (c) 2017 Massimiliano Patacchiola
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#Example of Monte Carlo methods for control.
#In this example I will use the class invertedPendulum to generate an
#environment in which the cleaning robot will move. Using the Monte Carlo method
#I will estimate the policy and the state-action matrix of each state.

import numpy as np
from inverted_pendulum import InvertedPendulum
import matplotlib.pyplot as plt

def print_policy(policy_matrix):
    """Print the policy using specific symbol.

    O noop, < left, > right
    """
    counter = 0
    shape = policy_matrix.shape
    policy_string = ""
    for row in range(shape[0]):
        for col in range(shape[1]):           
            if(policy_matrix[row,col] == 0): policy_string += " <  "
            elif(policy_matrix[row,col] == 1): policy_string += " O  "
            elif(policy_matrix[row,col] == 2): policy_string += " >  "           
            counter += 1
        policy_string += '\n'
    print(policy_string)

def get_return(state_list, gamma):
    """Get the return for a list of action-state values.

    @return get the Return
    """
    counter = 0
    return_value = 0
    for visit in state_list:
        reward = visit[2]
        return_value += reward * np.power(gamma, counter)
        counter += 1
    return return_value

def update_policy(episode_list, policy_matrix, state_action_matrix, tot_bins):
    """Update a policy making it greedy in respect of the state-action matrix.

    @return the updated policy
    """
    for visit in episode_list:
        observation = visit[0]
        col = observation[1] + (observation[0]*tot_bins)
        if(policy_matrix[observation[0], observation[1]] != -1):      
            policy_matrix[observation[0], observation[1]] = \
                np.argmax(state_action_matrix[:,col])
    return policy_matrix

def return_decayed_value(starting_value, minimum_value, global_step, decay_step):
    """Returns the decayed value.

    decayed_value = starting_value * decay_rate ^ (global_step / decay_steps)
    @param starting_value the value before decaying
    @param global_step the global step to use for decay (positive integer)
    @param decay_step the step at which the value is decayed
    """
    decayed_value = starting_value * np.power(0.9, (global_step/decay_step))
    if decayed_value < minimum_value:
            return minimum_value
    else:
            return decayed_value

def return_epsilon_greedy_action(policy_matrix, observation, epsilon=0.1):
    """Return an action choosing it with epsilon-greedy

    @param policy_matrix the matrix before the update
    @param observation the state obsrved at t
    @param epsilon the value used for computing the probabilities
    @return the updated policy_matrix
    """
    tot_actions = int(np.nanmax(policy_matrix) + 1)
    action = int(policy_matrix[observation[0], observation[1]])
    non_greedy_prob = epsilon / tot_actions
    greedy_prob = 1 - epsilon + non_greedy_prob
    weight_array = np.full((tot_actions), non_greedy_prob)
    weight_array[action] = greedy_prob
    return np.random.choice(tot_actions, 1, p=weight_array)

def plot_curve(data_list, filepath="./my_plot.png", 
               x_label="X", y_label="Y", 
               x_range=(0, 1), y_range=(0,1), color="-r", kernel_size=50, alpha=0.4, grid=True):
        """Plot a graph using matplotlib

        """
        if(len(data_list) <=1):
            print("[WARNING] the data list is empty, no plot will be saved.")
            return
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=x_range, ylim=y_range)
        ax.grid(grid)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.plot(data_list, color, alpha=alpha)  # The original data is showed in background
        kernel = np.ones(int(kernel_size))/float(kernel_size)  # Smooth the graph using a convolution
        tot_data = len(data_list)
        lower_boundary = int(kernel_size/2.0)
        upper_boundary = int(tot_data-(kernel_size/2.0))
        data_convolved_array = np.convolve(data_list, kernel, 'same')[lower_boundary:upper_boundary]
        #print("arange: " + str(np.arange(tot_data)[lower_boundary:upper_boundary]))
        #print("Convolved: " + str(np.arange(tot_data).shape))
        ax.plot(np.arange(tot_data)[lower_boundary:upper_boundary], data_convolved_array, color, alpha=1.0)  # Convolved plot
        fig.savefig(filepath)
        fig.clear()
        plt.close(fig)
        # print(plt.get_fignums())  # print the number of figures opened in background

def main():

    env = InvertedPendulum(pole_mass=2.0, cart_mass=8.0, pole_lenght=0.5, delta_t=0.1)

    # Define the state arrays for velocity and position
    tot_action = 3  # Three possible actions
    tot_bins = 12  # the value used to discretize the space
    velocity_state_array = np.linspace(-np.pi, np.pi, num=tot_bins-1, endpoint=False)
    position_state_array = np.linspace(-np.pi/2.0, np.pi/2.0, num=tot_bins-1, endpoint=False)

    #Random policy
    policy_matrix = np.random.randint(low=0, high=tot_action, size=(tot_bins,tot_bins))
    print("Policy Matrix:")
    print_policy(policy_matrix)

    state_action_matrix = np.zeros((tot_action, tot_bins*tot_bins))
    #init with 1.0e-10 to avoid division by zero
    running_mean_matrix = np.full((tot_action, tot_bins*tot_bins), 1.0e-10) 
    gamma = 0.999
    tot_episode = 500000 # 500k
    epsilon_start = 0.99  # those are the values for epsilon decay
    epsilon_stop = 0.1
    epsilon_decay_step = 10000
    print_episode = 500  # print every...
    movie_episode = 20000  # movie saved every...
    reward_list = list()
    step_list = list()

    for episode in range(tot_episode):
        epsilon = return_decayed_value(epsilon_start, epsilon_stop, episode, decay_step=epsilon_decay_step)
        #Starting a new episode
        episode_list = list()
        #Reset and return the first observation and reward
        observation = env.reset(exploring_starts=True)
        observation = (np.digitize(observation[1], velocity_state_array), 
                       np.digitize(observation[0], position_state_array))
        #action = np.random.choice(4, 1)
        #action = policy_matrix[observation[0], observation[1]]
        #episode_list.append((observation, action, reward))
        is_starting = True
        cumulated_reward = 0
        for step in range(100):
            #Take the action from the action matrix
            action = return_epsilon_greedy_action(policy_matrix, observation, epsilon=epsilon)
            #If the episode just started then it is
                #necessary to choose a random action (exploring starts)
            if(is_starting): 
                action = np.random.randint(0, tot_action)
                is_starting = False   
            #if(episode % print_episode == 0):
            #    print("Step: " + str(step) + "; Action: " + str(action) + "; Angle: " + str(observation[0]) + "; Velocity: " + str(observation[1]))   
            #Move one step in the environment and get obs and reward
            new_observation, reward, done = env.step(action)
            new_observation = (np.digitize(new_observation[1], velocity_state_array), 
                               np.digitize(new_observation[0], position_state_array))  
            #Append the visit in the episode list
            episode_list.append((observation, action, reward))
            observation = new_observation
            cumulated_reward += reward
            if done: break
        #The episode is finished, now estimating the utilities
        counter = 0
        #Checkup to identify if it is the first visit to a state
        checkup_matrix = np.zeros((tot_action, tot_bins*tot_bins))
        #This cycle is the implementation of First-Visit MC.
        #For each state stored in the episode list check if it
        #is the rist visit and then estimate the return.
        for visit in episode_list:
            observation = visit[0]
            action = visit[1]
            col = observation[1] + (observation[0]*tot_bins)
            row = action
            if(checkup_matrix[row, col] == 0):
                return_value = get_return(episode_list[counter:], gamma)
                running_mean_matrix[row, col] += 1
                state_action_matrix[row, col] += return_value
                checkup_matrix[row, col] = 1
            counter += 1
        #Policy Update
        policy_matrix = update_policy(episode_list, 
                                      policy_matrix, 
                                      state_action_matrix/running_mean_matrix,
                                      tot_bins)
        # Store the data for statistics
        reward_list.append(cumulated_reward)
        step_list.append(step)
        # Printing utilities
        if(episode % print_episode == 0):
            print("")
            print("Episode: " + str(episode+1))
            print("Epsilon: " + str(epsilon))
            print("Episode steps: " + str(step+1))
            print("Cumulated Reward: " + str(cumulated_reward))
            print("Policy matrix: ") 
            print_policy(policy_matrix)
        if(episode % movie_episode == 0):
            print("Saving the reward plot in: ./reward.png")
            plot_curve(reward_list, filepath="./reward.png", 
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(reward_list)), y_range=(-0.1,100),
                       color="red", kernel_size=500, 
                       alpha=0.4, grid=True)
            print("Saving the step plot in: ./step.png")
            plot_curve(step_list, filepath="./step.png", 
                       x_label="Episode", y_label="Steps", 
                       x_range=(0, len(step_list)), y_range=(-0.1,100),
                       color="blue", kernel_size=500, 
                       alpha=0.4, grid=True)
            print("Saving the gif in: ./inverted_pendulum.gif")
            env.render(file_path='./inverted_pendulum.gif', mode='gif')
            print("Complete!")

    print("Policy matrix after " + str(tot_episode) + " episodes:")
    print_policy(policy_matrix)

if __name__ == "__main__":
    main()
