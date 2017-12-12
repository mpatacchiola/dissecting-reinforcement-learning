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

#In this script the TD(0) linear approximator is used to estimate the utilities
#of the boolean worlds.

import numpy as np
from gridworld import GridWorld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d

def init_and(bias=True):
    '''Init the boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[-1.0, 0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0, -1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    if bias:
        return env, np.random.uniform(-1, 1, 3)
    else:
        return env, np.random.uniform(-1, 1, 2)

def init_nand(bias=True):
    '''Init the boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[1.0, 0.0, 0.0, 0.0, -1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    if bias:
        return env, np.random.uniform(-1, 1, 3)
    else:
        return env, np.random.uniform(-1, 1, 2)

def init_or(bias=True):
    '''Init the boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    if bias:
        return env, np.random.uniform(-1, 1, 3)
    else:
        return env, np.random.uniform(-1, 1, 2)

def init_xor(bias=True):
    '''Init the boolean environment

    @return the environment gridworld object
    '''
    env = GridWorld(5, 5)
    #Define the state matrix
    state_matrix = np.array([[1.0, 0.0, 0.0, 0.0, 1.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the index matrix
    index_matrix = np.array([[(4,0), (4,1), (4,2), (4,3), (4,4)],
                             [(3,0), (3,1), (3,2), (3,3), (3,4)],
                             [(2,0), (2,1), (2,2), (2,3), (2,4)],
                             [(1,0), (1,1), (1,2), (1,3), (1,4)],
                             [(0,0), (0,1), (0,2), (0,3), (0,4)]])
    #Define the reward matrix
    reward_matrix = np.array([[1.0, 0.0, 0.0, 0.0, -1.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0],
                              [-1.0, 0.0, 0.0, 0.0, 1.0]])
    #Define the transition matrix
    transition_matrix = np.array([[0.8, 0.1, 0.0, 0.1],
                                  [0.1, 0.8, 0.1, 0.0],
                                  [0.0, 0.1, 0.8, 0.1],
                                  [0.1, 0.0, 0.1, 0.8]])
    env.setStateMatrix(state_matrix)
    env.setIndexMatrix(index_matrix)
    env.setRewardMatrix(reward_matrix)
    env.setTransitionMatrix(transition_matrix)
    if bias:
        return env, np.random.uniform(-1, 1, 3)
    else:
        return env, np.random.uniform(-1, 1, 2)

def update(w, x, x_t1, reward, alpha, gamma, done):
    '''Return the updated weights vector w_t1

    @param w the weights vector before the update
    @param x the feauture vector obsrved at t
    @param x_t1 the feauture vector observed at t+1
    @param reward the reward observed after the action
    @param alpha the ste size (learning rate)
    @param gamma the discount factor
    @param done boolean True if the state is terminal
    @return w_t1 the weights vector at t+1
    '''
    if done:
        w_t1 = w + alpha * ((reward - np.dot(x,w)) * x)
    else:
        w_t1 = w + alpha * ((reward + (gamma*(np.dot(x_t1,w))) - np.dot(x,w)) * x)
    return w_t1



def print_utility(w, tot_rows, tot_cols, decimal=2, flip=True):
    '''Print on terminal the utility matrix of a discrete state space
       having states defined by tuples: (0,0); (0,1); (0,2) ...

    @param w the weights vector
    @param tot_rows total number of rows
    @param tot_cols total number of columns
    @param decimal is the precision of the printing (default: 2 decimal places)
    @param flip boolean which defines if vertical flip is applied (default: True)
    '''
    utility_matrix = np.zeros((tot_rows, tot_cols))
    for row in range(tot_rows):
        for col in range(tot_cols):
            x = np.ones(w.shape[0])
            x[0] = row
            x[1] = col
            utility_matrix[row,col] = np.dot(x,w)
    np.set_printoptions(precision=decimal) #set print precision of numpy
    if flip:
        print(np.flipud(utility_matrix))
    else:
        print(utility_matrix)
    np.set_printoptions(precision=8) #reset to default


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

def plot_3d(w_list, bool_op_list, world_size, filename="figure.png"):
        #Define the main figure property
        fig, ax_array = plt.subplots(nrows=1, ncols=len(w_list), subplot_kw={'projection': '3d', 'autoscale_on':False, 'aspect':'equal'})
        #x += 0.5
        #y += 0.5
        #Iteration on all the subplots
        counter = 0
        for ax in ax_array:
            w = w_list[counter]
            x, y = np.meshgrid(np.arange(0.5, world_size-0.5, 0.5), np.arange(0.5, world_size-0.5, 0.5))
            if w.shape[0] == 3:
                z = w[0]*x + w[1]*y + w[2]
            elif w.shape[0] == 2:
                z = w[0]*x + w[1]*y
            else:
                raise ValueError('[BOOLEAN WORLDS][ERROR] The weight vector has a wrong shape')
            ax.clear()
            #_add_rectangles(ax, bool_op=bool_op_list[counter])
            bool_op = bool_op_list[counter]
            if bool_op == 'AND':
                color_00 = "red"
                color_11 = "green"
                color_10 = "red"
                color_01 = "red"
            elif bool_op == 'NAND':
                color_00 = "green"
                color_11 = "red"
                color_10 = "green"
                color_01 = "green"
            elif bool_op == 'OR':
                color_00 = "red"
                color_11 = "green"
                color_10 = "green"
                color_01 = "green"
            elif bool_op == 'XOR':
                color_00 = "red"
                color_11 = "red"
                color_10 = "green"
                color_01 = "green"
            else:
                color_00 = "red"
                color_11 = "red"
                color_10 = "red"
                color_01 = "red"
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the subplot properties
            #ax.tick_params(labelsize=10)
            ax.set_xticks(np.arange(0, world_size+1, 1))
            ax.set_xticklabels('', fontsize=10)
            ax.set_yticklabels('', fontsize=10)
            ax.set_yticks(np.arange(0, world_size+1, 1))
            ax.set_zlim(-1.0,1.0)
            ax.set_zticklabels(['-1.0','','0','','1.0'], fontsize=10)
            ax.view_init(elev=30, azim=-115)
            ax.plot_surface(x,y,z, color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax.plot_surface(x,y,z, color='white', alpha=0.01)
            counter += 1
        #Save the figure
        fig.tight_layout()
        fig.savefig(filename, dpi=300) #, bbox_inches='tight')



def main():

    use_bias = True
    env_and, w_and = init_and(bias=use_bias)
    env_nand, w_nand = init_nand(bias=use_bias)
    env_or, w_or = init_or(bias=use_bias)
    env_xor, w_xor = init_xor(bias=use_bias)

    mse_and_list = list()
    mse_nand_list = list()
    mse_or_list = list()
    mse_xor_list = list()

    gamma = 0.9
    alpha_start = 0.001
    alpha_stop = 0.000001 #constant step size
    tot_epoch = 30001 #30k epochs
    alpha_array = np.linspace(alpha_start, alpha_stop, tot_epoch)
    print_epoch = 1000

    for epoch in range(tot_epoch):

        alpha = alpha_array[epoch]

        #AND-world episode
        observation = env_and.reset(exploring_starts=True)
        if use_bias:
            x = np.array(observation+[1])
        else:
            x = np.array(observation)
        mse_and = 0.0
        for step_and in range(1000):
            action = np.random.randint(0,4)
            new_observation, reward, done = env_and.step(action)
            if use_bias:
                x_t1 = np.array(new_observation+[1])
            else:
                x_t1 = np.array(new_observation)
            w_and = update(w_and, x, x_t1, reward, alpha, gamma, done)
            #Estimate the MSE for creating a plot later
            mse_and += (np.dot(x_t1,w_and) - np.dot(x,w_and))**2
            x = x_t1
            if done: break
        mse_and /= step_and + 0.000000001
        if step_and != 0: mse_and_list.append(mse_and)

        #NAND-world episode
        observation = env_nand.reset(exploring_starts=True)
        if use_bias:
            x = np.array(observation+[1])
        else:
            x = np.array(observation)
        mse_nand = 0.0
        for step_nand in range(1000):
            action = np.random.randint(0,4)
            new_observation, reward, done = env_nand.step(action)
            if use_bias:
                x_t1 = np.array(new_observation+[1])
            else:
                x_t1 = np.array(new_observation)
            w_nand = update(w_nand, x, x_t1, reward, alpha, gamma, done)
            #Estimate the MSE for creating a plot later
            mse_nand += (np.dot(x_t1,w_nand) - np.dot(x,w_nand))**2
            x = x_t1
            if done: break
        mse_nand /= step_nand + 0.000000001
        if step_nand != 0: mse_nand_list.append(mse_nand)

        #OR-world episode
        observation = env_or.reset(exploring_starts=True)
        if use_bias:
            x = np.array(observation+[1])
        else:
            x = np.array(observation)
        mse_or = 0.0
        for step_or in range(1000):
            action = np.random.randint(0,4)
            new_observation, reward, done = env_or.step(action)
            if use_bias:
                x_t1 = np.array(new_observation+[1])
            else:
                x_t1 = np.array(new_observation)
            w_or = update(w_or, x, x_t1, reward, alpha, gamma, done)
            #Estimate the MSE for creating a plot later
            mse_or += (np.dot(x_t1,w_or) - np.dot(x,w_or))**2
            x = x_t1
            if done: break
        mse_or /= step_or + 0.000000001
        if step_or != 0: mse_or_list.append(mse_or)

        #XOR-world episode
        observation = env_xor.reset(exploring_starts=True)
        if use_bias:
            x = np.array(observation+[1])
        else:
            x = np.array(observation)
        mse_xor = 0.0
        for step_xor in range(1000):
            action = np.random.randint(0,4)
            new_observation, reward, done = env_xor.step(action)
            if use_bias:
                x_t1 = np.array(new_observation+[1])
            else:
                x_t1 = np.array(new_observation)
            w_xor = update(w_xor, x, x_t1, reward, alpha, gamma, done)
            #Estimate the MSE for creating a plot later
            mse_xor += (np.dot(x_t1,w_xor) - np.dot(x,w_xor))**2
            x = x_t1
            if done: break
        mse_xor /= step_xor + 0.000000001
        if step_xor != 0: mse_xor_list.append(mse_xor)

        if(epoch % print_epoch == 0):
            print("")
            print("Epoch: " + str(epoch+1))
            print("Alpha: " + str(alpha))
            print("------AND-world------")
            print("Tot steps: " + str(step_and))
            print("MSE: " + str(mse_and))
            print("w: " + str(w_and))
            print_utility(w_and, tot_rows=5, tot_cols=5)
            print("------NAND-world------")
            print("Tot steps: " + str(step_nand))
            print("MSE: " + str(mse_nand))
            print("w: " + str(w_nand))
            print_utility(w_nand, tot_rows=5, tot_cols=5)
            print("------OR-world------")
            print("Tot steps: " + str(step_or))
            print("MSE: " + str(mse_or))
            print("w: " + str(w_or))
            print_utility(w_or, tot_rows=5, tot_cols=5)
            print("------XOR-world------")
            print("Tot steps: " + str(step_xor))
            print("MSE: " + str(mse_xor))
            print("w: " + str(w_xor))
            print_utility(w_xor, tot_rows=5, tot_cols=5)

    print("Generating plot, please wait...")
    plot_3d([w_and, w_nand, w_or, w_xor], ["AND", "NAND", "OR", "XOR"], world_size=5, filename="boolean_planes.png")
    plot_curve(mse_and_list, filepath="./mse_and_plot.png", x_label="iterations", y_label="MSE", 
               x_range=(0, len(mse_and_list)), y_range=(0,0.5), color="-r", kernel_size=50, alpha=0.4, grid=True)
    plot_curve(mse_nand_list, filepath="./mse_nand_plot.png", x_label="iterations", y_label="MSE", 
               x_range=(0, len(mse_nand_list)), y_range=(0,0.5), color="-r", kernel_size=50, alpha=0.4, grid=True)
    plot_curve(mse_or_list, filepath="./mse_or_plot.png", x_label="iterations", y_label="MSE", 
               x_range=(0, len(mse_or_list)), y_range=(0,0.5), color="-r", kernel_size=50, alpha=0.4, grid=True)
    plot_curve(mse_xor_list, filepath="./mse_xor_plot.png", x_label="iterations", y_label="MSE", 
               x_range=(0, len(mse_xor_list)), y_range=(0,0.5), color="-r", kernel_size=50, alpha=0.4, grid=True)
    print("Done!")

if __name__ == "__main__":
    main()
