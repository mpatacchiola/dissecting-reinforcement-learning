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

#In this example I will use the class gridworld to generate a 5x5 world
#in which the cleaning robot will move. Rewards are allocated in the 4
#corners of the world following the XOR pattern. I will use an function
#approximator based on a paraboloid-like function in order to represent
#a TD(0) function approximator.

import numpy as np
from gridworld import GridWorld
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import cm
from mlp import MLP

def init_env():
    '''Init the XOR boolean environment

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
    return env


def update(my_mlp, new_observation, reward, learning_rate, gamma, done):
    '''Return the updated weights vector w_t1

    @param w the weights vector before the update
    @param x the feauture vector obsrved at t
    @param x_t1 the feauture vector observed at t+1
    @param reward the reward observed after the action
    @param gamma the discount factor
    @param done boolean True if the state is terminal
    @return w_t1 the weights vector at t+1
    '''
    if done:
        x = np.array(new_observation, dtype=np.float32)
        target = np.array([reward], dtype=np.float32) 
        #print(target)
        error = my_mlp.train(x, target, learning_rate)
    else:
        x = np.array(new_observation, dtype=np.float32)  
        target = np.array((reward + (gamma * my_mlp.forward(x))), dtype=np.float32)
        #print target
        error = my_mlp.train(x, target, learning_rate)  
    return my_mlp, error


def print_utility(my_mlp, tot_rows, tot_cols, decimal=2, flip=True):
    '''Print on terminal the utility matrix of a discrete state space
       having states defined by tuples: (0,0); (0,1); (0,2) ...

    @param my_mlp an MLP object having single output
    @param tot_rows total number of rows
    @param tot_cols total number of columns
    @param decimal is the precision of the printing (default: 2 decimal places)
    @param flip boolean which defines if vertical flip is applied (default: True)
    '''
    utility_matrix = np.zeros((tot_rows, tot_cols))
    for row in range(tot_rows):
        for col in range(tot_cols):
            x = np.array([row, col], dtype=np.float32)
            utility_matrix[row,col] = my_mlp.forward(x)
    np.set_printoptions(precision=decimal) #set print precision of numpy
    if flip:
        print(np.flipud(utility_matrix))
    else:
        print(utility_matrix)
    np.set_printoptions(precision=8) #reset to default


def subplot(my_mlp, world_size, filename="figure.png"):
            #Define the main figure property
            fig, ax = plt.subplots(nrows=1, ncols=4, subplot_kw={'projection': '3d', 'autoscale_on':False, 'aspect':'equal'})
            #XOR color
            color_00 = "red"
            color_11 = "red"
            color_10 = "green"
            color_01 = "green"

            #Quadratic subplot
            ax[0].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[0].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[0].set_xticks(np.arange(0, world_size+1, 1))
            ax[0].set_xticklabels('', fontsize=10)
            ax[0].set_yticklabels('', fontsize=10)
            ax[0].set_yticks(np.arange(0, world_size+1, 1))
            ax[0].set_zlim(-1.0,1.0)
            #ax[0].set_zticklabels(['-1.0','','0','','1.0'], fontsize=10)
            ax[0].view_init(elev=30, azim=-115)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))            
            grid = np.arange(0.0, world_size-1.0, 0.01)    
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[0].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[0].plot_surface(x,y,z, color='white', alpha=0.01)

            #Quadratic subplot
            ax[1].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[1].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[1].set_xticks(np.arange(0, world_size+1, 1))
            ax[1].set_xticklabels('', fontsize=10)
            ax[1].set_yticklabels('', fontsize=10)
            ax[1].set_yticks(np.arange(0, world_size+1, 1))
            ax[1].set_zlim(-1.0,1.0)
            ax[1].set_zticklabels([''], fontsize=10)
            ax[1].view_init(elev=30, azim=-65)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))
            grid = np.arange(0.0, world_size-1.0, 0.01)           
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[1].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[1].plot_surface(x,y,z, color='white', alpha=0.01)

            #Quadratic subplot
            ax[2].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[2].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[2].set_xticks(np.arange(0, world_size+1, 1))
            ax[2].set_xticklabels('', fontsize=10)
            ax[2].set_yticklabels('', fontsize=10)
            ax[2].set_yticks(np.arange(0, world_size+1, 1))
            ax[2].set_zlim(-1.0,1.0)
            ax[2].set_zticklabels([''], fontsize=10)
            ax[2].view_init(elev=30, azim=-45)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))
            grid = np.arange(0.0, world_size-1.0, 0.01)           
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[2].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[2].plot_surface(x,y,z, color='white', alpha=0.01)

            #Quadratic subplot
            ax[3].clear()
            #Draw the rectangles
            p = Rectangle((0, 0), 1, 1, color=color_00, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, world_size-1), 1, 1, color=color_11, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((0, world_size-1), 1, 1, color=color_01, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            p = Rectangle((world_size-1, 0), 1, 1, color=color_10, alpha=0.5)
            ax[3].add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=-1.0, zdir="z")
            #Set the plot
            ax[3].set_xticks(np.arange(0, world_size+1, 1))
            ax[3].set_xticklabels('', fontsize=10)
            ax[3].set_yticklabels('', fontsize=10)
            ax[3].set_yticks(np.arange(0, world_size+1, 1))
            ax[3].set_zlim(-1.0,1.0)
            ax[3].set_zticklabels([''], fontsize=10)
            ax[3].view_init(elev=30, azim=-25)
            x, y = np.meshgrid(np.arange(0.0, world_size-1.0, 0.01), np.arange(0.0, world_size-1.0, 0.01))
            grid = np.arange(0.0, world_size-1.0, 0.01)           
            z_matrix = list()          
            for x_i in grid:
                z_row  = list()
                for y_i in grid:
                    z_row.append(my_mlp.forward(np.array([x_i, y_i])))
                z_matrix.append(z_row)
            z = np.squeeze(np.array(z_matrix))
            ax[3].plot_surface(x+0.5,y+0.5,z, color='lightgrey', alpha=0.5, linewidth=0, antialiased=False) # color='lightgrey', alpha=0.5)
            #Draw a White background
            x, y = np.meshgrid(np.arange(0, world_size+1, 1), np.arange(0, world_size+1, 1))
            z = x*(-1.0)
            ax[3].plot_surface(x,y,z, color='white', alpha=0.01)

            #Save the figure
            fig.tight_layout()
            fig.savefig(filename, dpi=300) #, bbox_inches='tight')

def main():

    env = init_env()
    my_mlp = MLP(tot_inputs=2, tot_hidden=2, tot_outputs=1, activation="tanh")
    learning_rate = 0.1
    gamma = 0.9
    tot_epoch = 10001
    print_epoch = 100

    for epoch in range(tot_epoch):
        #XOR-world episode
        observation = env.reset(exploring_starts=True)
        #The episode starts here
        for step in range(1000):
            action = np.random.randint(0,4)
            new_observation, reward, done = env.step(action) #move in the world and get the state and reward
            my_mlp, error = update(my_mlp, new_observation, reward, learning_rate, gamma, done)
            observation = new_observation
            if done: break
        if(epoch % print_epoch == 0 and epoch!=0):
            print("")
            print("Epoch: " + str(epoch+1))
            print("Tot steps: " + str(step))
            print("Error: " + str(error))

            print_utility(my_mlp, tot_rows=5, tot_cols=5)
    print("Generating plot, please wait...")
    subplot(my_mlp, world_size=5, filename="xor_planes.png")
    print("Done!")

if __name__ == "__main__":
    main()
