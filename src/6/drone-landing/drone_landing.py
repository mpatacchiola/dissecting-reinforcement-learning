#!/usr/bin/env python

# MIT License
# Copyright (c) 2017 Massimiliano Patacchiola
# https://mpatacchiola.github.io/blog/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Implentation of the Drone Landing problem.
# A drone has to land on a ground platform moving in a 3-dimensional space.
# The space is discretized in cells. The marker is on the ground, in the center
# of a cubic room. In order to obtain a positive reward the drone has to land
# on the marker. 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d

class DroneLanding:
    def __init__(self, world_size=11):
        """ Create a new drone-landing environment.
        
        It is possible to pass the parameter of the simulation.
        @param world_size is the size of the world (in meters),
            it must be a positive integer.
        """
        self.position_list = list()
        self.world_size = world_size
        self.position_x = np.random.randint(low=0, high=world_size)
        self.position_y = np.random.randint(low=0, high=world_size)
        self.position_z = np.random.randint(low=0, high=world_size)
        self.center_x = int(world_size / 2)
        self.center_y = int(world_size / 2)
        self.center_z = 0


    def reset(self, exploring_starts=True, initial_position=[0,0,0]):
        """ It reset the car to an initial position [-1.2, 0.5]
        
        @param exploring_starts: if True a random position is taken
        @param initial_position: the initial position of the drone (requires exploring_starts=False)
        @return: it returns the initial position [x,y,z] of the drone 
        """
        if exploring_starts:
            self.position_x = np.random.randint(low=0, high=self.world_size)
            self.position_y = np.random.randint(low=0, high=self.world_size)
            self.position_z = np.random.randint(low=0, high=self.world_size)
        else:
            self.position_x = initial_position[0]
            self.position_y = initial_position[1]
            self.position_z = initial_position[2]            
        
        self.position_list = []  # clear the list
        self.position_list.append(initial_position)
        return [self.position_x, self.position_y, self.position_z]

    def step(self, action):
        """Perform one step in the environment following the action.
        
        @param action: an integer representing one of three actions [0, 1, 2, 3, 4, 5]
         where 0=forward, 1=left, 2=backward, 3=right, 4=up, 5=down
        @return: [x,y,z], reward, done
         where reward is always negative but when the goal is reached
         done is True when the goal is reached
        """      
        if action == 0:
            self.position_x += 1  # FORWARD
        elif action == 1:
            self.position_y += 1  # LEFT
        elif action == 2:
            self.position_x -= 1  # BACKWARD
        elif action == 3:
            self.position_y -= 1  # RIGHT
        elif action == 4:
            self.position_z += 1  # UP
        elif action ==5:
            self.position_z -= 1  # DOWN
        else:
            raise ValueError("[DRONE LANDING][ERROR] The action value "
                             + str(action) + " is out of range.")

        # Check if the drone is outside the bounding box
        if self.position_x >= self.world_size:
            self.position_x = self.world_size - 1
        elif self.position_x < 0:
            self.position_x = 0
        if self.position_y >= self.world_size:
            self.position_y = self.world_size - 1
        elif self.position_y < 0:
            self.position_y = 0
        if self.position_z >= self.world_size:
            self.position_z = self.world_size - 1
        elif self.position_z < 0:
            self.position_z = 0            

        # append the position in the list
        self.position_list.append([self.position_x, self.position_y, self.position_z])

        # The drone landed!!!
        if self.position_x == self.center_x and self.position_y == self.center_y and self.position_z == self.center_z:
            reward = 1.0
            done = True
            return [self.position_x, self.position_y, self.position_z], reward, done

        # Drone landed outside the pad.
        if self.position_z == self.center_z:
            reward = -1.0
            done = True
            return [self.position_x, self.position_y, self.position_z], reward, done        

        # Nothing special happened
        reward = -0.01
        done = False
        return [self.position_x, self.position_y, self.position_z], reward, done

    def render(self, file_path='./drone.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.
        
        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Internal function used to plot animation
        def _animate(i):
            x = self.position_list[i][0]
            y = self.position_list[i][1]
            z = self.position_list[i][2]
            #print x,y,z,num
            graph._offsets3d = ([x], [y], [z])
            title.set_text('Time={}s'.format(i))
        # Define the figure and the plot (with axis limits)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', autoscale_on=False, 
                             xlim=(-1, self.world_size+1), ylim=(-1, self.world_size+1), zlim=(0, self.world_size))
        # Set the tick label
        if self.world_size < 15:
            ax.set_xticklabels(np.arange(0, self.world_size+1))
            ax.set_yticklabels(np.arange(0, self.world_size+1))
        else:
            #ax.set_xticklabels([])
            step = int(self.world_size / 10.0)
            label_list = list()
            for i in range(self.world_size):
                if i % step == 0:
                    label_list.append(str(i))
                else:
                    label_list.append('')
            ax.set_yticklabels(label_list)
            ax.set_xticklabels(label_list)
            ax.set_zticklabels(label_list)
           
        # Set the ticks   
        ax.set_xticks(np.arange(0,self.world_size+1)) 
        ax.set_yticks(np.arange(0,self.world_size+1))
        ax.set_zticks(np.arange(0,self.world_size)) 
        # Set title and title location
        title = ax.set_title('', loc='left')
        # Draw the red rectangle
        p = Rectangle((0, 0), self.world_size, self.world_size, color="red", alpha=0.5)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        # Draw the green rectangle
        p = Rectangle((self.center_x, self.center_y), 1, 1, color="green")
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0.1, zdir="z")
        # Get the starting position
        x = self.position_list[0][0]
        y = self.position_list[0][1]
        z = self.position_list[0][2]
        graph = ax.scatter([x], [y], [z], c='r', marker='o')
        # Define the animation
        ani = animation.FuncAnimation(fig, _animate, frames=np.arange(1, len(self.position_list)), blit=False, repeat=False)
        # Saving in gif or video
        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=5)
        elif mode == 'mp4':
            ani.save(file_path, fps=5, writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)
