#!/usr/bin/env python

# MIT License
# Copyright (c) 2019 Massimiliano Patacchiola
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
import glob
import os 
from PIL import Image
import collections
       
class DroneVisualLanding:
    def __init__(self, world_size=11, tile_size=64, patch_folder_path=None, landmark_folder_path=None):
        """ Create a new DroneVisualLanding environment.
        
        This environment is made of a ground floor built with image patches,
        and a landmark. The goal is to move the quadcopter on top of the landmark.
        After each action the sequence of last 4 images (84x84 grayscale) is
        returned.
        @param world_size is the size of the world (in meters), it defines
            a cube representing the horizon of the world. It must be a odd positive integer.
        @param patch_folder_path path to the folder containing square images 
            of the that floor you want to use during the simulation.
        @param landmark_folder_path path to the folder containing square
            images of the landmark you want to use during the simulation.
        @param tile_size is the resolution of a single patch, all images are downscaled or
            upscaled to fit this size.
        """
        if(world_size % 2 == 0 or world_size<3):
            raise ValueError("[ERROR] drone_visual_landing.py world_size must be an odd positive integer >=3.")
        if(tile_size<8):
            raise ValueError("[ERROR] drone_visual_landing.py tile_size must be a positive integer >=8.")
        self.actions_dict = {0:"forward", 1:"left", 2:"backward", 3:"right", 4:"up", 
                             5:"down", 6:"rotate-cw", 7:"rotate-ccw"}
        self.position_list = list()
        self.world_size = world_size
        self.tile_size = tile_size
        self.position_x = np.random.randint(low=0, high=world_size)
        self.position_y = np.random.randint(low=0, high=world_size)
        self.position_z = np.random.randint(low=0, high=world_size)
        self.position_r = np.random.randint(low=0, high=4)
        self.center_x = int(world_size / 2)
        self.center_y = int(world_size / 2)
        self.center_z = 0
        self.patch_list = self._return_file_list(patch_folder_path)
        self.landmark_list = self._return_file_list(landmark_folder_path)
        self._update_floor()

    def _return_file_list(self, path):
        """ It checks if the images in the folder have the right size and shape,
            then return the file list.
        
        @param path to the folder to check
        @return: it returns the list of available images
        """      
        path = str(path)
        if not os.path.isdir(path):
            raise ValueError("[ERROR] drone_visual_landing.py the path specified does not exist!")
        file_list = glob.glob(path + "/*.jpg")
        file_list += glob.glob(path + "/*.jpeg")
        file_list += glob.glob(path + "/*.png")
        file_list += glob.glob(path + "/*.JPG")
        file_list += glob.glob(path + "/*.JPEG")
        file_list += glob.glob(path + "/*.PNG")
        if(len(file_list)==0): 
            raise ValueError("[ERROR] drone_visual_landing.py there are no valid images in: " + path)
        else:
            for file_path in file_list:
                img = Image.open(file_path)
                w, h = img.size
                if(w != h): raise ValueError("[ERROR] drone_visual_landing.py non-square image shape: " + file_path)
            return file_list        

    def reset(self, exploring_starts=True, initial_position=[0,0,0,0]):
        """ It reset the car to an initial position [-1.2, 0.5]
        
        @param exploring_starts: if True a random position is taken
        @param initial_position: the initial position of the drone (requires exploring_starts=False)
        @return: it returns the initial position [x,y,z,r] of the drone (r=yaw)
        """
        if exploring_starts:
            self.position_x = np.random.randint(low=0, high=self.world_size)
            self.position_y = np.random.randint(low=0, high=self.world_size)
            self.position_z = np.random.randint(low=0, high=self.world_size)
            self.position_r = np.random.randint(low=0, high=4)
        else:
            self.position_x = initial_position[0]
            self.position_y = initial_position[1]
            self.position_z = initial_position[2]
            self.position_r = initial_position[3]
        
        self.position_list = []  # clear the list
        self.position_list.append(initial_position)
        
        self._update_floor()
        state_list = self._get_state_list()
        return state_list

    def _update_floor(self):
        """It generates the floor randomly picking a patch and a landmark
        
        """ 
        #get a random patch
        idx = np.random.randint(0, len(self.patch_list))
        patch_path = self.patch_list[idx]
        img = Image.open(patch_path)#.convert('L')
        img = img.resize([self.tile_size, self.tile_size], Image.ANTIALIAS)
        patch = np.array(img, dtype=np.uint8)[:,:,0]
        #patch = np.mean(patch, axis=2)
        #get a random landmark
        idx = np.random.randint(0, len(self.landmark_list))
        landmark_path = self.landmark_list[idx]
        img = Image.open(landmark_path)#.convert('L')       
        img = img.resize([self.tile_size, self.tile_size], Image.ANTIALIAS)
        landmark = np.array(img, dtype=np.uint8)[:,:,0]
        #landmark = np.mean(landmark, axis=2)
        #generate the floor
        side = ((self.world_size*2)+1)
        floor_row = np.repeat(patch, repeats=side, axis=1)
        self.floor = np.repeat(floor_row, repeats=side, axis=0)
        print(self.floor.shape)
        start_row = int(self.floor.shape[0]/2.0)-int(self.tile_size/2.0)
        end_row = int(self.floor.shape[0]/2.0)+int(self.tile_size/2.0)
        start_col = int(self.floor.shape[1]/2.0)-int(self.tile_size/2.0)
        end_col = int(self.floor.shape[1]/2.0)+int(self.tile_size/2.0)
        self.floor[start_row:end_row, start_col:end_col] = landmark
        
    def _get_state_list(self):
        """Get a stack of four images representing the current state
        
        @return: image list
        """
        #estimate the view submatrix based on current position
        fov = int(self.position_z * self.tile_size)
        if(fov==0): fov=1
        center_row = (self.position_y * self.tile_size) + int(self.world_size/2)
        start_row = center_row - (self.tile_size * fov)
        end_row = center_row + (self.tile_size * fov)
        center_col = (self.position_x * self.tile_size) + int(self.world_size/2)
        start_col = center_col - (self.tile_size * fov)
        end_col = center_col + (self.tile_size * fov)
        #get the state stack
        state = self.floor[start_row:end_row, start_col:end_col].copy()
        #rotate the patch based on the rotation parameter
        if(self.position_r!=0): state = np.rot90(state, k=self.position_r)
        #Resize the image if size is wrong
        img = Image.fromarray(state)
        img = img.resize([84, 84], Image.ANTIALIAS)        
        #Cast to float and normalize between zero an one
        state = np.array(img, dtype=np.float32) / 255.0
        #stack of last four images, since this is the starting state,
        #the four images are the same.
        try:
            self.state
        except AttributeError:
            self.state = collections.deque([state, state, state, state], maxlen=4)
        else:
            self.state.append(state)          
        return list(self.state)       

    def step(self, action):
        """Perform one step in the environment following the action.
        
        @param action: an integer representing one of three actions [0, 1, 2, 3, 4, 5, 6, 7]
         where 0=forward, 1=left, 2=backward, 3=right, 4=up, 5=down, 6=rotate-cw, 7=rotate-ccw
        @return: [frame_0, frame_1, frame_2, frame_3], reward, done
         the list contains the last four images (84x84) seen by the drone
         the reward is always negative but when the goal is reached
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
        elif action == 6:
            self.position_r += 1  # ROTATE-CW
        elif action ==7:
            self.position_r -= 1  # ROTATE-CCW
        else:
            raise ValueError("[ERROR] drone_visual_landing.py The action value " + str(action) +
                             " is out of range. Available actions are: 0, 1, 2, 3, 4, 5, 6, 7")

        # Check if the drone is outside the bounding box
        # and if the rotation is inside the limits.
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
        if self.position_r == 4:
            self.position_r = 0
        elif self.position_r == -1:
            self.position_r = 3 
            
        # append the position in the list
        self.position_list.append([self.position_x, self.position_y, self.position_z, self.position_r])

        # The drone landed!!!
        if self.position_x == self.center_x and self.position_y == self.center_y and self.position_z == self.center_z:
            reward = 1.0
            done = True
            return self._get_state_list(), reward, done

        # Drone landed outside the pad.
        if self.position_z == self.center_z:
            reward = -1.0
            done = True
            return self._get_state_list(), reward, done        

        # Nothing special happened
        reward = -0.01
        done = False
        return self._get_state_list(), reward, done

    def render(self, file_path='./drone.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.
        
        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        if(len(self.position_list) <=1): return
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
