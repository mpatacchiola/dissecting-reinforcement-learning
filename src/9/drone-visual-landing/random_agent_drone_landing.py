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

from drone_visual_landing import DroneVisualLanding
import numpy as np
import matplotlib.pyplot as plt

my_drone = DroneVisualLanding(world_size=11, tile_size=64, patch_folder_path="./etc/patch", landmark_folder_path="./etc/landmark")
cumulated_reward = 0
print("Starting random agent...")
for step in range(50):
    action = np.random.randint(low=0, high=8)
    observation, reward, done = my_drone.step(action)
    print("x-y-z-r: " + str(my_drone.position_x) + "-" + str(my_drone.position_y) 
                      + "-" + str(my_drone.position_z) + "-" + str(my_drone.position_r))
    print("Action: " + str(action) + " (" + my_drone.actions_dict[action] + ")")
    print("Image shape: " + str(observation[0].shape))
    print("")
    cumulated_reward += reward
    if done: break
print("Finished after: " + str(step+1) + " steps")
print("Cumulated Reward: " + str(cumulated_reward))
print("Rendering GIF, please wait...")
my_drone.render(file_path='./drone_landing.gif', mode='gif')
for i in range(len(observation)): observation[i] = np.pad(observation[i], ((3, 3), (3, 3)), 'constant', constant_values=0)
img = (np.concatenate(observation, axis=1)*255.0).astype(np.uint8)
plt.title("Last observation (4 images)")
imgplot = plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.show()
plt.title("World")
imgplot = plt.imshow(my_drone.floor, cmap='gray',vmin=0,vmax=255)
plt.show()
print("Complete!")
