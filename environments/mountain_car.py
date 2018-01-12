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

# Implentation of the Mountain Car problem using the notation of the book:
# 'Statistical Reinforcement Learning' by Masashi Sugiyama

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MountainCar:
    def __init__(self, mass=0.2, friction=0.3, delta_t=0.1):
        """ Create a new mountain car object.
        
        It is possible to pass the parameter of the simulation.
        @param mass: the mass of the car (default 0.2) 
        @param friction:  the friction in Newton (default 0.3)
        @param delta_t: the time step in seconds (default 0.1)
        """
        self.position_list = list()
        self.gravity = 9.8
        self.friction = friction
        self.delta_t = delta_t  # second
        self.mass = mass  # the mass of the car
        self.position_t = -0.5
        self.velocity_t = 0.0

    def reset(self, exploring_starts=True, initial_position=-0.5):
        """ It reset the car to an initial position [-1.2, 0.5]
        
        @param exploring_starts: if True a random position is taken
        @param initial_position: the initial position of the car (requires exploring_starts=False)
        @return: it returns the initial position of the car and the velocity
        """
        if exploring_starts:
            initial_position = np.random.uniform(-1.2,0.5)
        if initial_position < -1.2:
            initial_position = -1.2
        if initial_position > 0.5:
            initial_position = 0.5
        self.position_list = []  # clear the list
        self.position_t = initial_position
        self.velocity_t = 0.0
        self.position_list.append(initial_position)
        return [self.position_t, self.velocity_t]

    def step(self, action):
        """Perform one step in the environment following the action.
        
        @param action: an integer representing one of three actions [0, 1, 2]
         where 0=move_left, 1=do_not_move, 2=move_right
        @return: (postion_t1, velocity_t1), reward, done
         where reward is always negative but when the goal is reached
         done is True when the goal is reached
        """
        if(action >= 3):
            raise ValueError("[MOUNTAIN CAR][ERROR] The action value "
                             + str(action) + " is out of range.")
        done = False
        reward = -0.01
        action_list = [-0.2, 0, +0.2]
        action_t = action_list[action]
        velocity_t1 = self.velocity_t + \
                      (-self.gravity * self.mass * np.cos(3*self.position_t)
                       + (action_t/self.mass)
                       - (self.friction*self.velocity_t)) * self.delta_t
        position_t1 = self.position_t + (velocity_t1 * self.delta_t)
        # Check the limit condition (car outside frame)
        if position_t1 < -1.2:
            position_t1 = -1.2
            velocity_t1 = 0
        # Assign the new position and velocity
        self.position_t = position_t1
        self.velocity_t= velocity_t1
        self.position_list.append(position_t1)
        # Reward and done when the car reaches the goal
        if position_t1 >= 0.5:
            reward = +1.0
            done = True
        # Return state_t1, reward, done
        return [position_t1, velocity_t1], reward, done

    def render(self, file_path='./mountain_car.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.
        
        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Plot init
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
        ax.grid(False)  # disable the grid
        x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
        y_sin = np.sin(3 * x_sin)
        # plt.plot(x, y)
        ax.plot(x_sin, y_sin)  # plot the sine wave
        # line, _ = ax.plot(x, y, 'o-', lw=2)
        dot, = ax.plot([], [], 'ro')
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        _position_list = self.position_list
        _delta_t = self.delta_t

        def _init():
            dot.set_data([], [])
            time_text.set_text('')
            return dot, time_text

        def _animate(i):
            x = _position_list[i]
            y = np.sin(3 * x)
            dot.set_data(x, y)
            time_text.set_text("Time: " + str(np.round(i*_delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return dot, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.position_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1/self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1/self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)
