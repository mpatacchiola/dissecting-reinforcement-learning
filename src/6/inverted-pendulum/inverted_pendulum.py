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

# Implentation of the Inverted Pendulum problem using the notation of the book:
# 'Statistical Reinforcement Learning' by Masashi Sugiyama

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class InvertedPendulum:
    def __init__(self, pole_mass=2.0, cart_mass=8.0, pole_lenght=0.5, delta_t=0.1):
        """ Create a new pendulum object.
        
        It is possible to pass the parameter of the simulation.
        @param pole_mass: the mass of the pole (default 2.0 Kg) 
        @param cart_mass:  the mass of the cart (default 8.0 Kg)
        @param pole_lenght: the lenght of the pole (default 0.5 m)
        @param delta_t: the time step in seconds (default 0.1 s)
        """
        self.angle_list = list()
        self.gravity = 9.8
        self.delta_t = delta_t
        self.pole_mass = pole_mass
        self.cart_mass = cart_mass
        self.pole_lenght = pole_lenght
        self.angle_t = 0.0 + np.random.normal(0, 0.1) # radians (vertical position)
        self.angular_velocity_t = 0.0
        self.alpha = 1.0 / (self.pole_mass + self.cart_mass)

    def reset(self, exploring_starts=True, initial_angle=0.0):
        """ It reset the pendulum to an initial position [0, 2*pi]
        
        @param exploring_starts: if True a random position is taken
        @param initial_angle: the initial position of the pendulum (requires exploring_starts=False)
        @return: it returns the initial position of the pendulum and the velocity
        """
        if exploring_starts:
            initial_angle = np.random.uniform(0, 2*np.pi)
        #if initial_angle < -np.pi:
        #    initial_angle = -np.pi
        #elif initial_angle > np.pi:
        #    initial_angle = np.pi
        #else:
        small_noise = np.random.normal(0, 0.1)  # adding small gaussian noise
        if small_noise >=0: 
            initial_angle = small_noise
        else: 
            initial_angle = (2*np.pi) + small_noise
        self.angle_list = []  # clear the list
        self.angle_t = initial_angle
        self.angular_velocity_t = 0.0
        self.angle_list.append(initial_angle)
        return [self.angle_t, self.angular_velocity_t]

    def step(self, action):
        """Perform one step in the environment following the action.
        
        @param action: an integer representing one of three actions [0, 1, 2]
         where 0=move_left, 1=do_not_move, 2=move_right
        @return: (angle_t1, angular_velocity_t1), reward, done
         where reward is 0.0 when the pole is horizontal and 1.0 if vertical
         done is True when the goal is reached
        """
        if(action >= 3):
            raise ValueError("[INVERTED PENDULUM][ERROR] The action value "
                             + str(action) + " is out of range.")
        done = False
        reward = -0.01
        action_list = [-50, 0, +50]
        action_t = action_list[action]
        angular_velocity_t1 = self.angular_velocity_t + \
                              (self.gravity * np.sin(self.angle_t) - \
                              self.alpha * self.pole_mass * self.pole_lenght * np.power(self.angular_velocity_t, 2) * (np.sin(2*self.angular_velocity_t)/2.0) + \
                              self.alpha * np.cos(self.angle_t) * action_t) / \
                              ((4/3) * self.pole_lenght - self.alpha * self.pole_mass * self.pole_lenght * np.power(np.sin(self.angle_t), 2)) * self.delta_t

        angle_t1 = self.angle_t + (angular_velocity_t1 * self.delta_t)
        # Check the limit condition (horizontal pole)
        if angle_t1 < -(np.pi/2.0):
            angle_t1 = -(np.pi/2.0)
            angular_velocity_t1 = 0
        if angle_t1 > (np.pi/2.0):
            angle_t1 = (np.pi/2.0)
            angular_velocity_t1 = 0
        # Assign the new position and velocity
        self.angle_t = angle_t1
        self.angular_velocity_t= angular_velocity_t1
        self.angle_list.append(angle_t1)
        # Reward and done
        if angle_t1 >= (np.pi/2.0) or angle_t1 <= -(np.pi/2.0):
            reward = 0.0
            done = True
        else:
            reward = np.cos(angle_t1)
            done = False           
        # Return state_t1, reward, done
        return [angle_t1, angular_velocity_t1], reward, done

    def render(self, file_path='./inverted_pendulum.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of the steps happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        or every k episodes.
        
        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Plot init
        fig = plt.figure()
        axis_limit = self.pole_lenght + (self.pole_lenght * 0.5)
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-axis_limit, axis_limit), ylim=(0.0, 1.5*axis_limit))
        ax.grid(False)  # disable the grid
        ax.set_aspect('equal')
        ax.set_yticklabels([])
        # x_line = np.linspace(start=-axis_limit, stop=axis_limit, num=100)
        # y_line = np.zeros(100)
        # ax.plot(x_line, y_line)  # plot the base-line
        # line, _ = ax.plot(x, y, 'o-', lw=2)
        line, = ax.plot([], [],color='black', linestyle='solid', linewidth=1.5, marker='o', markerfacecolor='#aa0000', markersize=10, zorder=1)
        # Adding the brown circle pad
        circle = plt.Circle((0.0,-0.01), radius=0.05, color='#2b2200', fill=True, zorder=2)
        ax.add_patch(circle)
        # Adding the text
        time_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)
        _angle_list = self.angle_list
        _delta_t = self.delta_t

        def _init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text

        def _animate(i):
            angle_cos = np.cos(_angle_list[i]) * self.pole_lenght
            angle_sin = np.sin(_angle_list[i]) * self.pole_lenght
            x1, y1 = [0, angle_sin], [0, angle_cos]
            #y1 = (angle_cos, angle_sin)
            line.set_data(x1, y1)
            time_text.set_text("Time: " + str(np.round(i*_delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return line, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.angle_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1/self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1/self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)
