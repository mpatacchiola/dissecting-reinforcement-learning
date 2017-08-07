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

# Implentation of the multi-armed bandit environment
# The rewards associated with each arm are modeled by
# Bernoulli distributions return 1 or 0.

import numpy as np

class MultiArmedBandit:
    def __init__(self, reward_probability_list=[0.3, 0.5, 0.8]):
        """ Create a new bandit environment.
        
        It is possible to pass the parameter of the simulation.
        @param: reward_probability_list: each value correspond to
            probaility [0, 1] of obtaining a positive reward of +1.
            For each value in the list an arm is defined.
            e.g. [0.3, 0.5, 0.2, 0.8] defines 4 arms, the last one
            having higher probability (0.8) of returnig a reward. 
        """
        self.reward_probability_list = reward_probability_list


    def step(self, action):
        """Pull the arm indicated in the 'action' parameter.
        
        @param: action an integer representing the arm to pull.        
        @return: reward it returns the reward obtained pulling that arm
        """
        if action > len(self.reward_probability_list):
            raise Exception("MULTI ARMED BANDIT][ERROR] the action" + str(action) + " is out of range, total actions: " + str(len(self.reward_probability_list)))
        p = self.reward_probability_list[action]
        q = 1.0-p
        return np.random.choice(2, p=[q, p])


