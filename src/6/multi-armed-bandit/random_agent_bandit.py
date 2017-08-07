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

from multi_armed_bandit import MultiArmedBandit
import numpy as np

my_bandit = MultiArmedBandit(reward_probability_list=[0.3, 0.5, 0.8])
tot_arms = 3
tot_episodes = 2000
tot_steps = 1000
print_every_episodes = 100
cumulated_reward_list = list()
print_every_episodes = 100
print("Starting random agent...")
for episode in range(tot_episodes):
    cumulated_reward = 0
    for step in range(tot_steps):
        action = np.random.randint(low=0, high=tot_arms)
        reward = my_bandit.step(action)
        cumulated_reward += reward
    cumulated_reward_list.append(cumulated_reward)
    if episode % print_every_episodes == 0:
            print("Episode: " + str(episode))
            print("Cumulated Reward: " + str(cumulated_reward))
            print("")
print("Average Cumulated Reward: " + str(np.mean(cumulated_reward_list)))
print("Std Cumulated Reward: " + str(np.std(cumulated_reward_list)))
