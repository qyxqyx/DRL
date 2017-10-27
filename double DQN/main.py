"""
Double DQN

Using:
mxnet 0.11.0
gym 0.8.0
"""


import numpy as np
import gym
import mxnet as mx
from double_dqn import double_DQN
from pylab import *


np.random.seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 70
MAX_EP_STEPS = 400


RENDER = False
ENV_NAME = 'Pendulum-v0'
ACTION_SPACE = 21

net_hidden_num    = [10]
epislon = 0.98
lr = 0.02
memory_size = 7000
batch_size = 32
ctx = mx.cpu()
gamma = 0.9

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high


ddqn = double_DQN(actions_num=ACTION_SPACE,
                    state_dim=s_dim,
                    net_hidden_num=net_hidden_num,
                    lr=lr,
                    memory_size=memory_size,
                    batch_size=batch_size,
                    gamma=gamma,
                    ctx=ctx,
                    beta=0.9,
                  epislon = epislon
                 )


rewards = []
td_errors =[]

q=0
td_square=0

var = 3  # control exploration
for i in range(MAX_EPISODES):
    obsevation = env.reset()
    ep_reward = 0
    ep_td_error = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        action_index = ddqn.choose_action(obsevation)
        f_action = (action_index - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)  # convert to [-2 ~ 2] float actions
        obsevation_, reward, done, info = env.step(np.array([f_action]))

        ddqn.store_transition(obsevation, action_index, reward / 10, obsevation_)

        if ddqn.pointer > ddqn.memory_size:
            var *= .9995    # decay the action randomness
            _, td_square = ddqn.learn()

        obsevation = obsevation_

        ep_td_error += td_square
        ep_reward += reward

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -1000:RENDER = True
            break

    rewards.append(ep_reward)
    td_errors.append(ep_td_error)





