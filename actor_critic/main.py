"""
Actor-Critic reinforcement learning algorithm reimplementation with mxnet
Using:
mxnet 0.11.0
gym 0.8.0
"""


import numpy as np
import gym
import mxnet as mx
from AC_core import Actor_Critic



np.random.seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 3000
MAX_STEPS_PER_EP = 1000
DISPLAY_REWARD_THRESHOLD = 200
RENDER = False
Filter = True

actor_lr = 0.001
critic_lr = 0.01
batch_size = 1
ctx = mx.cpu()
gamma = 0.9

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

s_dim = env.observation_space.shape[0]
actions_num = env.action_space.n

AC   = Actor_Critic(actions_num=actions_num,
                    state_dim=s_dim,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    batch_size=batch_size,
                    gamma=gamma,
                    ctx=ctx
                    )



rewards = []
filtered_reward = 0
var = 3  # control exploration
for i in range(MAX_EPISODES):
    obsevation = env.reset()
    episode_reward = 0
    t = 0
    for j in range(MAX_STEPS_PER_EP):
        if RENDER:
            env.render()

        action = AC.choose_action(obsevation)
        obsevation_next, r, done, info = env.step(action)
        if done:
            r = -20

        AC.learn(obsevation, action, r, obsevation_next)

        obsevation = obsevation_next
        t += 1
        episode_reward += r

        if done  or t >= MAX_STEPS_PER_EP:
            if Filter:
                filtered_reward = filtered_reward * 0.95 + episode_reward * 0.05
            else:
                filtered_reward = episode_reward

            rewards.append(filtered_reward)
            if filtered_reward > DISPLAY_REWARD_THRESHOLD: RENDER=True

            print('episode:', i, 'reward:', int(filtered_reward))
            break




