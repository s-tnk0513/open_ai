import gym
env = gym.make("MountainCarContinuous-v0")
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import chainerrl
from chainerrl.agents.ddpg import DDPG, DDPGModel
from chainerrl import policy, q_functions
from chainerrl import experiments, explorers, misc
from chainerrl import policy, replay_buffer, distribution
from chainerrl.misc.batch_states import batch_states
from chainerrl.replay_buffer import ReplayUpdater
from chainerrl.recurrent import Recurrent
from chainerrl.recurrent import state_kept

import gym
from gym import spaces
import numpy as np
import time
import random
from logging import getLogger
import copy



env = gym.make("MountainCarContinuous-v0")

obs_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

print("obs_size",obs_size)
print("actions_size",action_size)

pi = policy.FCDeterministicPolicy(
    obs_size,
    n_hidden_layers=2,
    n_hidden_channels=100,
    action_size=1,
    min_action=0,
    max_action=1,
    bound_action=True,
    last_wscale=1)

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        super().__init__(
            l0=L.Linear(obs_size+n_actions, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, 1))
    
    def __call__(self, state, action, test=False):
        h = F.concat((state, action), axis=1)
        h = F.tanh(self.l0(h))
        h = F.tanh(self.l1(h))
        h = F.tanh(self.l2(h))
        return h



q_func = QFunction(obs_size, action_size)




model = DDPGModel(q_func=q_func, policy=pi)
actor_optimizer = chainer.optimizers.Adam(eps=1e-2)
actor_optimizer.setup(model['q_function'])
actor_optimizer.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
critic_optimizer = chainer.optimizers.Adam(eps=1e-2)
critic_optimizer.setup(model['policy'])
critic_optimizer.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**6)
gamma = 0.99
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)
phi = lambda x:x.astype(np.float32, copy=False)
agent = DDPG(
    model, actor_optimizer, critic_optimizer, replay_buffer, 
    gamma, explorer=explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, phi=phi)



#エピソード（試行）の数
n_episodes = 2000
#1時間ステップに行う試行数
max_episode_len = 200


#====学習フェーズ====
for i in range(1, n_episodes + 1):
    env.reset()
    obs, reward, done, _ = env.step(env.action_space.sample())
    R = 0  # return (sum of rewards)
    t = 0  # time step
    #1時間ステップに最大200回Q関数を試行する。
    #もし、倒れたらそのステップは終了
    #doneは終了判定
    while not done and t < max_episode_len:
        #action=0or1
        env.render()
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
agent.stop_episode_and_train(obs, reward, done)
env.close()