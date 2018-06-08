from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()  # NOQA
import sys
sys.path.append("/anaconda3/lib/python3.6/site-packages/chainer_master/tests/agents_tests")
from chainer import optimizers
from chainer import testing
import gym
gym.undo_logger_setup()  # NOQA

import basetest_agents as base
from chainerrl import agents
from chainerrl import explorers
from chainerrl import policies
from chainerrl import q_functions
from chainerrl import replay_buffer
from chainerrl import v_function



env = gym.make("MountainCarContinuous-v0")



def create_deterministic_policy_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    return policies.FCDeterministicPolicy(
        n_input_channels=ndim_obs,
        action_size=env.action_space.low.size,
        n_hidden_channels=200,
        n_hidden_layers=2,
        bound_action=False)


def create_state_action_q_function_for_env(env):
    assert isinstance(env.observation_space, gym.spaces.Box)
    assert isinstance(env.action_space, gym.spaces.Box)
    ndim_obs = env.observation_space.low.size
    return q_functions.FCSAQFunction(
        n_dim_obs=ndim_obs,
        n_dim_action=env.action_space.low.size,
        n_hidden_channels=200,
        n_hidden_layers=2)





class TestDDPG(base._TestAgentInterface):

    def create_agent(env):
        model = agents.ddpg.DDPGModel(
            policy=create_deterministic_policy_for_env(env),
            q_func=create_state_action_q_function_for_env(env))
        rbuf = replay_buffer.ReplayBuffer(10 ** 5)
        opt_a = optimizers.Adam()
        opt_a.setup(model.policy)
        opt_b = optimizers.Adam()
        opt_b.setup(model.q_function)
        explorer = explorers.AdditiveGaussian(scale=1)
        return agents.DDPG(model, opt_a, opt_b, rbuf, gamma=0.99,
                           explorer=explorer)




agent = TestDDPG.create_agent(env)




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
        #env.render()
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