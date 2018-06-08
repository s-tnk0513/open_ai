import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
import time

#ゴールは旗を確保すること
env = gym.make("MountainCar-v0")



class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
       
       #n_hidden_channels=繋がっているノード数。
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels,n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))
        
    def __call__(self, x, test=False): 
        h = F.tanh(self.l0(x)) 
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))

#状態の数
#ここでは
#obs_size=2
obs_size = env.observation_space.shape[0]
#行動の数
#0:push left
#1:no push
#2:push right
#n_actions=3
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)



optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func) #設計したq関数の最適化にAdamを使う

#割引率
gamma = 0.95

#ε-greedyで決める。
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**6)
phi = lambda x:x.astype(np.float32, copy=False)##型の変換(chainerはfloat32型。float64は駄目)

#DoubleDQNはDQNの中に含まれる。
agent = chainerrl.agents.DoubleDQN(
    #作成したQ関数
    q_func,

    #Q関数の最適化にAdamアルゴリズムを利用する。
    #（Adamアルゴリズムは最急降下法のように、損失関数の最小値を求める。）
    optimizer, 

    #bufferは一時的にデータを貯めて置く場所
    replay_buffer, 
    #割引率
    gamma, 
    #ε-greedy法
    explorer,
    
    replay_start_size=500, 
    #Model update interval in step
    update_interval=1,
    #ターゲットモデルは理想のモデル
    #Target model update interval in step
    target_update_interval=100000, 
    phi=phi
    )

#エピソード（試行）の数
n_episodes = 3000
#1時間ステップに行う試行数
max_episode_len = 200

agent.load("ε_0.3_agent")

for i in range(1, n_episodes + 1):
    env.reset()
    obs, r, done, _ = env.step(env.action_space.sample())
    R = 0  # return (sum of rewards)
    t = 0
    #1時間ステップに最大200回Q関数を試行する。
    #もし、倒れたらそのステップは終了
    while not done and t < max_episode_len:
        env.render()
        #action=0or1
        #ターゲットモデルはact_and_train
        action = agent.act_and_train(obs, r)
        obs, r, done, _ = env.step(action)
        if obs[0] > 0.5:
            print("SUCCESS_REWARD:",r)
            r = 10000000
            R = 0
        R += r
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
        agent.save("ε_0.3_agent")
    #1エピソードを終了させて次のエピソードへ
    agent.stop_episode_and_train(obs, r, done)

agent.save("ε_0.3_agent")
env.close()






