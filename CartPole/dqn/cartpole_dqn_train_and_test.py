import gym
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np

env = gym.make("CartPole-v0")


class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
       
       #n_hidden_channels=繋がっているノード数。
        super(QFunction,self).__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels,n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))
        
    def __call__(self, x, test=False): 
        h = F.tanh(self.l0(x)) 
        h = F.tanh(self.l1(h))
        #ここで誤差関数を定義しているのでは？裏でおそらく二条誤差をやっている。
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))


#状態空間の変数の数
obs_size = env.observation_space.shape[0]
#行動空間の変数の数
n_actions = env.action_space.n
#Q関数
q_func = QFunction(obs_size, n_actions)

#設計したq関数の最適化にAdamを使う
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func) 
#割引率
gamma = 0.95

#ε-greedyで決める。
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.action_space.sample)

#【DQNの工夫1】Experiment Replay
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity = 10**6)

##型の変換(chainerはfloat32型。float64は駄目)
phi = lambda x:x.astype(np.float32, copy=False)


agent = chainerrl.agents.DoubleDQN(
    #作成したQ関数
    q_func,
    #Q関数の最適化にAdamアルゴリズムを利用する。
    #（Adamアルゴリズムは最急降下法のように、損失関数の最小値を求める。）
    optimizer, 
    #bufferは一時的にデータを貯めて置く場所（それに対して、メモリは一時的にためておく場所）
    replay_buffer, 
    #割引率
    gamma, 
    #ε-greedy法
    explorer,
    #500溜まったら、ニューラルネットは学習を始める。
    replay_start_size=500, 
    #Model update frequency in step
    update_interval=1,
    #【DQNの工夫2】Fixed Target Q-Network
    #Target model update frequency in step
    #1ステップに100回はコピーしよう。
    target_update_interval=100, 
    phi=phi
    )


#エピソード（試行）の数
n_episodes = 200
#1時間ステップに行う試行数
max_episode_len = 200

# agentのモデル
# agent.load("cartpole_dqn_agent")

for i in range(1, n_episodes + 1):
    obs = env.reset()
    #【DQNの工夫その３】報酬のクリッピング
    reward = 0
    done = False
    R = 0  # return (sum of rewards)
    t = 0  # time step
    #1時間ステップに最大200回Q関数を試行する。
    #もし、倒れたらそのステップは終了
    #doneは終了判定
    while not done and t < max_episode_len:
        action = agent.act_and_train(obs, reward)
        env.render()
        obs, reward, done, _ = env.step(action)
        R += reward
        t += 1
    if i % 10 == 0:
        print('episode:', i,
              'R:', R,
              'statistics:', agent.get_statistics())
agent.stop_episode_and_train(obs, reward, done)
agent.save("cartpole_dqn_agent")



#エピソード（試行）の数
n_episodes = 200
#1時間ステップに行う試行数
max_episode_len = 200
#load model
# agent.load("cartpole_dqn_agent")

for i in range(1,n_episodes):
    obs = env.reset()
    done = False
    R = 0
    t = 0
    while not done and t < 200:
        env.render()
        action = agent.act(obs)
        obs, r, done, _ = env.step(action)
        R += r
        t += 1
    print('test episode:', i, 'R:', R)
    agent.stop_episode()
