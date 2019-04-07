"""
According to https://morvanzhou.github.io/tutorials/
required pytorch=0.41

"""
import numpy as np
import gym
import multiprocessing as mp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

N_KID = 10                  # half of the training population
N_GENERATION = 5000         # training step
LR = .05                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
CONFIG = [
    dict(game="CartPole-v0",
         n_feature=4, n_action=2, continuous_a=[False], ep_max_step=700, eval_threshold=500),
    dict(game="MountainCar-v0",
         n_feature=2, n_action=3, continuous_a=[False], ep_max_step=200, eval_threshold=-120),
    dict(game="Pendulum-v0",
         n_feature=3, n_action=1, continuous_a=[True, 2.], ep_max_step=200, eval_threshold=-180)
][0]    # choose your game


class net(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(net,self).__init__()
        self.fc1 = nn.Linear(input_dim,30)
        self.fc1.weight.data.normal_(0,1)
        self.fc2 = nn.Linear(30,20)
        self.fc2.weight.data.normal_(0,1)
        self.fc3 = nn.Linear(20,output_dim)
        self.fc3.weight.data.normal_(0,1)
    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        out = self.fc3(x)
        return out


def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def get_reward(network_param, num_p,env, ep_max_step, continuous_a, seed_and_id=None,):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        # for layer in network.children():
        #     np.random.seed(seed)
        #     layer.weight.data += torch.FloatTensor(sign(k_id) * SIGMA * np.random.randn(layer.weight.shape[0],layer.weight.shape[1]))
        #     np.random.seed(seed)
        #     layer.bias.data += torch.FloatTensor(sign(k_id) * SIGMA * np.random.randn(layer.bias.shape[0]))
        np.random.seed(seed)
        params = torch.FloatTensor(sign(k_id) * SIGMA * np.random.randn(num_p))
        Net = net(CONFIG['n_feature'],CONFIG['n_action'])
        Net.load_state_dict(network_param)
        for layer in Net.children():
            layer.weight.data += params[:layer.weight.shape[0]*layer.weight.shape[1]].view(layer.weight.shape[0],layer.weight.shape[1])
            layer.bias.data += params[layer.weight.shape[0]*layer.weight.shape[1]:layer.bias.shape[0]+layer.weight.shape[0]*layer.weight.shape[1]]
            params = params[layer.bias.shape[0]+layer.weight.shape[0]*layer.weight.shape[1]:]
    else:
        Net = net(CONFIG['n_feature'], CONFIG['n_action'])
        Net.load_state_dict(network_param)
    # run episode
    s = env.reset()
    ep_r = 0.
    for step in range(ep_max_step):
        a = get_action(Net, s, continuous_a)  # continuous_a 动作是否连续
        s, r, done, _ = env.step(a)
        # mountain car's reward can be tricky
        if env.spec._env_name == 'MountainCar' and s[0] > -0.1: r = 0.
        ep_r += r
        if done: break
    return ep_r


def get_action(network, x, continuous_a):
    x = torch.unsqueeze(torch.FloatTensor(x), 0)
    x = network.forward(x)
    if not continuous_a[0]: return np.argmax(x.detach().numpy(), axis=1)[0]      # for discrete action
    else: return continuous_a[1] * np.tanh(x.detach().numpy())[0]                # for continuous action


def train(network_param, num_p,optimizer, utility, pool):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling
    # 生成一些镜像的噪点,每一个种群一个噪点seed
    # distribute training in parallel
    '''apply_async 是异步非阻塞的。即不用等待当前进程执行完毕，随时根据系统调度来进行进程切换。'''
    jobs = [pool.apply_async(get_reward, (network_param, num_p,env, CONFIG['ep_max_step'], CONFIG['continuous_a'],
                                          [noise_seed[k_id], k_id], )) for k_id in range(N_KID*2)]
    # 塞了2*种群个进去
    rewards = np.array([j.get() for j in jobs])
    # 排列reward
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
    #All_data = []

    # for layer in network.children():
    #     weight_data = 0
    #     bias_data = 0
    #     for ui, k_id in enumerate(kids_rank):
    #         np.random.seed(noise_seed[k_id])
    #         weight_data += utility[ui] * sign(k_id) * np.random.randn(layer.weight.shape[0],layer.weight.shape[1])
    #         np.random.seed(noise_seed[k_id])
    #         bias_data += utility[ui] * sign(k_id) * np.random.randn(layer.bias.shape[0])
    #     weight_data = weight_data.flatten()
    #     All_data.append(weight_data)
    #     All_data.append(bias_data)
    All_data = 0
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])  # reconstruct noise using seed
        All_data += utility[ui] * sign(k_id) * np.random.randn(num_p)  # reward大的乘的utility也大
        # 用的噪声配列降序相乘系数 相加
    '''utility 就是将 reward 排序, reward 最大的那个, 对应上 utility 的第一个, 反之, reward 最小的对应上 utility 最后一位'''
    #All_data = [data/(2*N_KID*SIGMA) for data in All_data]
    #All_data = np.concatenate(All_data)
    gradients = optimizer.get_gradients(All_data/(2*N_KID*SIGMA))
    gradients = torch.FloatTensor(gradients)

    for layer in network_param.keys():
        if 'weight' in layer:
            network_param[layer] += gradients[:network_param[layer].shape[0]*network_param[layer].shape[1]].view(network_param[layer].shape[0],network_param[layer].shape[1])
            gradients = gradients[network_param[layer].shape[0] * network_param[layer].shape[1]:]
        if 'bias' in layer:
            network_param[layer] += gradients[:network_param[layer].shape[0]]
            gradients = gradients[network_param[layer].shape[0]:]
    return network_param, rewards


if __name__ == "__main__":
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2    # *2 for mirrored sampling  种群数
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    Net_org = net(CONFIG['n_feature'],CONFIG['n_action']).state_dict()
    #print(Net.fc1.weight.data[0][0])
    num_params = 0
    for r in list(Net_org):
        num_params+=Net_org[r].numel()
    env = gym.make(CONFIG['game']).unwrapped
    optimizer = SGD(num_params, LR)
    pool = mp.Pool(processes=N_CORE)  # 多线程
    mar = None      # moving average reward
    for g in range(N_GENERATION):
        t0 = time.time()
        Net_org, kid_rewards = train(Net_org, num_params,optimizer, utility, pool)
        # 更新了参数
        # test trained net without noise
        net_r = get_reward(Net_org, num_params,env, CONFIG['ep_max_step'], CONFIG['continuous_a'], None,)
        mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
        print(
            'Gen: ', g,
            '| Net_R: %.1f' % mar,
            '| Kid_avg_R: %.1f' % kid_rewards.mean(),
            '| Gen_T: %.2f' % (time.time() - t0),)
        if mar >= CONFIG['eval_threshold']: break

    # test
    print("\nTESTING....")
    #p = params_reshape(net_shapes, net_params)
    while True:
        s = env.reset()
        for _ in range(CONFIG['ep_max_step']):
            env.render()
            net_test = net(CONFIG['n_feature'],CONFIG['n_action'])
            net_test.load_state_dict(Net_org)
            a = get_action(net_test, s, CONFIG['continuous_a'])
            s, _, done, _ = env.step(a)
            if done: break
