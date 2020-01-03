from SPPI.GridMaze.maze_env20 import Maze
from SPPI.GridMaze.RL_brain_SIQ import SIQLearning
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal  # 求极值用
import pandas as pd
from matplotlib.pyplot import plot,savefig
def generate_trjs(M_trjs,N_steps,RL,env):

    trjs =[]
    for eps in range(M_trjs):
        trj = []
        step = 0
        observation = env.reset()
        state = RL.obs_to_state(observation)
        trj.append(state)
        while step < N_steps and RL.getRflag == False:
            step += 1
            env.render()
            #action = RL.random_action(observation)
            action = RL.choose_action(str(state))
            observation_, reward, done = env.step(action)
            observation = observation_
            if done:
                print("done during generate")
                RL.getRflag = True
                state=RL.obs_to_state(observation)
                trj.append(state)
                RL.resetIm(trj,reward)
                break
            state = RL.obs_to_state(observation)
            trj.append(state)
        trjs.append(trj)
    return trjs

def train(N,RL,env):
    """

    :param N: 每次训练的步数
    :return:
    """
    #n_state = len(S_space)
    trainnumber = 100000
    r = np.zeros(trainnumber)
    usingstep = [] # 跟主函数中的NumEps重复了
    steps = 0 # 用来评价收敛的速度
    for eps in range(trainnumber):
        """
        需要重新定义局部收敛的条件，我们只要得到局部的收敛策略即可
        """
        observation = env.reset()
        step = 0
        temp_trj =[]
        usingstep.append(step)
        while step < N:
            step +=1
            usingstep[eps]=step
            env.render()
            state =  RL.obs_to_state(observation)
            temp_trj.append(state)
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            if reward >0 :
                RL.getRflag =True
                state = RL.obs_to_state(observation)
                temp_trj.append(state)
                RL.resetIm(temp_trj,reward)

            state_= RL.obs_to_state(observation_)

            if str(state_) in RL.Im.index:
                maxImS = RL.Im['ImS'].max()
                # if RL.getRflag:
                #     """ 所有Im中的值都指导探索"""
                #     reward = reward*maxImS + maxImS
                #     r[eps] = r[eps] + reward  # 用来判断收敛
                #     RL.learn(str(state), action, reward, str(state_))
                if RL.Im.loc[str(state_), 'ImS'] == maxImS:
                    # if RL.getRflag == False:
                    """只有最大值指导探索 """
                    reward = reward*maxImS + maxImS
                    r[eps] = r[eps] + reward  # 用来判断收敛
                    RL.learn(str(state), action, reward, str(state_))
                    break

            r[eps] = r[eps] + reward  # 用来判断收敛
            RL.learn(str(state), action, reward, str(state_))
            observation = observation_


            if done:
                break


        #steps = steps + step
        #print("Im \n",RL.Im['ImS'])
        print("max Im \n",RL.Im)
        print("sum r",sum(r[eps - 10:eps]))

        if eps>10:

            # print("usingstep[eps]",usingstep[eps])
            # print("usingstep[eps-10]",usingstep[eps-10])

            if (sum(r[eps - 10:eps]) > 9* RL.Im['ImS'].max()) and sum(usingstep[eps - 10:eps]) == usingstep[eps] * 10:
                print("this turn have done")
                print("temp_trj",temp_trj)
                for state in temp_trj:
                    StateID = str(state)
                    RL.check_state_exist_Im(StateID)
                    RL.Im.loc[StateID, 'beforestate'] = 1
                break

    return  usingstep

def main_MAZE(env):
    M_trjs=100
    N_steps=50#will be changed dynamicly
    N0=10
    # a new  framework should be applied in this file
    RL = SIQLearning(env,actions=list(range(env.n_actions)))
    #collect trjs by current policy in limited steps
    tempn = 0
    while tempn < 40:
        tempn += 1
        trjs = generate_trjs(M_trjs,N_steps,RL,env)
        #print(trjs)
    # calculate current SI based intrinsic reward for known states
        P= RL.stochastic_trjs(trjs)
        print("stochastic value of trjs \n", P)
        Im = RL.Im_s(P,tempn) 
        print(Im)
        # keep the i^{SI}, train policy until convergency
        print("train")
        if RL.getRflag == False:
            usingstep=train(N_steps+2*N0, RL, env)
        print(usingstep)



if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)