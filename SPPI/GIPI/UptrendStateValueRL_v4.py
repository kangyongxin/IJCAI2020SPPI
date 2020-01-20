
"""
Created on Sun Jul 14 16:17:21 2019

小环境，希望能找到解，并并完善流程
与 QL形成对比实验

@author: Administrator
"""

"""
接着State_Im_MAZE.py
获取关键状态，并进行可视化
按步运行，并动态显示我们找到的关键状态

为关键状态赋值，并将其融入Qlearning 算法
设计并完成比较实验，证明， 这种发现关键状态的方法可以加速Q-Learning的收敛

1.隔离状态编码，与，求解过程
2.重新设计策略提升方法
3.从随机策略学起
4.尽量精简流程

"""
import time
import gym
import numpy as np
from SPPI.GIPI.maze_env20 import Maze
from SPPI.GIPI.RL_brain4 import UptrendVS,QLearningTable, InternalModel
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

def show_trjs(trjs,RL):
    for trj in trjs:
        print("trj :")
        for i in range(len(trj)):
            print(RL.show_state(trj[i]))

    # def Im_s1(self,P):
    #     n_state = len(P)
    #     Im = pd.DataFrame([{'ImS': 0, 'beforestate': 0}], index=P.index, columns=['ImS', 'beforestate'],
    #                       dtype=np.float64)
    #
    #     ind_C = Im[P['BEplusAF'] == P['BEplusAF'].max()].index  # argmax??
    #     print(ind_C)
    #     Im.loc[ind_C, 'ImS'] = 1
    #
    #     return Im
# def trainQ(N,RL,env):
#     trainnumber = 100000




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

def trainOnly(N,RL,env):
    """

        :param N: 每次训练的步数
        :return:
        """
    # n_state = len(S_space)
    trainnumber = 100000
    r = np.zeros(trainnumber)
    usingstep = [] # 用来评价收敛的速度

    for eps in range(trainnumber):
        """
        需要重新定义局部收敛的条件，我们只要得到局部的收敛策略即可
        """
        #RL.resetgetstate()
        observation = env.reset()
        step = 0
        temp_trj = []
        RL.getstatereset()
        usingstep.append(step)
        while step < N:
            step += 1
            usingstep[eps] = step
            #print("step ",step)
            env.render()
            state = RL.obs_to_state(observation)
            temp_trj.append(state)
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            state_ = RL.obs_to_state(observation_)
            if reward > 0 and step < RL.minstep:
                RL.minstep = step
                RL.getRflag = True
                temp_trj.append(state_)
                RL.resetIm(temp_trj, reward)


            if str(state_) in RL.Im.index:
                #print("state_",state_)
                if RL.Im.loc[str(state_),'getstate']== 0:
                    reward = reward*RL.Im.loc[str(state_),'ImS'] + RL.Im.loc[str(state_),'ImS']
                    RL.Im.loc[str(state_), 'getstate'] = 1
                else:
                    reward = reward

            r[eps] = r[eps] + reward  # 用来判断收敛
            RL.learn(str(state), action, reward, str(state_))
            observation = observation_


            if done:

                print("done step",step)
                break
        #steps = steps + step
        #print("max Im \n", RL.Im)
        #print("sum r", sum(r))
        if eps >10:
            if sum(usingstep[eps - 10:eps]) == usingstep[eps] * 10:
                print("this turn have done")
                #print("temp_trj",temp_trj)
                # for state in temp_trj:
                #     StateID = str(state)
                #     RL.check_state_exist_Im(StateID)
                #     RL.Im.loc[StateID, 'beforestate'] = 1
                break

    return usingstep
def plotresults(NumEps,usingstep_Q):
    totalsteps = []
    for i in range(len(NumEps)):
        totalsteps.append(sum(NumEps[0:i]))
    t = range(0, len(NumEps))
    # mp.gcf().set_facecolor(np.ones(3) * 240 / 255)  # 设置背景色
    fig, ax1 = plt.subplots()  # 使用subplots()创建窗口
    ax2 = ax1.twinx()  # 创建第二个坐标轴
    ax1.plot(t, NumEps, '-', c='orangered', label='y1', linewidth=1)  # 绘制折线图像1,圆形点，标签，线宽
    ax2.plot(t, totalsteps,'-', c='blue', label='y2', linewidth=1)  # 同上

    ax1.set_xlabel('n epsodic', size=10)
    ax1.set_ylabel('steps each epsodic', size=10)
    ax2.set_ylabel('total search steps', size=10)
    # mp.gcf().autofmt_xdate()  # 自动适应刻度线密度，包括x轴，y轴
    #r_t=pd.DataFrame(t)
    r_NumEps = pd.DataFrame(NumEps)
    r_totalsteps = pd.DataFrame(totalsteps)
    #r_t.to_csv('r_t.csv')
    r_NumEps.to_csv('r_NumEps.csv')
    r_totalsteps.to_csv('r_totalsteps.csv')
    # mp.legend()  # 显示折线的意义
    plt.show()
    #savefig('./pic/env20tempn30_201908022014.png')

    totalsteps_q=[]
    for i in range(len(usingstep_Q)):
        totalsteps_q.append(sum(usingstep_Q[0:i]))
    t0 = range(0,len(usingstep_Q))
    fig1, ax10 = plt.subplots()  # 使用subplots()创建窗口
    ax20 = ax10.twinx()  # 创建第二个坐标轴
    ax10.plot(t0, usingstep_Q, '-', c='orangered', label='y1', linewidth=1)  # 绘制折线图像1,圆形点，标签，线宽
    ax20.plot(t0, totalsteps_q, '-', c='blue', label='y2', linewidth=1)  # 同上

    ax10.set_xlabel('n epsodic', size=10)
    ax10.set_ylabel('steps each epsodic', size=10)
    ax20.set_ylabel('total search steps', size=10)
    # mp.gcf().autofmt_xdate()  # 自动适应刻度线密度，包括x轴，y轴
    #r_t0= pd.DataFrame(t0)
    r_usingstep_Q= pd.DataFrame(usingstep_Q)
    r_totalsteps_q = pd.DataFrame(totalsteps_q)
    #r_t0.to_csv('r_t0.csv')
    r_usingstep_Q.to_csv('r_usingstep_Q.csv')
    r_totalsteps_q.to_csv('r_totalsteps_q.csv')
    # mp.legend()  # 显示折线的意义
    plt.show()
    #savefig('./pic/env20tempn30_201908022014_Q.png')

def train_Q(RL_Q,env):
    steps = []
    trainnumber = 100000
    while len(steps)<trainnumber:
        """
        需要重新定义局部收敛的条件，我们只要得到局部的收敛策略即可
        """
        observation = env.reset()
        step = 0 #规定步骤内，稳定地达到某个分数就为局部收敛
        score = 0
        while True:#每次只训练100步
            step += 1
            env.render()
            # action = RL.random_action(str(observation))
            action = RL_Q.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            # if reward == -1:
            #     break

            RL_Q.learn(str(observation), action, reward, str(observation_))
            score = score +reward
            observation = observation_
            if done:
                print("done steps: ", step)
                break

        steps.append(step)
        print("steps", sum(steps),"score",score)
        if len(steps) >10:
            if sum(steps[len(steps) - 11:len(steps)-1]) == steps[-1] * 10:
                print("this turn have done")
                #print("temp_trj",temp_trj)
                # for state in temp_trj:
                #     StateID = str(state)
                #     RL.check_state_exist_Im(StateID)
                #     RL.Im.loc[StateID, 'beforestate'] = 1
                break
    return steps

def main_MAZE(env):
    RL = UptrendVS(env,actions=list(range(env.n_actions)))
    M_trjs =100#30
    N_steps = 50#10
    N0=5
    NumEps = [] # 统计每个训练回合到达终点所用步数

    tempn =0
    strr =0
    while tempn<40:
        tempn +=1
        print("tempn",tempn)
        #show_trjs(trjs,RL)
        #P=RL.stochastic_trj(trjs[1])
        if RL.getRflag == False:
            print("generate")
            trjs = generate_trjs(M_trjs, N_steps - N0, RL, env)
            P= RL.stochastic_trjs(trjs)
            print("stochastic value of trjs \n", P)
            #BE = BE_Sk()
            Im = RL.Im_s(P,tempn)
            #print('Im s \n',Im)
            print("train")
            if RL.getRflag == False:
                usingstep=train(N_steps+2*N0, RL, env)
        else:
            print("train only")
            #RL.epsilon = 0.5
            usingstep= trainOnly(N_steps + 10 * N0, RL, env)
        # print("steps",steps)
        # print("step",sr)
        ShortestStep = usingstep[-1]
        N_steps = N_steps + ShortestStep
        NumEps=NumEps+usingstep
        #print(NumEps)

    RL_Q= QLearningTable(actions=list(range(env.n_actions)))
    usingstep_Q = train_Q(RL_Q, env)  # 根据各个状态的价值提升策略

    print(RL.Im)
    plotresults(NumEps,usingstep_Q)




if __name__ == "__main__":
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)



