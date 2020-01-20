"""
获取关键状态，并进行可视化
按步运行，并动态显示我们找到的关键状态
为关键状态赋值，并将其融入Qlearning 算法
设计并完成比较实验，证明， 这种发现关键状态的方法可以加速Q-Learning的收敛

"""
import gym
import numpy as np
import pandas as pd
#from baselines.UpTrend.maze_env import Maze
from baselines.GIPI.minimaze import Maze
from baselines.GIPI.RL_brain import QLearningTable,InternalModel
import matplotlib.pyplot as plt
import scipy.signal as signal #求极值用

def stochastic_trj(seq):
    """

    :param seq:  Ti

    :return: Ui={u_i1, u_i2, ..., u_iW_i} & Ci ={c_i1, c_i2, ...,c_iW_i}
    """
    #print("trj,", seq)
    n_state = len(S_space)
    Ui = []  # list of the unique state, ordered by the first time they appeared
    Ci = []  # list of the number of each state corresponded with Ui
    Wik = np.zeros((n_state, 1))
    k = 0
    for i in seq:
        if Ui.count(i) == 0:
            Ci.append(seq.count(i))
            Ui.append(i)
            k = k + 1
    for k in range(len(S_space)):
        for w in range(len(Ui)):
            if Ui[w]== S_space[k]:
                Wik[k]=w+1 #第一个出现的状态计为1， 没有出现的记为0
    # print("S_space",S_space)
    # print("seq",seq)
    # print("Ui",Ui)
    # print("Ci",Ci)
    # print("Wik",Wik)

    return Ui, Ci, Wik

def State_Importance_calculate(trjs):
    a=[]
    r=[]
    n_state= len(S_space)
    Im_s = np.zeros((n_state, 1))
    Im_p = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        #print("trjs",trjs[eps])
        [u_state, count_state] = stochastic_trj(trjs[eps])
        a.append(u_state)
        r.append(count_state)  # 每条轨迹的统计特性。状态出现的次数
        for i in range(len(S_space)):  # 某个状态
            for w in range(len(a[eps])):  # 第w个 出现
                if a[eps][w] == S_space[i]:
                    Im_s[i] = Im_s[i] + w  # 出现的顺序 越晚月重要？轨迹中
                    Im_p[i] = Im_p[i] + r[eps][w]  # 状态出现的次数
    for i in range(len(S_space)):
        Im_s[i] = Im_p[i] / Im_s[i]
        if Im_s[i]<0.01:
             Im_s[i]=0
    return Im_s
def Im_SumWik(trjs):
    """
    only calculate the sum of Wik, where the Wik is the order of state k in the list Ui
    :param trjs:
    :return:
    """
    U = []
    C = []
    n_state = len(S_space)
    SumWi = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        SumWi=SumWi + Wik
    return SumWi

def Im_Pk(trjs):
    """
    the number of S_k in all the trjs
    :param trjs:
    :return: Im_s
    """
    n_state = len(S_space)
    P = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        # print("S_space",S_space)
        # print("seq",trjs[eps])
        # print("Ui",Ui)
        # print("Ci",Ci)
        # print("Wik",Wik)
        for k in range(len(S_space)):
            if Wik[k]==0:#没有出现的不计
                continue
            else:
                ind= int(Wik[k])-1#第一个出现的状态计为1， 没有出现的记为0
                P[k]=P[k]+Ci[ind]

        # print("P",P)
    return P

def Im_BEk(trjs):
    """
    the stochastic of states BEfore S_k in each trj
    :param trjs:
    :return:
    """
    n_state = len(S_space)
    BE = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        # print("S_space",S_space)
        # print("seq",trjs[eps])
        # print("Ui",Ui)
        # print("Ci",Ci)
        #print("Wik",Wik)
        for k in range(len(S_space)):
            # print("k",k)
            # print("WIK",Wik[k])
            if Wik[k] == 0 or Wik[k] == 1:#没有出现的和第一个出现的都不计入
                continue # never appear in this trj
            else:
                ind=int(Wik[k])-1
                # print("ind",ind)
                for j in range(ind): #只计算之前的，自己不算
                    # print("j",j)
                    # print("BE",BE[k])
                    # print("CI",Ci[j])
                    BE[k]=BE[k]+Ci[j]
        #             print("BE",BE[k])
        # print("BE",BE)
    return BE

def Im_AFk(trjs):
    """
    the stochastic of states AFter S_k in each trjs
    :param trjs:
    :return:
    """
    n_state = len(S_space)
    AF = np.zeros((n_state, 1))
    for eps in range(len(trjs)):
        [Ui, Ci, Wik] = stochastic_trj(trjs[eps])
        # print("S_space",S_space)
        # print("seq",trjs[eps])
        # print("Ui",Ui)
        # print("Ci",Ci)
        # print("Wik",Wik)
        for k in range(len(S_space)):
            # print("k",k)
            # print("WIK",Wik[k])

            if Wik[k] == 0 or Wik[k] == len(Ui):  # 没有出现的和第一个出现的都不计入if Wik[k] == 0 or Wik[k] == len(Ui) :  # 没有出现的和第一个出现的都不计入
                continue  # never appear in this trj
            else:
                ind = int(Wik[k]) - 1
                # print("ind",ind)
                for j in range(ind+1,len(Ui)):  # 只计算之前的，自己不算
                    # print("j",j)
                    # print("AF",AF[k])
                    # print("CI",Ci[j])
                    AF[k] = AF[k] + Ci[j]
                    # print("AF",AF[k])
        # print("BE",BE)
        #AF[0]=0
    return AF
def Im_O_OBE_OAF(trjs):
    """
    Order Pk, BEk, and AFk
    :param trjs:
    :return: O OBE AF
    """
    n_state = len(S_space)
    P = Im_Pk(trjs)
    #print("p",P)
    BE = Im_BEk(trjs)
    #print("BE",BE)
    AF = Im_AFk(trjs)
    #print("AF",AF)
    O = np.sort(-P,axis=0)# 下降
    #print("O",O)
    index_O = np.argsort(-P,axis=0)
    #print("index_O",index_O)
    OBE = np.zeros((n_state, 1))
    OAF = np.zeros((n_state, 1))
    i=0
    for ind in index_O:
        OBE[i] = BE[ind]
        OAF[i] = AF[ind]
        i+=1
    #print("obe",OBE)
    #print("OAF",OAF)
    return -O,index_O,OBE,OAF,P,BE,AF
def Im_BEDividedByAF(trjs): # BE/AF
    n_state = len(S_space)
    BE = Im_BEk(trjs)
    #print("BE",BE)
    AF = Im_AFk(trjs)
    #print("AF",AF)
    Im = BE/AF
    #print(Im)
    return Im

def main_MAZE(env):
    n_trj = 1000
    RL = QLearningTable(actions=list(range(env.n_actions)))

    trjs =[]
    for eps in range(n_trj):
        observation = env.reset()
        state = env.obs_to_state(observation)
        #print(state)
        trj = []
        trj.append(state)
        step = 0


        while step <100:
            step +=1
            env.render()

            action = RL.random_action(str(observation))
            observation_, reward, done = env.step(action)
            # if reward == -1:
            #     break

            observation = observation_

            if done:
                print("done!")
                break
            state = env.obs_to_state(observation)
            #print(state)
            trj.append(state)
        trjs.append(trj)
    O, index_O, OBE, OAF, P, BE, AF= Im_O_OBE_OAF(trjs)

    r_P = pd.DataFrame(P)
    r_BE = pd.DataFrame(BE)
    r_AF = pd.DataFrame(AF)
    r_AF.to_csv('r_AF.csv')
    r_BE.to_csv('r_BE.csv')
    r_P.to_csv('r_P.csv')

    txt_O = index_O.copy()+ 1
    firmsP = [i for i in range(1,len(S_space)+1)]
    firms = txt_O
    x_axs = [i for i in range(len(txt_O))]


    print("index_o",index_O)

    print("MAX VALUE OF OBE ",OBE[signal.argrelextrema(OBE, np.greater)])
    print("MAX POINT OF OBE ",signal.argrelextrema(OBE, np.greater))
    inds,_ = signal.argrelextrema(OBE, np.greater)
    #print("MAX POINT OF OBE ", inds)
    l= len(inds)
    for i in range(l):
        print("extrema points",index_O[inds[i]]+1)#= 0*(i+1)/l # 对应回到原先的状态上
    ax1 = plt.subplot(1,3,1)
    plt.plot(P,color='blue')
    plt.xticks(x_axs, firmsP, rotation=90)
    ax1.set_title("f_D")

    ax3 = plt.subplot(1,3,2)
    plt.plot(BE,color='blue')
    plt.xticks(x_axs, firmsP, rotation=90)
    ax3.set_title("f_A")

    ax5 = plt.subplot(1,3,3)
    plt.plot(AF,color='blue')
    plt.xticks(x_axs, firmsP, rotation=90)
    ax5.set_title("f_P")
    #
    plt.show()
    # ax2 = plt.subplot(3,2,2)
    # plt.plot(O,color='red')
    # plt.xticks(x_axs, firms, rotation=90)
    # ax2.set_title("O")
    #
    # ax4 = plt.subplot(3, 2, 4)
    # plt.plot(OBE, color='red',label="point")
    # plt.xticks(x_axs, firmsP, rotation=90)
    # ax4.set_title("OBE")
    #
    #
    # # ax6 = plt.subplot(3, 2, 6)
    # # plt.plot(OAF, color='red')
    # # plt.xticks(x_axs, firms, rotation=90)
    # # ax6.set_title("OAF")
    #
    # OBE_O = BE/P+AF/P
    # ax6 = plt.subplot(3, 2, 6)
    # plt.plot(OBE_O, color='red')
    # plt.xticks(x_axs, firmsP, rotation=90)
    # ax6.set_title("OBE_O")
    # ax4 = plt.subplot(1, 1, 1)
    # plt.plot(OBE, color='red',label="point")
    # plt.xticks(x_axs, firms, rotation=90)
    # ax4.set_title("OBE")
    # plt.show()

    # Im = Im_BEDividedByAF(trjs)
    # plt.plot(Im)
    # plt.show()



if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)