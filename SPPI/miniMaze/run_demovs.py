
from SPPI.miniMaze.minimaze import Maze
from SPPI.miniMaze.RL_brain import DemoSV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def ShowInHeatmap(p,S_space):
    pk = np.zeros(len(S_space))
    be = np.zeros(len(S_space))
    af = np.zeros(len(S_space))
    print(p.shape[0])
    t=p.iloc[2]
    print(t.loc["SumWik"])
    print(p.iloc[2,2])
    for i in S_space:
        for j in range(p.shape[0]):
            if int(i)==int(p.iloc[j,0]):#'sta_visul'
                pk[i-1] = p.iloc[j,3]#'Pk'
                be[i-1] = p.iloc[j,5]
                af[i-1] = p.iloc[j,6]
    print("pk",pk,"\n","be",be,"\n","af",af,"\n")
    pkarray= np.zeros([6,6])
    bearray= np.zeros([6,6])
    afarray= np.zeros([6,6])
    for i in range(6):
        for j in range(6):
            pkarray[i,j]= pk[i*6+j]
            bearray[i,j]= be[i*6+j]
            afarray[i,j]= af[i*6+j]
    sns.heatmap(afarray, cmap='Reds')
    plt.show()
    sns.heatmap(pkarray,cmap='Blues')
    plt.show()
    sns.heatmap(bearray,cmap='Greens')
    plt.show()


    


def main_MAZE(env,S_space):
    n_trj = 1000
    RL = DemoSV(env,actions=list(range(env.n_actions)))
    trjs =[]
    for eps in range(n_trj):
        trj=[]
        observation = env.reset()
        state=RL.obs2state(observation)
        step = 0
        trj.append(state)
        while step <100:
            step +=1
            env.render()

            action = RL.random_action(str(observation))
            observation_, reward, done = env.step(action)

            state_ = RL.obs2state(observation_)
            trj.append(state_)

            observation = observation_

            if done:
                print("done!")
                break
        trjs.append(trj)
    # #show trjs
    # for trj in trjs:
    #     print("trj :")
    #     for i in range(len(trj)):
    #         print(trj[i])
    #     print("\n")

    p = RL.stochastic_trjs(trjs)
    print(p)

    ShowInHeatmap(p,S_space)
    

    


if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env,S_space)