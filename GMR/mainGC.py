#main2: 在实际任务中训练策略，训练过程中将记忆检索结果作为奖励值的一部分。
'''
图的基本构成：
    节点（node）：状态（或者是聚类的中心）  
    边(edge)：（已经尝试过的）动作
    节点的特征(attribute)：状态的特征向量（也可以有诸如值函数v的特征，或者访问次数，当然奖励也是必不可少的节点特征）
    边的特征（weight）:最简单的是有无和方向，还要有相应的状态动作值函数，（也可以有访问次数）
    链（sequence）：策略（策略组合）
    图（graph）：记忆
    图的更新（update of graph）：记忆的重构

核心内容是通过GNN 的信息传播来完成策略的更新。
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import curses

from maze.maze_env20 import Maze
#from maze.maze_env5 import Maze
#from maze.maze_env25 import Maze
#from maze.maze3_env20 import Maze

from GMRAgent import GMRAgent
import matplotlib.pyplot as plt
import numpy as np 

def main(argv=()):
    print("hello world")
    #game = make_game(int(argv[1]) if len(argv) > 1 else 0)
    #humanplayer_scrolly(game)
    env=Maze()
    agent= GMRAgent(actions=list(range(env.n_actions)))
    n_trj = 100
    reward_list=[]
    step_r=[]
    for eps in range(n_trj):
        observation = env.reset()
        step = 0
        re_vec = []
        r_episode =0
        while step <200:
            step +=1
            env.render()
            #action = agent.random_action(str(observation))
            state = agent.obs2state(observation)
            action = agent.ActAccordingToGM(state)
            observation_, reward, done = env.step(action)
            state_ = agent.obs2state(observation_)
            re_vec.append([state, action, reward, state_])
            #agent.MemoryWriter(re_vec)
            agent.PairWriter(state,action,reward,state_)
            r_episode += reward
            step_r.append(reward)
            observation = observation_
            if done:
                print("done!")
                break
        #print("re_vec",re_vec)
        #agent.MemoryWriter(re_vec)
        reward_list.append(r_episode)
        t1=120
        t2=100
        agent.MemoryReconstruction(t1,t2)

    #agent.plotGmemory()
    plt.figure(1)
    plt.plot(reward_list)
    #plt.show()
    plt.savefig("./RESULT/GCQ/reward_list003.png")
    reward_list=np.array(reward_list)
    np.save('./RESULT/GCQ/reward_list003.npy',reward_list)
    temp_step_r=[]
    for i in range(len(step_r)):
        if i<500 :
            temp_step_r.append(step_r[i])
        else:
            temp_step_r.append(sum(step_r[(i-490):i])/490)
    #print("temp_step_r",temp_step_r)
    plt.figure(2)
    plt.plot(temp_step_r)
    #plt.show()
    plt.savefig("./RESULT/GCQ/step_r003.png")
    temp_step_r=np.array(temp_step_r)
    np.save('./RESULT/GCQ/temp_step_r003.npy',temp_step_r)
    

    
    # print('state',state)
    # agent.MemoryReader(state)
    # plt.plot(reward_list)
    # plt.show()

    # temp_step_r=[]
    # for i in range(len(step_r)):
    #     if i<200 :
    #         temp_step_r.append(step_r[i])
    #     else:
    #         temp_step_r.append(sum(step_r[(i-19):i])/19)
    # plt.plot(temp_step_r)
    # plt.show()

if __name__ == '__main__':
    print("first come here")
    main(sys.argv)