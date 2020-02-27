#main2: 在实际任务中训练策略，训练过程中将记忆检索结果作为奖励值的一部分。
'''
图记忆重构：
    节点：状态（的聚类中心）
    边：已经尝试过的动作
    边的权重：（频次_无奖励/动作的值函数_有奖励）
    图中的链： 策略
    链上的权： 策略的值函数
    记忆编码是：节点的构造，相似节点的聚类，
    记忆的存储：节点的加入，边的加入，边权的改变
    （核心）记忆的重构：节点状态的更新，信息的传播，重新聚类

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