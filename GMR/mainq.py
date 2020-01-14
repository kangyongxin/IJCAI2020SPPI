#baseline

'''
用作对比实验
'''

#from maze.maze_env20 import Maze
# from maze.maze_env5 import Maze
from maze.maze3_env5 import Maze
from RL_brain_QLearning import QLearningTable
import matplotlib.pyplot as plt


def main_MAZE(env):
    n_trj = 2000
    RL = QLearningTable(actions=list(range(env.n_actions)))
    reward_list=[]
    step_r=[]
    for eps in range(n_trj):
        observation = env.reset()
        step = 0
        r_episode=0
        while step<50:
            step +=1
            env.render()
            #action = RL.random_action(str(observation))
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation),action,reward,str(observation_))
            r_episode += reward
            step_r.append(reward)
            observation = observation_
            if done:
                print("done!")
                break
        
        reward_list.append(r_episode)
    plt.plot(reward_list)
    plt.show()
    temp_step_r=[]
    for i in range(len(step_r)):
        if i<200 :
            temp_step_r.append(step_r[i])
        else:
            temp_step_r.append(sum(step_r[(i-190):i])/190)
    plt.plot(temp_step_r)
    plt.show()
        
            
        


if __name__ == "__main__":
    # env = gym.make('MontezumaRevengeNoFrameskip-v4')
    # main_MR()
    env = Maze()
    S_space = env.state_space()
    main_MAZE(env)