import matplotlib.pyplot as plt
import pandas as pd
import csv
import os


results_path = './SPPI/GridMaze/results/'
n_trjs = 10
n_steps = 100

path_reward_episode = results_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_episode.csv'
path_reward_step = results_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_step.csv'
reward_episode = pd.read_csv(path_reward_episode)
reward_episode = pd.DataFrame(reward_episode)
reward_step = pd.read_csv(path_reward_step)
reward_step = pd.DataFrame(reward_step)
print(reward_episode)
print(reward_step)


plt.plot(reward_episode['reward_eps'])
plt.savefig(results_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_episode.png')
plt.show()
plt.plot(reward_step['reward_step_sum'])
plt.savefig(results_path+'random'+'_'+str(n_trjs)+'_'+str(n_steps)+'_reward_step.png')
plt.show()