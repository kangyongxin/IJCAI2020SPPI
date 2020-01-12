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


from pycolab.examples.scrolly_maze import *
from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import drapes as prefab_drapes
from pycolab.prefab_parts import sprites as prefab_sprites

from maze.maze_env20 import Maze
from GMRAgent import GMRAgent

def humanplayer_scrolly(game):
    print(game)
    ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN: 1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4,
                       'q': 5, 'Q': 5},
      delay=100, colour_fg=COLOUR_FG, colour_bg=COLOUR_BG)

    # Let the game begin!
    ui.play(game)



def main(argv=()):
    print("hello world")
    #game = make_game(int(argv[1]) if len(argv) > 1 else 0)
    #humanplayer_scrolly(game)
    env=Maze()
    agent= GMRAgent(actions=list(range(env.n_actions)))
    n_trj = 1000
    for eps in range(n_trj):
        observation = env.reset()
        step = 0
        while step <100:
            step +=1
            env.render()
            #action = agent.random_action(str(observation))
            state = agent.obs2state(observation)
            action = agent.ActAccordingToGM(state)
            observation_, reward, done = env.step(action)
            state_ = agent.obs2state(observation_)
            agent.MemoryWriter(state, action, reward, state_)
            observation = observation_
            if done:
                print("done!")
                break
    agent.plotGmemory()
    print('state',state)
    agent.MemoryReader(state)

if __name__ == '__main__':
  main(sys.argv)