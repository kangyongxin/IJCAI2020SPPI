"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""


import numpy as np
import random
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


# UNIT = 40   # pixels
# MAZE_H = 20 # grid height
# MAZE_W = 20  # grid width
# N_HELL = 120
UNIT = 40   # pixels
MAZE_H = 10 # grid height
MAZE_W = 10  # grid width
N_HELL = 12
SLEEPtime = 0.0000011

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        self.hell_list =[]
        for i in range(N_HELL):
            random.seed(i+0.758)
            p = random.randint(0,MAZE_H)
            q = random.randint(0,MAZE_H)
            if p==0 and  q==0:
                continue
            if p==MAZE_H and q==MAZE_H:
                continue
            hc = origin + np.array([UNIT*p,UNIT*q])
            self.hc = self.canvas.create_rectangle(
                hc[0] - 15, hc[1] - 15,
                hc[0] + 15, hc[1] + 15,
                fill='black')
            self.hell_list.append(self.canvas.coords(self.hc))

        # hell1_center = origin + np.array([UNIT * 2, UNIT])
        # self.hell1 = self.canvas.create_rectangle(
        #     hell1_center[0] - 15, hell1_center[1] - 15,
        #     hell1_center[0] + 15, hell1_center[1] + 15,
        #     fill='black')
        # # hell
        # hell2_center = origin + np.array([UNIT, UNIT * 2])
        # self.hell2 = self.canvas.create_rectangle(
        #     hell2_center[0] - 15, hell2_center[1] - 15,
        #     hell2_center[0] + 15, hell2_center[1] + 15,
        #     fill='black')
        #
        # # hell
        # hell3_center = origin + np.array([UNIT*3, UNIT*0])
        # self.hell3 = self.canvas.create_rectangle(
        #     hell3_center[0] - 15, hell3_center[1] - 15,
        #     hell3_center[0] + 15, hell3_center[1] + 15,
        #     fill='black')
        #
        # # hell
        # hell4_center = origin + np.array([UNIT*3, UNIT*4])
        # self.hell4 = self.canvas.create_rectangle(
        #     hell4_center[0] - 15, hell4_center[1] - 15,
        #     hell4_center[0] + 15, hell4_center[1] + 15,
        #     fill='black')
        # # hell
        # hell5_center = origin + np.array([UNIT*2, UNIT*5])
        # self.hell5 = self.canvas.create_rectangle(
        #     hell5_center[0] - 15, hell5_center[1] - 15,
        #     hell5_center[0] + 15, hell5_center[1] + 15,
        #     fill='black')
        #
        # # hell
        # hell6_center = origin + np.array([UNIT*4, UNIT*3])
        # self.hell6 = self.canvas.create_rectangle(
        #     hell6_center[0] - 15, hell6_center[1] - 15,
        #     hell6_center[0] + 15, hell6_center[1] + 15,
        #     fill='black')

        # create oval
        oval_center = origin + UNIT *(MAZE_H-1)
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(SLEEPtime)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def sit(self,action):
        s_ = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up 换乘 down
            base_action[1] += UNIT
        elif action == 1:   # down => up
            base_action[1] -= UNIT
        elif action == 2:   # right =>left
            base_action[0] -= UNIT
        elif action == 3:   # left =>right
            base_action[0] += UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

    def step(self, action):
        s = self.canvas.coords(self.rect)
        temp = s.copy()
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 200
            done = True
            s_ = 'terminal'
        elif s_ in self.hell_list:#[self.canvas.coords(self.hell1), self.canvas.coords(self.hell2),self.canvas.coords(self.hell3),self.canvas.coords(self.hell4),self.canvas.coords(self.hell5), self.canvas.coords(self.hell6)]:
            # reward = -1
            # done = True
            # # s_ = s#'terminal'
            # # # done = False
            # # # s_=temp
            reward =-1
            done = False
            self.sit(action)
            s_ = s


        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(SLEEPtime)
        self.update()

    def state_space(self):
        S_space = []
        for i in range(MAZE_H*MAZE_W):
            S_space.append(i+1)
        return S_space

    def obs_to_state(self,obs):
        """
        obs=[45.0, 5.0, 75.0, 35.0]
        第1行 第2列
        state = 2
        state:
        1 2 3 4 5 6
        7 8 9 10 11 12
        ...
        重新更改，从 0开始，避免多次进行切换赵城的不便。

        """
        states = ((obs[1] + 15.0 - 20.0) / 40) *MAZE_H + (obs[0] + 15.0 - 20.0) / 40+1
        return int(states)

    def state_to_obs(self,stat):
        stat=stat-1
        c1,c0 = stat/MAZE_H,stat%MAZE_H
        obser=[0,0,0,0]
        obser[0] = c0*40 + 5
        obser[1] = c1*40 + 5
        obser[2] = obser[0] + 30
        obser[3] = obser[1] + 30
        self.update()
        return obser
    def state_visualization(self,obs):
        """
        obs=[45.0, 5.0, 75.0, 35.0]
        第1行 第2列
        state = 2
        state:
        1 2 3 4 5 6
        7 8 9 10 11 12
        ...
        重新更改，从 0开始，避免多次进行切换赵城的不便。

        """
        states = ((obs[1] + 15.0 - 20.0) / 40) *MAZE_H + (obs[0] + 15.0 - 20.0) / 40+1
        return int(states)

def update():
    for t in range(10.10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()