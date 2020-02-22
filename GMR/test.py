# import matplotlib

# import matplotlib.pyplot as plt 

# A=list(range(5,100))
# print(A)
# B=A
# plt.plot(A,B)
# plt.show()
# plt.savefig('/home/kangyx/IJCAI2020SPPI/GMR/RESULT/test.png')

# #!/usr/bin/python
# # -*- coding: UTF-8 -*-
 
# from tkinter import *           # 导入 Tkinter 库
# root = Tk()                     # 创建窗口对象的背景色
#                                 # 创建两个列表
# li     = ['C','python','php','html','SQL','java']
# movie  = ['CSS','jQuery','Bootstrap']
# listb  = Listbox(root)          #  创建两个列表组件
# listb2 = Listbox(root)
# for item in li:                 # 第一个小部件插入数据
#     listb.insert(0,item)
 
# for item in movie:              # 第二个小部件插入数据
#     listb2.insert(0,item)
 
# listb.pack()                    # 将小部件放置到主窗口中
# listb2.pack()
# root.mainloop()                 # 进入消息循环
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import curses


# #from maze.maze_env5 import Maze
from maze.maze_env25 import Maze
# #from maze.maze3_env20 import Maze

from GMRAgent import GMRAgent
# import matplotlib.pyplot as plt

def main(arg):
    print("hello world")

if __name__ == '__main__':
    print("first come here")
    main(sys.argv)