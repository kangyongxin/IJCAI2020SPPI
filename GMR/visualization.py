#https://blog.csdn.net/monotonomo/article/details/83342768

import re

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.cm as cm

import shutil

import os

sns.set_style('whitegrid')

import numpy as np


Qpath = './RESULT/Q/temp_step_r00'
Q1name= Qpath + '1.npy'
Q2name= Qpath + '2.npy'
Q3name= Qpath + '3.npy'
Q1=np.load(Q1name)
Q2=np.load(Q2name)
Q3=np.load(Q3name)
Q1=Q1.tolist()
Q2=Q2.tolist()
Q3=Q3.tolist()
GQpath = './RESULT/GQ/temp_step_r00'
GQ1name= GQpath + '1.npy'
GQ2name= GQpath + '2.npy'
GQ3name= GQpath + '3.npy'
GQ1=np.load(GQ1name)
GQ2=np.load(GQ2name)
GQ3=np.load(GQ3name)
GQ1=GQ1.tolist()
GQ2=GQ2.tolist()
GQ3=GQ3.tolist()
GCQpath = './RESULT/GCQ/temp_step_r00'
GCQ1name= GCQpath + '1.npy'
GCQ2name= GCQpath + '2.npy'
GCQ3name= GCQpath + '3.npy'
GCQ1=np.load(GCQ1name)
GCQ2=np.load(GCQ2name)
GCQ3=np.load(GCQ3name)
GCQ1=GCQ1.tolist()
GCQ2=GCQ2.tolist()
GCQ3=GCQ3.tolist()
it = range(max([min([len(Q1),len(Q2),len(Q3)]),min([len(GQ1),len(GQ2),len(GQ3)]),min([len(GCQ1),len(GCQ2),len(GCQ3)])]))
for i in it:
    if i>len(Q1):
        Q1.append(Q1[-1])
    if i>len(Q2):
        Q2.append(Q2[-1])
    if i>len(Q3):
        Q3.append(Q3[-1])

    if i>len(GQ1):
        GQ1.append(GQ1[-1])
    if i>len(GQ2):
        GQ2.append(GQ2[-1])
    if i>len(GQ3):
        GQ3.append(GQ3[-1])

    if i>len(GCQ1):
        GCQ1.append(GCQ1[-1])
    if i>len(GCQ2):
        GCQ2.append(GCQ2[-1])
    if i>len(GCQ3):
        GCQ3.append(GCQ3[-1])

meanQ=[(Q1[i]+Q2[i]+Q3[i])/3 for i in range(min(len(Q1),len(Q2),len(Q3)))]
maxQ=[max([Q1[i],Q2[i],Q3[i]]) for i in range(min(len(Q1),len(Q2),len(Q3)))]
minQ=[min([Q1[i],Q2[i],Q3[i]]) for i in range(min(len(Q1),len(Q2),len(Q3)))]

meanGQ=[(GQ1[i]+GQ2[i]+GQ3[i])/3 for i in range(min(len(GQ1),len(GQ2),len(GQ3)))]
maxGQ=[max([GQ1[i],GQ2[i],GQ3[i]]) for i in range(min(len(GQ1),len(GQ2),len(GQ3)))]
minGQ=[min([GQ1[i],GQ2[i],GQ3[i]]) for i in range(min(len(GQ1),len(GQ2),len(GQ3)))]

meanGCQ=[(GCQ1[i]+GCQ2[i]+GCQ3[i])/3 for i in range(min(len(GCQ1),len(GCQ2),len(GCQ3)))]
maxGCQ=[max([GCQ1[i],GCQ2[i],GCQ3[i]]) for i in range(min(len(GCQ1),len(GCQ2),len(GCQ3)))]
minGCQ=[min([GCQ1[i],GCQ2[i],GCQ3[i]]) for i in range(min(len(GCQ1),len(GCQ2),len(GCQ3)))]

f, x = plt.subplots(1,1)
x.plot(range(0,len(meanQ)), meanQ, color='red',label='Q')
x.plot(range(0,len(meanGQ)),meanGQ,color='blue',label='GQ')
x.plot(range(0,len(meanGCQ)),meanGCQ,color='green',label='GCQ')
# # r1 = list(map(lambda x: x[0]-x[1], zip(returnavg, returnstd)))
# # r2 = list(map(lambda x: x[0]+x[1], zip(returnavg, returnstd)))
x.fill_between(range(0,len(meanQ)), maxQ, minQ, color='red', alpha=0.2)
x.fill_between(range(0,len(meanGQ)),maxGQ,minGQ, color='blue',alpha=0.2)
x.fill_between(range(0,len(meanGCQ)),maxGCQ,minGCQ, color='green',alpha=0.2)
x.legend()
x.set_xlabel('step')
x.set_ylabel('rewards')
# # exp_dir = 'Plot/'
# # if not os.path.exists(exp_dir):
# #     os.makedirs(exp_dir, exist_ok=True)
f.savefig('./RESULT/Q/Q.png', dpi=1000)
