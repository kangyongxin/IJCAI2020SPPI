#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:31:56 2019

@author: zhao
"""

import numpy as np
import matplotlib.pyplot as plt

Y=np.load('/home/zhao/桌面/shiyan/QHER/scores2.npy')
Y1=np.load('/home/zhao/桌面/shiyan/Q/scores1.npy')
Y2=np.load('/home/zhao/桌面/shiyan/QPER/scores1.npy')
K = 40
Z = Y.reshape(int(1000/K),K)
Z1 = Y1.reshape(int(1000/K),K)
Z2 = Y2.reshape(int(1000/K),K)
T = 1 - np.mean(Z,axis=1)
T1 = 1 - np.mean(Z1,axis=1)
T2 = 1 - np.mean(Z2,axis=1)
x = np.linspace(0, len(T), len(T))*K

plt.plot(x, 1-T)
plt.plot(x, 1-T1)
plt.plot(x, 1-T2)
plt.xlabel('episodes')
plt.ylabel('sucess rate')

ii=np.load('/home/zhao/桌面/shiyan/a2c+sil/mean_r.npy')
p=ii[0:640]
p[400:640]=400
w=np.load('/home/zhao/桌面/shiyan/a2c+sil+per/mean_r1.npy')
g=w[0:640]
g[220:640]=400
o=np.load('/home/zhao/桌面/shiyan/a2c+per/mean_r1.npy')
c=o[0:640]
K=10
Z = c.reshape(int(640/K),K)
T = np.mean(Z,axis=1)

K=10
Z1 = g.reshape(int(640/K),K)
T1 = np.mean(Z1,axis=1)
x = np.linspace(0, len(T), len(T))*K

plt.plot(x, T)
plt.plot(x, T1)
plt.plot(range(640), c)
plt.plot(range(640),g)
plt.plot(range(640),iu)
plt.plot(range(640),p)

tt=np.load('/home/zhao/桌面/shiyan/a2c+per/mean_r3.npy')
ert=tt[0:640]
iu=np.zeros(640)
iu[640-113:640]=ert[6:119]
iu[592]=30.26