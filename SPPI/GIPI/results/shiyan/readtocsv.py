import numpy as np
import pandas as pd

a2cgipi = np.load('./a2c+gipi/mean_r.npy')
#print(len(c))
A2CGIPI = pd.DataFrame(a2cgipi)
A2CGIPI.to_csv('A2CGIPI.csv')

print('done')

a2cgipisil = np.load('./a2c+gipi+sil/mean_r.npy')
#print(len(c))
A2CGIPISIL = pd.DataFrame(a2cgipisil)
A2CGIPISIL.to_csv('A2CGIPISIL.csv')

print('done')

a2csil = np.load('./a2c+sil/mean_r.npy')
#print(len(c))
A2CSIL = pd.DataFrame(a2csil)
A2CSIL.to_csv('A2CSIL.csv')

print('done')

a2csil1 = np.load('./a2c+sil/mean_r1.npy')
#print(len(c))
A2CSIL1 = pd.DataFrame(a2csil1)
A2CSIL1.to_csv('A2CSIL1.csv')

print('done')

a2csil2 = np.load('./a2c+sil/mean_r2.npy')
#print(len(c))
A2CSIL2 = pd.DataFrame(a2csil2)
A2CSIL2.to_csv('A2CSIL2.csv')

print('done')

dqngipi = np.load('./DQN+GIPI/mean_r1.npy')
#print(len(c))
DQNGIPI = pd.DataFrame(dqngipi)
DQNGIPI.to_csv('DQNGIPI.csv')

print('done')

a2c = np.load('./A2C/mean_r.npy')
#print(len(c))
A2C = pd.DataFrame(a2c)
A2C.to_csv('A2C.csv')

print('done')