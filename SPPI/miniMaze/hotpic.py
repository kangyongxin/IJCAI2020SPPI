import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

#a = np.random.uniform(0, 1, size=(10, 10))
p = np.array([[87736,54207,0,1833,5893,3608],
              [61835,54197,0,0,12068,5335],
              [38465,44728,32464,26143,22299,0],
              [0,0,0,0,13831,6288],
              [465,1247,2173,4186,8349,5886],
              [356,1201,2004,3429,4434,3764]])

af= pd.read_csv("./baselines/SPPI/data/r_AF.csv")

aflist= af.values.tolist()
afarray= np.zeros([6,6])
print(afarray)
for i in range(6):
    for j in range(6):
        afarray[i,j]= aflist[i*6+j][1]
sns.heatmap(afarray, cmap='Reds')
plt.show()
