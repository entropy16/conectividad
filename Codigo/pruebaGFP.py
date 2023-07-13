import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1,3,4,2],[1,1,0,2],[2,3,2,1],[1,1,1,1]])

N=4
Vmean=np.mean(data[:,1:],axis=1)
print(Vmean.shape)
GFP=np.zeros(data.shape[0])
for i in range(1,N):
    GFP=GFP+(data[:,i]-Vmean)**2
print(Vmean)
GFP=np.sqrt(GFP/N)
plt.plot(data[:,0],GFP)

plt.show()