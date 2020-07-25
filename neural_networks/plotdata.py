# %%
from scipy.io import loadmat
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

df=loadmat('ex3data1.mat')
X=df['X']
#Y=df['y']
#X=np.array(df.get('X').astype(np.float64))
#print(X[0])

# %%
#ax=plt.subplot()
#hi=np.max(X)
#lo=np.min(X)
#im=(((X-lo)/(hi-lo))*255)
plt.figure(figsize=(5,5))
imag1=X[1900].reshape(20,20)
plt.imshow(imag1.T,cmap='gray')
plt.colorbar()
plt.show()

# %%
cost=np.genfromtxt('cost_file_ex1.csv',delimiter=',')
plt.plot(cost[:,0],cost[:,1])
plt.show()

# %%
