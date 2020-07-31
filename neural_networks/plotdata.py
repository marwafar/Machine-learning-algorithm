# %%
from scipy.io import loadmat
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random

df=loadmat('ex3data1.mat')
X=df['X']

# %%
plt.figure(figsize=(5,5))
imag1=X[200].reshape(20,20)
plt.imshow(imag1.T,cmap='gray')
plt.colorbar()
plt.show()

# %%
cost=np.genfromtxt('cost_file_ex1.csv',delimiter=',')
plt.plot(cost[:,0],cost[:,1])
plt.show()

# %%
cost=np.genfromtxt('./backpropagation-algorithm/cost_file_ex4.csv',delimiter=',')
plt.plot(cost[:,0],cost[:,1])
plt.show()

# %%
