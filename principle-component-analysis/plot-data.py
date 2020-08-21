# %%
from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np

#-----------------------
# %%
df=loadmat('ex7data1.mat')
print(df['X'])


# %%
plt.plot(df['X'][:,0],df['X'][:,1],marker='o',linestyle='')
plt.show()
# %%
n_data,n_features=df['X'].shape
X_norm=np.zeros((n_data,n_features))
mu=np.mean(df['X'],axis=0)
std=np.std(df['X'],axis=0)

X_norm[:,0]=(df['X'][:,0]-mu[0])/std[0]
X_norm[:,1]=(df['X'][:,1]-mu[1])/std[1]


# %%
plt.plot(X_norm[:,0],X_norm[:,1],marker='o',linestyle='')
plt.show()

# %%
z=np.genfromtxt('reduced_coord.csv',delimiter=',')
#print(z.shape,n_data)

# %%
plt.plot(X_norm[:,0],X_norm[:,1],marker='o',linestyle='')
plt.plot(z[:],z[:],marker='o',linestyle='')
plt.show()

# %%
