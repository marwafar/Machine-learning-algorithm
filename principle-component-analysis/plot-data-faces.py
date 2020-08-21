#%%
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

# %%
df=loadmat('ex7faces.mat')
print(df['X'].shape)

# %%

face=df['X'][30].reshape(32,32)
plt.imshow(face.T, cmap='gray', interpolation='spline16')
plt.show()

# %%
pca=np.genfromtxt('pca-vect.csv', delimiter=',')
pca=pca.reshape(32,32)
plt.imshow(pca.T, cmap='gray',interpolation='spline36')
plt.show()

# %%
approx_face=np.genfromtxt('approx-face.csv',delimiter=',')
approx_face=approx_face.reshape(32,32)
plt.imshow(approx_face.T, cmap='gray',interpolation='spline36')
plt.show()

# %%
