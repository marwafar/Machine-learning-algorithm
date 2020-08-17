#%%
from matplotlib import pyplot as plt
import numpy as np
from scipy.io import loadmat
import matplotlib.image as mpimg
import cv2
#---------------------
# %%
df=loadmat('ex7data2.mat')
#print(df)

# %%
plt.plot(df['X'][:,0],df['X'][:,1],marker='o',linestyle='')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

# %%
img=mpimg.imread('bird_small.png')
plt.imshow(img)

#%%
img_3D=cv2.imread('bird_small.png')
print(img_3D.shape)

# %%
centroids=np.genfromtxt('centroid-rand0.csv',delimiter=',')
#print(centroids.shape)
centroids=centroids.reshape(16,3)
print(centroids.shape)


# %%
n,m,l=img_3D.shape
img_2D=np.reshape(img_3D,(n*m,l))

n_pixel,n_feature=img_2D.shape
y=np.zeros(n_pixel)
for i in range(n_pixel):
    c=(((img_2D[i]/255-centroids)**2)).sum(axis=1)
    y[i]=np.argmin(c)

X_recover=np.zeros((n_pixel,n_feature))
for i in range(n_pixel):
    X_recover[i,:]=centroids[int(y[i]),:]

X_recover=X_recover.reshape(n,m,l)


# %%
img=mpimg.imread('bird_small.png')
plt.imshow(img)

# %%
plt.imshow(X_recover)
# %%
