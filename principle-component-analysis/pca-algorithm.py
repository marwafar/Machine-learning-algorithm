import numpy as np
from scipy.io import loadmat
from numpy import linalg as LA
#------------------------------
def covariance_matrix(X):
    n_data,n_features=X.shape
    sigma=X.T.dot(X)/n_data
    
    return sigma
#----------------------------
def diag(M):
    E,V=LA.eigh(M)

    return E,V
#----------------------------
def new_coord(X,V):
    coord=X.dot(V)
    return coord
#-----------------------------
def recover(Z,V):
    x_approx=Z.dot(V.T)
    return x_approx
#------------------------------
# Test
#df=loadmat('ex7data1.mat')

#n_data,n_features=df['X'].shape
#X_norm=np.zeros((n_data,n_features))
#mu=np.mean(df['X'],axis=0)
#std=np.std(df['X'],axis=0)

#X_norm[:,0]=(df['X'][:,0]-mu[0])/std[0]
#X_norm[:,1]=(df['X'][:,1]-mu[1])/std[1]

#sigma=covariance_matrix(X_norm)
#diag,vect=diag(sigma)
#print(vect)
#k=1
#z_coord=new_coord(X_norm,vect[:,1])
#print(z_coord)
#project_data=open('reduced_coord.csv',"w+")
#project_data.write(",".join(z_coord.astype(str)))
#x_rec=np.outer(z_coord,vect[:,1])
#print(x_rec)
#print("\n")
#print(X_norm)
#-----------------------------------------------
# PCA on faces: reduction of image dimensionality
#-------------------------------------------------
faces=loadmat('ex7faces.mat')
# Normalize dataset
#mu=np.mean(faces['X'],axis=0)
#print(mu)
#std=np.std(faces['X'],axis=0)
#X_norm=(faces['X']-mu)/std
X_norm=(faces['X']-faces['X'].mean())/faces['X'].std()
#print(X_norm[1,100:300])
#------------------------------
sigma=covariance_matrix(X_norm)
#print('covariance matrix', sigma.shape)
diag,vect=diag(sigma)
#print(diag)
#print('pca eigenvector', vect.shape)
z_coord=new_coord(X_norm,vect[:,924:])
#print('projected coordinate', z_coord.shape)
x_approx=recover(z_coord,vect[:,924:])
#print('approximated coordinate', x_approx.shape)
check_k=diag[924:].sum()/diag.sum()
print(check_k, 'variance is retained')

vect_file=open("pca-vect.csv","w+")
vect_file.write(",".join(vect[:,1020].astype(str)))
vect_file.close()

approx_file=open("approx-face.csv", 'w+')
approx_file.write(",".join(x_approx[30,:].astype(str)))
approx_file.close()