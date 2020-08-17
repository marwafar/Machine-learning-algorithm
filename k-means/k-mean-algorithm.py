from scipy.io import loadmat
import numpy as np
import random
import cv2
#----------------------------
def find_closest_centroids(x,centroids):

    n_data,n_features=x.shape
    y=np.zeros(n_data)

    for i in range(n_data):
        c=(((x[i]-centroids)**2)).sum(axis=1)
        y[i]=np.argmin(c)
        
    return y
#-----------------------------
def compute_mean_centroids(y,x,k_centroids):
    n_data,n_feature=x.shape
    centroids=np.zeros((k_centroids,n_feature))

    for j in range(k_centroids):
        centroids[j,:]=np.mean([x[i] for i in range(n_data) if y[i]==j], axis=0)
    
    return centroids
#-------------------------------
def optimization_objective(x,y,centroids):
    n_data,n_features=x.shape
    cost=0
    for i in range(n_data):
        cost+=((x[i]-centroids[int(y[i])])**2).sum()

    return cost/n_data
#-----------------------------
def initialize_centroids(x,n_data,k):
    rand_idx=random.sample(range(n_data),k)
    centroids=x[rand_idx[:],:]
    return centroids
#--------------------------------------------------
if __name__ == "__main__":
    df=loadmat('ex7data2.mat')
    #print(df)
    n_data,n_features=df['X'].shape
    #-----------------------------------------------
    # Initialize the centroids to be random example
    # from the training set.

    k_centroids=3
    #centroids=initialize_centroids(df['X'],n_data,k_centroids)
    #print(centroids)

    #centroids=np.array([[5.69797866,2.94977132],\
    #    [1.30882588,5.30158701],[3.85384314,0.7920479]])
    centroids=np.array([[3.0,3.0],[6.0,2.0],[8.0,5.0]])
    #------------------------------------------
    # Run K-means
    n_steps=10
    for i in range(n_steps):
        # Assign data points to the closest centroid.
        y=find_closest_centroids(df['X'],centroids)
        # Compute the mean based on centroid assignment and move centroids.
        centroids=compute_mean_centroids(y,df['X'],k_centroids)
        # Compute the cost
        cost=optimization_objective(df['X'],y,centroids)

    #print(centroids)
    #print(cost)
    #print(y)
#--------------------------------------------------
    # Image compression with k-means

    img_3D=cv2.imread('bird_small.png')
    img_3D=img_3D/255
    n,m,l=img_3D.shape
    img_2D=np.reshape(img_3D,(n*m,l))
    #print(img_2D.shape)
    n_pixel,n_color=img_2D.shape
#=---------------------------------------------------------
    n_random=5
    n_steps=100
    for i in range(n_random):
        print('Random set: ',i)
        # Initialize the centroids to be random example
        # from the training set.
        k_centroids=16
        centroids=initialize_centroids(img_2D,n_pixel,k_centroids)
        #print(centroids) 

        cost_file=open('cost-random%i.csv' %i, 'w+')
        centroid_file=open('centroid-rand%i.csv' %i, 'w+')
        #Run k-means
        for i in range(n_steps):
            # Assign data points to the closest centroid.
            y=find_closest_centroids(img_2D,centroids)
            # Compute the mean based on centroid assignment and move centroids.
            centroids=compute_mean_centroids(y,img_2D,k_centroids)
            # Compute the cost function.
            cost=optimization_objective(img_2D,y,centroids)
            cost_file.write(str(i)+ ","+str(cost)+"\n")
        centroid_file.write(",".join((np.reshape(centroids,(-1))).astype(str)))
    
    #print(centroids)
    cost_file.close()
    centroid_file.close()
        





