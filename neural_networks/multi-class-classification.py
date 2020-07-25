from scipy.io import loadmat
import numpy as np
#-----------------------------
def sigmoid(X,theta):

    z=X.dot(theta.T)
    hypothesis=1.0/(1.0+np.exp(-z))

    return hypothesis
#--------------------------------
def cost_func(X,theta,Y,lambda_c):

    n_dataset,n_features=X.shape
    hypothesis=sigmoid(X,theta)

    cost=(1.0/n_dataset)*\
        (-Y.dot(np.log(hypothesis))-\
            (1.0-Y).dot(np.log(1.0-hypothesis)))+lambda_c/(2*n_dataset)*\
                (theta[1:]*theta[1:]).sum()
    
    #print(cost)
    return cost
#-----------------------------------------
def gradient_descent(X,theta,Y,steps,alpha,lambda_c,i):

    n_dataset,n_features=X.shape

    if i==3:
        cost_file = open('cost_file_ex1.csv','w+')

    for step in range(steps):
        hypothesis=sigmoid(X,theta) 
        cost=cost_func(X,theta,Y,lambda_c)

        if i==3:
            cost_file.write(str(step)+","+str(cost)+"\n")

        derv=(1.0/n_dataset)*X.T.dot(hypothesis-Y)
        derv[1:]+=(lambda_c/n_dataset)*theta[1:]
        theta=theta-alpha*derv
    if i==3:
        cost_file.close()
    return theta
#-------------------------------
def one_vs_all(classes,X,Y,steps,alpha,lambda_c):

    n_dataset,n_features=X.shape 
    Y_class=np.zeros((classes,n_dataset))

    theta=np.zeros((classes,n_features))

    for i in range(classes):
        Y_class[i]=(Y==i+1).astype(float)
    
    #y_file=open('y_class.csv','w+')
    #for i in range(classes):
    #    y_file.write(",".join(Y_class[i].astype(str)) +"\n") 
    #y_file.close()

    for i in range(classes):
        theta[i]=gradient_descent(X,theta[i],Y_class[i],steps,alpha,lambda_c,i)
    
    return theta
#---------------------------------
df=loadmat('ex3data1.mat')

n_row,n_colm=df['X'].shape
X_0=np.ones(n_row)
Y=df['y'].flatten()
#X=df['X']
X=np.insert(df['X'],0,X_0, axis=1)
lambda_c=1.0
steps=6000
alpha=0.01
classes=10

theta=one_vs_all(classes,X,Y,steps,alpha,lambda_c)

t_file=open('theta_ex1.csv','w+')

for i in range(classes):
    t_file.write(",".join(theta[i].astype(str)) +"\n")

t_file.close()
#------------------------------
#TEST
#-------------------------
#print(theta[:20])

#print(X.shape)
#print(X[1])
#print(df['x_0'])
#print(X.shape)
#print(n_row,n_colm)
#print(Y.shape)
#print(df.items())
#print(df['X'][0].reshape(20,20))