from scipy.io import loadmat
import numpy as np
#-----------------------------
def poly_feature(x_t,degree,n_train):

    x=np.zeros((n_train,degree))
    for i in range(degree):
        x[:,i]=np.power(x_t,i)
    return x
#-----------------------------
def hypothesis(x,theta):
    h=x.dot(theta)
    return h
#----------------------------
def cost(x,y,theta,lambda_x):

    n_dataset,nfeature=x.shape
    h_x=hypothesis(x,theta)

    cost_val=(((h_x-y[:,0])**2).sum())/(2.0*n_dataset)+\
        (((theta[1:])**2).sum())*lambda_x/(2.0*n_dataset)

    return cost_val
#----------------------------
def derivative(x,y,theta,lambda_x):

    n_dataset,feature=x.shape
    h_x=hypothesis(x,theta)

    derv=(x.T.dot(h_x-y[:,0]))/n_dataset
    derv[1:]+=lambda_x/n_dataset*theta[1:]

    return derv
#------------------------------
def gradient_descent(x,y,theta,lambda_x,steps,alpha):

    #cost_file=open("cost_val_poly.csv","w+")
    for i in range(steps):
        h_x=hypothesis(x,theta)
        cost_val=cost(x,y,theta,lambda_x)
        #cost_file.write(str(i)+","+str(cost_val)+"\n")
        derv=derivative(x,y,theta,lambda_x)

        theta=theta-alpha*derv

    return theta
#-----------------------------
df=loadmat('ex5data1.mat')

y=df['y']
data_set,feature=y.shape
x_1=df['X'][:,0]

degree=9
x=poly_feature(x_1,degree,data_set)

mu=np.mean(x,axis=0)
x[:,1:]=x[:,1:]-mu[1:]
sigma=np.std(x,axis=0)
x[:,1:]=x[:,1:]/sigma[1:]

data_training,feature_training=x.shape
theta=np.zeros(feature_training)
lambda_x=1.0
steps=50000
alpha=0.1

theta_file=open("theta-poly.csv","w+")
theta=gradient_descent(x,y,theta,lambda_x,steps,alpha)
#theta[1:]=(theta[1:]*sigma[1:])+mu[1:]
theta_file.write(",".join(theta.astype(str)))
theta_file.close()

y_val=df['yval']
x_1_val=df['Xval'][:,0]
data_val,f=y_val.shape

x_val=poly_feature(x_1_val,degree,data_val)
mu=np.mean(x_val,axis=0)
x_val[:,1:]=x_val[:,1:]-mu[1:]
sigma=np.std(x_val,axis=0)
x_val[:,1:]=x_val[:,1:]/sigma[1:]

#learn_curve=open("learn_curve_poly.csv","w+")
#for i in range(0,data_set,2):
#    theta=gradient_descent(x[:i+2],y[:i+2],theta,lambda_x,steps,alpha)
    
    # compute error for learning curve
#    error_train=cost(x[:i+2],y[:i+2],theta,0.0)
#    error_val=cost(x_val,y_val,theta,0.0)
#    learn_curve.write(str(i+2)+","+str(error_train)+","+str(error_val)+"\n")
#learn_curve.close()

lambda_set=np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
n=len(lambda_set)
select_lambda=open("select_lambda.csv","w+")
for i in range(n):
    lambda_x=lambda_set[i]
    theta=gradient_descent(x,y,theta,lambda_x,steps,alpha) 
    error_train=cost(x,y,theta,0.0)
    error_val=cost(x_val,y_val,theta,0.0) 
    select_lambda.write(str(lambda_x)+","+str(error_train)+","+str(error_val)+"\n")
select_lambda.close()