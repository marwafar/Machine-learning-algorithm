from scipy.io import loadmat
import numpy as np

#---------------------------
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

    cost_file=open("cost_val.csv","w+")
    for i in range(steps):
        h_x=hypothesis(x,theta)
        cost_val=cost(x,y,theta,lambda_x)
        cost_file.write(str(i)+","+str(cost_val)+"\n")
        derv=derivative(x,y,theta,lambda_x)

        theta=theta-alpha*derv

    return theta
#----------------------------
df=loadmat('ex5data1.mat')

y=df['y']
data_set,feature=y.shape
x_0=np.ones(data_set)
x=np.insert(df['X'],0,x_0, axis=1)
theta=np.ones(feature+1)
lambda_x=0.0
steps=3000
alpha=0.001

y_val=df['yval']
data_cross,f=y_val.shape
x_0=np.ones(data_cross)
x_val=np.insert(df['Xval'],0,x_0,axis=1)

#theta_file=open("data_theta.csv","w+")
learn_curve=open("learn_curve.csv","w+")
for i in range(0,data_set,2):
    theta=gradient_descent(x[:i+2],y[:i+2],theta,lambda_x,steps,alpha)
    #theta_file.write(str(i)+","+",".join(theta.astype(str))+"\n")
    # compute error for learning curve
    error_train=cost(x[:i+2],y[:i+2],theta,0.0)
    error_val=cost(x_val,y_val,theta,0.0)
    learn_curve.write(str(i+2)+","+str(error_train)+","+str(error_val)+"\n")







