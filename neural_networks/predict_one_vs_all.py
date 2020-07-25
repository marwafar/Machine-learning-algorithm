import numpy as np
from scipy.io import loadmat
#------------------------------
def sigmoid(X,theta):

    z=X.dot(theta.T)
    hypothesis=1.0/(1.0+np.exp(-z))

    return hypothesis
#---------------------------
df=loadmat('ex3data1.mat')
theta=np.genfromtxt('theta_ex1.csv',delimiter=',')

n_row,n_colm=df['X'].shape
X_0=np.ones(n_row)
Y=df['y'].flatten()
X=np.insert(df['X'],0,X_0, axis=1)

classes=10
h_theta=np.zeros((classes,n_row))
Y_class=np.zeros((classes,n_row))

for i in range(classes):
    Y_class[i]=(Y==i+1).astype(float)
    h_theta[i]=sigmoid(X,theta[i])

print(h_theta[:,1900])

for i in range(classes):
    actual_yes=[num for num in Y_class[i] if num==1.0]
    actual_no=[num for num in Y_class[i] if num==0.0]
    predict_yes=[h_theta[i,j] for j in range(len(h_theta[i])) if h_theta[i,j]>=0.5 and Y_class[i,j]==1.0]
    predict_no=[h_theta[i,j] for j in range(len(h_theta[i])) if h_theta[i,j]<0.5 and Y_class[i,j]==0.0]

    #print('yes ',len(actual_yes),len(predict_yes))
    #print('no ',len(actual_no),len(predict_no))

    accuracy=(len(predict_yes)+len(predict_no))/float(n_row)
    print(accuracy)







