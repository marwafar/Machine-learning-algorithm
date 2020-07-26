from scipy.io import loadmat
import numpy as np
#-----------------------------
def sigmoid(X,theta):

    z=X.dot(theta.T)
    hypothesis=1.0/(1.0+np.exp(-z))

    return hypothesis
#--------------------------------

df=loadmat('ex3data1.mat')
n_row,n_colm=df['X'].shape
X_0=np.ones(n_row)
Y=df['y'].flatten()
X=np.insert(df['X'],0,X_0, axis=1)

n_data,n_feature=X.shape

theta=loadmat('ex3weights.mat')
#print(theta)
#print(theta['Theta2'].shape)
#print(theta['Theta1'].shape)

theta_1=theta['Theta1']
theta_2=theta['Theta2']

n1_layer=26
a_1=np.zeros((n_data,n1_layer))

a_1[:,1:]=sigmoid(X,theta_1)
a_1[:,0]=np.ones(n_data)

n2_layer=10
a_2=np.zeros((n_data,n2_layer))

a_2=sigmoid(a_1,theta_2)

#---------------------------------
print(a_2[200,:])
#------------------------
#accuracy

j=0
count=0
for i in range(n2_layer):
    j+=1
    predict_yes=[el for el in range(n_data) if a_2[el,i]>=0.5 and Y[el]==j]
    predict_no=[el for el in range(n_data) if a_2[el,i]<0.5 and Y[el]!=j]

    count=(len(predict_yes)+len(predict_no))/float(n_data)
    #print(len(predict_yes),len(predict_no),count/float(n_data))
    print(count)
    count=0
    










