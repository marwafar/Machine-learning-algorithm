from scipy.io import loadmat
import numpy as np
#------------------------------
def sigmoid(X,theta):
    z=X.dot(theta.T)
    hypothesis=1.0/(1.0+np.exp(-z))

    return hypothesis
#-------------------------------
def forward_propag(theta_1,theta_2,a_1,layer_2,layer_3):

    n_dataset,n_feature=a_1.shape

   #1- compute first hidden layer a_2[5000,26]
    a_2=np.zeros((n_dataset,layer_2))

    a_2[:,1:]=sigmoid(a_1,theta_1)
    a_2[:,0]=np.ones(n_dataset) 

    #2- compute output layer a_3[5000,10]
    a_3=np.zeros((n_dataset,layer_3))
    a_3=sigmoid(a_2,theta_2)

    return a_2,a_3
#-------------------------------
def cost(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k):

    n_dataset,n_feature=a_1.shape

    # Forward propagation:
    a_2,a_3=forward_propag(theta_1,theta_2,a_1,layer_2,layer_3)

    # Define the y-class for each output unit:
    y_class=np.zeros((n_dataset,layer_3))

    for i in range(n_dataset):
        y_class[i]=[float(Y[i]==j+1) for j in range(layer_3)]
    
    # compute the cost function.
    cost_val=((-y_class*np.log(a_3))-\
        (1.0-y_class)*np.log(1.0-a_3)).sum()/n_dataset+\
            (((theta_1[:,1:]*theta_1[:,1:])).sum()+\
                ((theta_2[:,1:]*theta_2[:,1:])).sum())*\
                    (lambda_k/(2.0*n_dataset))


    return cost_val
#-------------------------------
def sigmoid_gradient(X,theta):

    hypothesis=sigmoid(X,theta)
    hypothesis_derv=hypothesis*(1-hypothesis)

    return hypothesis_derv

#----------------------------------
def random_initialization(epsilon,l_in,l_out):
    theta=np.random.rand(l_out,l_in)*2.0*epsilon-epsilon
    return theta
#--------------------------------
def back_propag(theta_1,theta_2,a_1,layer_2,layer_3,lambda_k):

    n_dataset,n_feature=a_1.shape

    # Forward propagation:
    a_2,a_3=forward_propag(theta_1,theta_2,a_1,layer_2,layer_3)
    # Define the y-class for each output unit:
    y_class=np.zeros((n_dataset,layer_3))
    # define as classes with 0,1
    for i in range(n_dataset):
        y_class[i]=[float(Y[i]==j+1) for j in range(layer_3)]
    
    # back propagation 
    big_delta_1=np.zeros((layer_2-1,n_feature))
    big_delta_2=np.zeros((layer_3,layer_2))
    derv_1=np.zeros((layer_2-1,n_feature))
    derv_2=np.zeros((layer_3,layer_2))
    for i in range(n_dataset):
        delta_3=a_3[i]-y_class[i]
        delta_2=theta_2.T.dot(delta_3)*(a_2[i]*(1.0-a_2[i]))

        delta_3=delta_3.reshape(layer_3,1)
        delta_2=delta_2.reshape(layer_2,1)
        l_1=a_1[i,].reshape(n_feature,1)
        l_2=a_2[i,].reshape(layer_2,1)

        big_delta_1+=delta_2[1:,:].dot(l_1.T)
        big_delta_2+=delta_3.dot(l_2.T)
    
    # Compute the derivatives for each layer
    derv_1[:,0]=big_delta_1[:,0]/n_dataset
    derv_1[:,1:]=big_delta_1[:,1:]/n_dataset+theta_1[:,1:]*lambda_k/n_dataset
    derv_2[:,0]=big_delta_2[:,0]/n_dataset
    derv_2[:,1:]=big_delta_2[:,1:]/n_dataset+theta_2[:,1:]*lambda_k/n_dataset


    return derv_1,derv_2
#--------------------------------------------
def gradient_descent(steps,alpha,theta_1,theta_2,a_1,layer_2,layer_3,lambda_k,Y):

    cost_file = open('cost_file_ex4.csv','w+')
    for step in range(steps):
        cost_val=cost(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k)
        cost_file.write(str(step)+","+str(cost_val)+"\n")

        derv_1,derv_2=back_propag(theta_1,theta_2,a_1,layer_2,layer_3,lambda_k)

        theta_1=theta_1-alpha*derv_1
        theta_2=theta_2-alpha*derv_2

    cost_file.close()
    return theta_1,theta_2
#--------------------------------------------
def numerical_gradient_check(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k):
    #This is only to check the back propagation.

    n_dataset,n_feature=a_1.shape
    derv_1,derv_2=back_propag(theta_1,theta_2,a_1,layer_2,layer_3,lambda_k)

    cost_val_1_plus=np.zeros((layer_2-1,n_feature))
    cost_val_1_minus=np.zeros((layer_2-1,n_feature)) 
    cost_val_2_plus=np.zeros((layer_3,layer_2))
    cost_val_2_minus=np.zeros((layer_3,layer_2))  
    num_derv_1=np.zeros((layer_2-1,n_feature))
    num_derv_2=np.zeros((layer_3,layer_2)) 

    for row in range(layer_2-1):
        for colm in range(n_feature):
            temp=theta_1[row,colm]
            theta_1[row,colm]=temp+0.0001
            cost_val_1_plus[row,colm]=cost(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k)
            theta_1[row,colm]=temp-0.0001
            cost_val_1_minus[row,colm]=cost(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k)
            theta_1[row,colm]=temp

            num_derv_1[row,colm]=(cost_val_1_plus[row,colm]-cost_val_1_minus[row,colm])/(2.0*0.0001)
        #    print(num_derv_1[row,colm]-derv_1[row,colm])

    for row in range(layer_3):
        for colm in range(layer_2):
            temp=theta_2[row,colm]
            theta_2[row,colm]=temp+0.0001
            cost_val_2_plus[row,colm]=cost(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k)
            theta_2[row,colm]=temp-0.0001
            cost_val_2_minus[row,colm]=cost(theta_1,theta_2,a_1,layer_2,layer_3,Y,lambda_k)
            theta_2[row,colm]=temp

            num_derv_2[row,colm]=(cost_val_2_plus[row,colm]-cost_val_2_minus[row,colm])/(2.0*0.0001)
        #    print(num_derv_2[row,colm]-derv_2[row,colm])

    # compute the difference between the two derivavtives.
    diff_1= num_derv_1-derv_1
    diff_2= num_derv_2-derv_1

    return diff_1,diff_2
#--------------------------------
df=loadmat('ex4data1.mat')

n_dataset,n_feature=df['X'].shape
X_0=np.ones(n_dataset)

a_1=np.insert(df['X'],0,X_0, axis=1)
Y=df['y'].flatten()

layer_2=26
layer_3=10

#weight=loadmat('ex4weights.mat')
#theta_1=weight['Theta1']
#theta_2=weight['Theta2']
lambda_k=1.0
steps=1000
alpha=0.1

theta_1=random_initialization(0.12,n_feature+1,layer_2-1)
theta_2=random_initialization(0.4,layer_2,layer_3)
#print(t_1.shape)
#print(t_2.shape)
theta_1,theta_2=gradient_descent(steps,alpha,theta_1,theta_2,a_1,layer_2,layer_3,lambda_k,Y)

# check accuracy:

a_2,a_3=forward_propag(theta_1,theta_2,a_1,layer_2,layer_3)

j=0
count=0
for i in range(layer_3):
    j+=1
    predict_yes=[el for el in range(n_dataset) if a_3[el,i]>=0.5 and Y[el]==j]
    predict_no=[el for el in range(n_dataset) if a_3[el,i]<0.5 and Y[el]!=j]

    count=(len(predict_yes)+len(predict_no))/float(n_dataset)
    
    print(count)
    count=0
    







#------------------------
#print(df['X'].shape)
#print(df['y'].shape)
#print(a_1.shape)
#print(Y.shape)

#print(theta_1.shape)
#print(theta_2.shape)

