import numpy as np
import pandas as pd
#-----------------
def sigmoid(x,theta):
    z=np.zeros(len(x))
    hypothesis=np.zeros(len(x))

    for i in range(len(x)):
        z[i]=np.dot(theta.T,x[i])
        hypothesis[i]=1.0/(1.0+np.exp(-z[i]))

    return hypothesis
#---------------------------
def cost_function(y,x,theta):
    hypothesis=sigmoid(x,theta)

    cost_theta=0.0
    for i in range(len(y)):
        A=y[i]*np.log(hypothesis[i])
        B=(1.0-y[i])*np.log(1.0-hypothesis[i])
        cost_theta+= -A-B
    cost_theta/=len(y)

    return cost_theta
#-------------------------------------
def gradient_descent(df1,y,x,theta,alpha,steps):

    derv=np.zeros(len(theta))
    temp=np.zeros(len(theta))
    cost_file=open("cost_function.csv","w+")
    cost_file.write("step" + ","+"cost_func"+"\n")

    for i in range(steps):
        hypothesis=sigmoid(x,theta)
        cost_theta=cost_function(y,x,theta)
        cost_file.write(str(i) +","+str(cost_theta)+"\n")

        k=0
        for colm in df1.columns:
            if colm !='y':
                D=np.subtract(hypothesis,y)
                derv[k]=np.dot(D.T,df1[colm])
                k+=1
        
        for l in range(len(theta)):
            temp[l]=theta[l]-alpha/len(df1)*derv[l]
        
        for k in range(len(theta)):
            theta[k]=temp[k]
    
    cost_file.close()
    return theta
#-------------------------------------------------------
if __name__ == "__main__":
    df1=pd.read_csv("ex2data1.csv")
    #print(df1.columns)
    #print(df1.head())
    df1['x0']=1.0

    # Initailze the theta parameters 
    theta=np.zeros(3)
    #print(theta)

    #Defin features x as matrix 
    x_features=[]
    for i in range(len(df1)):
        temp=[df1.Exam_1_score[i],df1.Exam_2_score[i],df1.x0[i]]
        x_features.append(temp)
    #print(x_features[1])

    y=df1.y

    # Define the step for gradient descent
    alpha=0.001
    steps=230000

    hypothesis=sigmoid(x_features,theta)
    cost_theta=cost_function(y,x_features,theta)
    theta=gradient_descent(df1,y,x_features,theta,alpha,steps)

    # Note: theta_[0] corresponds to theta_1 
    # and theta[n] corresponds to theta_0
    print(theta)
    





    