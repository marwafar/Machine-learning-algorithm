import pandas as pd
import numpy as np
#------------------------
def sigmoid(x,theta,df,total_features):
    z=np.zeros(len(df))
    hypothesis=np.zeros(len(df))

    for i in range(len(df)):
        for j in range(total_features):
            z[i]+=theta[j]*x[j,i]
        hypothesis[i]=1.0/(1.0+np.exp(-z[i]))

    return hypothesis
#------------------------------------------------
def cost_function(y,x,theta,df,total_features,lambda_1):

    hypothesis=sigmoid(x,theta,df,total_features)

    cost_theta=0.0
    for i in range(len(y)):
        A=y[i]*np.log(hypothesis[i])
        B=(1.0-y[i])*np.log(1.0-hypothesis[i])
        cost_theta+= -A-B
    cost_theta/=len(y)

    reg=0.0
    for i in range(1,total_features):
        reg+=theta[i]*theta[i]
    reg*=(lambda_1/(2.0*len(df)))
    cost_theta+=reg

    return cost_theta
#--------------------------------------------
def gradient_descent(df,y,x,theta,alpha,steps,total_features,lambda_1):

    derv=np.zeros(len(theta))
    temp=np.zeros(len(theta))
    cost_file=open("cost_function_polynomial.csv","w+")
    cost_file.write("step" + ","+"cost_func"+"\n")

    for i in range(steps):
        hypothesis=sigmoid(x,theta,df,total_features)
        cost_theta=cost_function(y,x,theta,df,total_features,lambda_1)
        cost_file.write(str(i) +","+str(cost_theta)+"\n")

        for i in range(total_features):
                D=np.subtract(hypothesis,y)
                if i==0:
                    derv[i]=np.dot(D,x[i])
                else:
                    derv[i]=np.dot(D,x[i])+lambda_1*theta[i]
        
        for l in range(len(theta)):
            temp[l]=theta[l]-alpha/len(df)*derv[l]
        
        for k in range(len(theta)):
            theta[k]=temp[k]
    
    cost_file.close()
    return theta
#--------------------------------------
if __name__ == "__main__":
    df=pd.read_csv("ex2data2.csv")
    #print(df.head())
    x1=df.Test_1
    x2=df.Test_2
    df['x0']=1.0
    y=df.y
    #print(y)

    degree=7
    total_features=28
    x=np.zeros((total_features,len(df)))
    x[0]=df.x0
    #print(x[0])
    index=1
    for i in range(1,degree):
        for j in range (0,i+1):
            k=i-j
            x[index]=np.power(x1,k)*np.power(x2,j)
            index+=1

    theta=np.zeros(total_features)
    #print(x[1,0],x1[0],x[0,1])

    hypothesis=sigmoid(x,theta,df,total_features)
    #print(hypothesis)
    lambda_1=1.0
    cost_func=cost_function(y,x,theta,df,total_features,lambda_1)
    #print(cost_func)

    alpha=0.01
    steps=30000
    theta=gradient_descent(df,y,x,theta,alpha,steps,total_features,lambda_1)
    theta_file=open("theta_features_polynomial.csv","w+")
    colm=np.array(range(total_features))
    theta_file.write(",".join(colm.astype(str))+"\n")
    theta_file.write(",".join(theta.astype(str)))
