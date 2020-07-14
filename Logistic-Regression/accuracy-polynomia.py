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
#----------------------------
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

    df_theta=pd.read_csv("theta_features.csv")
    theta=df_theta.iloc[0,:]
    #print(theta)

    hypothesis=sigmoid(x,theta,df,total_features)

    yes_p=[i for i in range(len(hypothesis)) if hypothesis[i]>=0.5 and df.y[i]==1.0]
    no_p=[i for i in range(len(hypothesis)) if hypothesis[i]<0.5 and df.y[i]==0.0]

    yes_actual=df[df.y==1]
    no_actual=df[df.y==0]

    print(len(yes_actual),len(yes_p))
    print(len(no_actual),len(no_p))

    accuracy=(len(yes_p)+len(no_p))/len(df)
    print("The accuracy of the model is", accuracy*100, "%")
    
