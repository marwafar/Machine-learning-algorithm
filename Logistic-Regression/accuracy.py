import pandas as pd
import numpy as np
#-----------------------
def sigmoid(x,theta):
    z=np.zeros(len(x))
    hypothesis=np.zeros(len(x))

    for i in range(len(x)):
        z[i]=np.dot(theta.T,x[i])
        hypothesis[i]=1.0/(1.0+np.exp(-z[i]))

    return hypothesis
#-------------------------
if __name__ == "__main__":

    theta=np.array([-8.05308516,0.07019601,0.06375572])
    
    df1=pd.read_csv("ex2data1.csv")
    df1['x0']=1.0

    #Defin features x as matrix 
    x_features=[]
    for i in range(len(df1)):
        temp=[df1.x0[i],df1.Exam_1_score[i],df1.Exam_2_score[i]]
        x_features.append(temp)
    
    hypothesis=sigmoid(x_features,theta)

    yes_p=[i for i in range(len(hypothesis)) if hypothesis[i]>=0.5 and df1.y[i]==1.0]
    no_p=[i for i in range(len(hypothesis)) if hypothesis[i]<0.5 and df1.y[i]==0.0]

    yes_actual=df1[df1.y==1]
    no_actual=df1[df1.y==0]

    #print(len(yes_actual),len(yes_p))
    #print(len(no_actual),len(no_p))

    accuracy=(len(yes_p)+len(no_p))/len(df1)
    print("The accuracy of the model is", accuracy*100, "%")

