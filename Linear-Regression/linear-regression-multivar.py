import numpy as np
import pandas as pd
#---------------------------------------------
def hypothesis(theta,x_values,data):
    fx=np.zeros(len(data))
    for i in range(len(data)):
        fx[i]=np.dot(theta.T,x_values[i])
    return fx
#----------------------------------------------
def cost_function(theta,x_values,y_values,data):
    
    cost_func=0.0
    fx=hypothesis(theta,x_values,data)
    for i in range(len(x_values)):
        cost_func+=(fx[i]-y_values[i])**2

    cost_func=cost_func/(2.0*len(x_values))

    return cost_func
#------------------------------
def gradient_descent(alpha,iteration,theta,data,x_values,y_values):

    cost_file=open("step-cost.csv","w+")
    cost_file.write("step"+","+"theta_1"+","+"theta_2"+","+"cost_func"+"\n")

    #derv_sum=np.zeros(len(theta)) 
    D=np.zeros(len(data))

    for step in range(iteration):
        fx=hypothesis(theta,x_values,data)
        derv_sum=np.zeros(len(theta))
        for i in range(len(data)):
            D[i]=fx[i]-y_values[i]
            
        derv_sum[0]=np.dot(data.x0.T,D)
        derv_sum[1]=np.dot(data.City_population.T,D)

        temp0=theta[0]-alpha/(len(data))*derv_sum[0]
        temp1=theta[1]-alpha/(len(data))*derv_sum[1]

        theta[0]=temp0
        theta[1]=temp1

        cost_func=cost_function(theta,x_values,y_values,data)
        cost_file.write(str(step)+","+str(theta[0])+","+str(theta[1])+","+str(cost_func)+"\n")
    cost_file.close()
    return theta
#----------------------------------
if __name__ == "__main__":

    data=pd.read_csv('ex1data1.csv')
    #print(data.head())
    #print(data.info())

    data['x0']=1
    y_values=data.Profit
    theta=np.array([0.0,0.0])
    x_values=[]
    for i in range(len(data)):
        temp=[data.x0[i],data.City_population[i]]
        x_values.append(temp)

    #print(cost_function(theta,x_values,y_values,data))
    alpha=0.01
    iteration=1500
    theta=gradient_descent(alpha,iteration,theta,data,x_values,y_values)
    print("The value of theta_0 is {}".format(theta[0]))
    print("The value of theta_1 is {}".format(theta[1]))




