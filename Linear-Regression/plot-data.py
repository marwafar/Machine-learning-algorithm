#%%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
#%%
data1=pd.read_csv('ex1data1.csv')
#print(data1.head())

#%%
x_values=data1.City_population
y_values=data1.Profit
theta=[-3.630,1.166]
hyp=[theta[0]+theta[1]*x for x in x_values]

#%%
plt.plot(x_values,y_values,marker="o",linestyle='')
plt.plot(x_values,hyp)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show

#%%
cost=pd.read_csv('step-cost.csv')
alpha=cost.step
J=cost.cost_func
x_1=cost.theta_1
x_2=cost.theta_2
# %%
plt.plot(alpha,J)
plt.show()

# %%
