#%%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# %%
df=pd.read_csv("ex2data2.csv")
#print(df.head())

# %%
plt.figure()
y_1_data=df[df.y==1]
y_0_data=df[df.y==0]
plt.plot(y_1_data.Test_1,y_1_data.Test_2,marker="o",linestyle='')
plt.plot(y_0_data.Test_1,y_0_data.Test_2,marker="o",linestyle="")
plt.xlabel("Microship Test 1")
plt.ylabel("Microship Test 2")
plt.legend(["y=1","y=0"])
plt.title("Training data set")
plt.show()

# %%
cost=pd.read_csv("cost_function_polynomial.csv")
steps=cost.step
j=cost.cost_func
plt.plot(steps,j)
plt.xlabel("Steps")
plt.ylabel("The cost")
plt.title("The cost function")
plt.show()

# %%
theta_df=pd.read_csv("theta_features_polynomial.csv")
theta=theta_df.iloc[0,:]
#print(theta)
#df=pd.read_csv("ex2data2.csv")

init=-1.0
final=1.5
n=50
#delta=(final-init)/n
x_1=np.linspace(init,final,n)
x_2=np.linspace(init,final,n)
U,V=np.meshgrid(x_1,x_2)

U=np.ravel(U)
V=np.ravel(V)
Z=np.zeros(len(x_1)*len(x_2))

degree=7
X_poly=np.ones(U.shape[0])
for i in range(1,degree):
    for j in range(0,i+1):
        k=i-j
        X_poly=np.column_stack((X_poly,(np.power(U,k)*np.power(V,j))))
        
#print(X_poly)
Z=X_poly.dot(theta)

U=U.reshape((len(x_1),len(x_2)))
V=V.reshape((len(x_1),len(x_2)))
Z=Z.reshape((len(x_1),len(x_2)))

ax=plt.subplot()
y_1_data=df[df.y==1]
y_0_data=df[df.y==0]
ax.plot(y_1_data.Test_1,y_1_data.Test_2,marker="o",linestyle='')
ax.plot(y_0_data.Test_1,y_0_data.Test_2,marker="o",linestyle="")
ax.contour(U,V,Z,levels=[0])
ax.set_xlabel("Microship Test 1")
ax.set_ylabel("Microship Test 2")
ax.legend(["y=1","y=0", "Decisison boundary"])
ax.set_title("Training data with decision boundary")

# %%
