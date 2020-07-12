#%%
from matplotlib import pyplot as plt
import pandas as pd

#%%
df1=pd.read_csv("ex2data1.csv")
print(df1.head())

# %%
plt.figure()
admitted=df1[df1.y > 0]
admitted.reset_index(drop=True,inplace=True)
not_admitted=df1[df1.y == 0]
not_admitted.reset_index(drop=True,inplace=True)
x1=admitted.Exam_1_score
y1=admitted.Exam_2_score
x2=not_admitted.Exam_1_score
y2=not_admitted.Exam_2_score
plt.plot(x1,y1,marker="o",linestyle='',label="Admitted")
plt.plot(x2,y2,marker="o",linestyle='',label="Not Admitted")
plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Training data")
plt.show()


# %%
cost=pd.read_csv("cost_function.csv")
steps=cost.step
j=cost.cost_func
plt.plot(steps,j)
plt.xlabel("Steps")
plt.ylabel("The cost")
plt.title("The cost function")
plt.show

# %%
theta_0=-8.05308516
theta_1=0.07019601
theta_2=0.06375572

x1_values=df1.Exam_1_score
x2_values=[(theta_0+theta_1*i)/-theta_2 for i in x1_values]

plt.figure()
admitted=df1[df1.y > 0]
admitted.reset_index(drop=True,inplace=True)
not_admitted=df1[df1.y == 0]
not_admitted.reset_index(drop=True,inplace=True)
x1=admitted.Exam_1_score
y1=admitted.Exam_2_score
x2=not_admitted.Exam_1_score
y2=not_admitted.Exam_2_score
plt.plot(x1,y1,marker="o",linestyle='',label="Admitted")
plt.plot(x2,y2,marker="o",linestyle='',label="Not Admitted")
plt.plot(x1_values,x2_values)
plt.legend()
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Training set with decision boundary")
plt.show()


# %%
