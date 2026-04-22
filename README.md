# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Normalize the input feature using mean and standard deviation to scale the values.
    This helps the gradient descent converge faster and more efficiently.

2.  Initialize slope (m) and intercept (b) to zero along with learning rate and epochs.
    Also determine the number of data points in the dataset.

3.  For each iteration, compute predicted values and calculate gradients.
    Update m and b using the gradient descent update rules.

4.  After training, compute final predictions using optimized m and b.
    Plot the data points and regression line to visualize the model fit.



## Program:
```
Developed by: DAWOOD M
RegisterNumber:212225040055

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("Startup.csv")

rnd=data['R&D Spend'].values
prof=data['Profit'].values

x=(rnd - rnd.mean()) / rnd.std()

m=0
b=0

learning_rate=0.01
epochs=1000
n=len(x)

for i in range(epochs):
  y_pred = m * x + b


  dm = ( -2 / n ) * np.sum(x * (prof - y_pred))
  db = ( -2 / n ) * np.sum(prof - y_pred)

  m = m - learning_rate * dm
  b = b - learning_rate * db

print("Slope(m):",m)
print("Intercept(b):",b)

y_pred = m * x + b


plt.scatter(x,prof)
plt.plot(x,y_pred)


plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")


plt.show()

```

## Output:
<img width="901" height="632" alt="image" src="https://github.com/user-attachments/assets/e88f6ff1-ed6c-4b62-8a8e-3cd9339e5b72" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
