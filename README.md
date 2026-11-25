# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the dataset (student_scores.csv).Split the data intotraining and testing sets.
2. Use the LinearRegression model from sklearn.linear_model.
3. Use the trained model to predict test data values.
4. Calculate Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to evaluate the model’s performance.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SRIVATSAN G
RegisterNumber: 212223230216
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
print(df)
print(df.head(0))
print(df.tail(0))
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pre=regressor.predict(x_test)
print("predicted value:",y_pre)
print("real value:",y_test)
#plot the train data
plt.scatter(x_train,y_train,color='blue')
plt.plot(x_train,regressor.predict(x_train),color="orange")
plt.title("hours vs score")
plt.xlabel("hours")
plt.ylabel("score")
plt.show()
#plot the test data
plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="black")
plt.title("hours vs score")
plt.xlabel("hours")
plt.ylabel("score")
plt.show()
mse=mean_absolute_error(y_test,y_pre)
msa=mean_squared_error(y_test,y_pre)
rmse=np.sqrt(mse)
print("mean square erro:",mse)
print("mean absolute error:",msa)
print("root mean square error:",rmse)

```
## Output:

<img width="308" height="288" alt="684408527-496875ea-c95f-4ae2-ad11-0eef8f718780" src="https://github.com/user-attachments/assets/67040779-689d-46c9-9d7b-2168a0c9bc18" />


<img width="752" height="83" alt="684408597-6396d139-8812-4be0-82e7-f0ebb4d959b9" src="https://github.com/user-attachments/assets/0929fa1a-1cfd-4c7c-b65b-7ede3be3f49b" />


<img width="311" height="72" alt="684408755-3c3b4c99-f874-4335-893f-f17dd375c820" src="https://github.com/user-attachments/assets/e5d515b1-f144-4add-ab2e-8b0da8c1dd1f" />


<img width="665" height="62" alt="684408895-71b1a3f2-2e5f-4ccf-9d60-6b3b13691447" src="https://github.com/user-attachments/assets/ea33b26f-0745-44fc-bf91-1750d7be9f22" />



<img width="530" height="400" alt="684408977-debd3490-cb2f-49fe-847e-aeeaf39c7e8e" src="https://github.com/user-attachments/assets/19e1f55c-d95f-48cc-a4e6-300ed15d4e95" />




<img width="536" height="397" alt="684409061-6e0d705b-107e-4a50-a9dc-37a026f9154d" src="https://github.com/user-attachments/assets/f3c10631-2ab4-4f77-88f5-ec74077db8ff" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
