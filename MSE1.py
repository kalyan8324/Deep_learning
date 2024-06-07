import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


coordinates = [(43, 41), (44, 45), (45, 49), (46, 47), (47, 44)]

x_value = [x for x,y in coordinates]
y_value = [y for x,y in coordinates]

X_train, X_test, y_train, y_test = train_test_split( x_value, y_value, test_size=0.2)
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)
y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test) 
print("coefficent  = >",model.coef_)
print("intercept = > ",model.intercept_)
print(y_pred)
#  y = mx+b 
y_pred_data = []
def predict_weight(hgt):
    predicted = []
    for i in range(len(hgt)):
        # predicted.append(8.25 + 0.8 * hgt[i]
        predicted.append(model.coef_ * hgt[i] + model.intercept_)
    return predicted

y_pred_data = predict_weight(x_value)

#  find error 
mse = np.mean((np.array(y_value) - np.array(y_pred_data)) ** 2)
print(f"Mean Squared Error: {mse}")

print("mean_squared_error ",mean_squared_error(y_value,x_value))