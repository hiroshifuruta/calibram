import streamlit as st
import numpy as np

class LinearRegression:
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])
        self.beta = None

    def add_data(self, x, y):
        self.X = np.append(self.X, x)
        self.y = np.append(self.y, y)
        self.X = self.X.reshape(-1, 1)
        self.y = self.y.reshape(-1, 1)

    def fit(self):
        X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        self.beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(self.y)

    def predict(self, x0):
        x0 = np.array([1, x0]).reshape(1, -1)
        y0 = x0.dot(self.beta)
        return y0[0][0]

lr = LinearRegression()

st.write("target value Raman_shift_Si = 520")
Raman_shift_Si = 520

y0 = float(st.number_input("Enter y_zero_coeff: "))
x0 = float(st.number_input("Enter x_Si_raman_shift: "))
lr.add_data(x0, y0)

while True:
    y = float(st.number_input("Enter y_zero_coeff: "))
    if(y == 0 or y == ""):
        break
    x = float(st.number_input("Enter x_Si_raman_shift: "))
    lr.add_data(x, y)
    lr.fit()
    xt = 520
    yt = lr.predict(xt)
    st.write(f"Estimated yt_Zero_coeff: {yt:.2f}")