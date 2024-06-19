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

while True:
    x = float(input("Enter x: "))
    y = float(input("Enter y: "))
    lr.add_data(x, y)
    lr.fit()
    x0 = float(input("Enter x0: "))
    y0 = lr.predict(x0)
    print(f"Estimated y0: {y0:.2f}")