class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold
    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
        plt.legend()
        x1_min, x1_max = X[:,0].min(), X[:,0].max(),
        x2_min, x2_max = X[:,1].min(), X[:,1].max(),
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        probs = model.predict_prob(grid).reshape(xx1.shape)
        plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black');
        
gd = GenerateData()
X, y = gd.logisticRegre()
Y = Y.reshape(-1,1)# This will reshape Y as a column vector. conversally reshape(1,-1) will reshape an array as row vector

X_train, X_test, y_train, y_test = train_test_split(\
                X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
model.plot()
