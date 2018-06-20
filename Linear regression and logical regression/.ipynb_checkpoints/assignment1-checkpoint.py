def featureNormalization(X):
    """ Feature Normalization """
    normilized_X = (X - np.mean(X))/np.std(X)
    return normilezed_X

def normalEquation(X, y):
    theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(y)
    return theta

def computeCost(X, y, theta=[[0], [0]]):
    """ Computing Cost (for Multiple Variables) """
    J = 0
    m = y.size
    h = X.dot(theta)
    J = (1/(2*m)) * np.sum(np.square(h-y))
    
    return(J)

def gradientDescent(X, y, theta=[[0], [0]], alpha=0.01, num_iters=1500):
    """ Gradient Descent (for Multiple Variables) """
    m = y.size
    J_history = []
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1/m)*(X.T.dot(h-y))
        J_history.append(computeCost(X, y, theta))
    return(theta, J_history)

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

def costFunction(theta, X, y):
    """ Logistic Regression Cost """
   J = (np.transpose(-y).dot(np.log(sigmoid(X.dot(theta))))-np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))/np.size(y)
    return(J[0])

def gradient(theta, X, y):
    """ Logistic Regression Gradient """
    grad = (np.transpose(X).dot(sigmoid(X.dot(theta)) - y.ravel()))/np.size(y)
    return grad

def predict(theta, X, threshold=0.5):
    """ Logistic Regretion predict """
    p = sigmoid(X.dot(theta)) >= 0.5
    return p

def costFunctionReg(theta, reg, *args):
    """ Regularized Logistic Regression Cost """
    J = costFunction(theta, args[0], args[1]) + reg * sum(np.square(theta[1:]))/(2 * np.size(args[1]))
    return(J)

def gradientReg(theta, reg, *args):
    """ Regularized Logistic Regression Gradient """
    tmp_theta = np.insert(theta[1:],0,0)
    h = sigmoid(args[0].dot(theta))
    grad = (args[0].T.dot(h - args[1].ravel()) / y.size + reg / y.size * tmp_theta)
    return grad