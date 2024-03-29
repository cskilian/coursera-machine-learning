import sys
import numpy as np
import scipy
import matplotlib as plot

# constants
DEF_DELIMITER = ","
DEF_EPSILON = 0.0000005
DEF_ALPHA = 1           # learning rate
DEF_FNAME = "data.txt"

# functions
def load(training_set, feature_count):
    X = np.loadtxt(open(training_set, "r"), delimiter = DEF_DELIMITER, usecols = (0, feature_count - 1))
    y = np.loadtxt(open(training_set, "r"), delimiter = DEF_DELIMITER, usecols = (feature_count))
    return (X, y)

def normalize(X, mu, sigma):
    '''
    This normalizes the values in the training set
    :param X: matrix of the training data
    :param mu: row vector of the mean of the training features
    :param sigma: row vector of the standard deviation of the training features
    :return: matrix of the training data after feature scaling
    '''
    X = X - mu
    X = X / sigma
    return X

def error(X, theta, y, m):
    '''
    This calculates the squared error
    '''
    delta = np.matmul(X, theta) - y
    return np.matmul(delta.transpose(), delta) / (2 * m)

def grad_descent(X, theta, y, m, alpha, epsilon):
    '''
    This is the gradient descent algorithm. We repeatedly calculate the squared error,
    adjust the weights, calculate the error again, and quit when the error function has converged
    :param X: matrix of the  training data with the constant terms contained in the 1st column
    :param theta: matrix of the weights of the hypothesis function
    :param y: column vector of the training data's output
    :param m: scalar of the number of training examples
    :param alpha: scalar of the learning rate
    :param epsilon: scalar of the acceptable error boundary
    :returns: the weights at a local minimum
    '''
    while True:
        before_error = error(X, theta, y, m)
        # update thetas
        delta = np.matmul(X, theta) - y
        theta = theta - (alpha / m) * np.matmul(delta.transpose(), X).transpose()
        after_error = error(X, theta, y, m)
        if after_error >= before_error:
            print("Squared error is not converging to a minimum!")
            sys.exit()
        elif before_error - after_error < epsilon:
            break
    return theta

def main():
    '''
    Arguments
    --file -f: sets the path of the training set file
    --alpha -a: sets the learning rate
    --epsilon -e: sets the maximum delta between 2 error calculations, if the error functions
    rate of change is smaller than this, we determine the squared error function has converged
    --prediction -p: an array of inputs you want to make a prediction for after training the model
    '''
    # init default parameters
    training_set = DEF_FNAME
    alpha = DEF_ALPHA
    epsilon = DEF_EPSILON
    prediction = []
    # parse arguments
    for i in range(0, len(sys.argv)):
        if (sys.argv[i] == "--file" or sys.argv[i] == "-f") and sys.argv[i + 1][0] != "-":
            training_set = sys.argv[i + 1]
            i += 1
        elif (sys.argv[i] == "--alpha" or sys.argv[i] == "-a") and sys.argv[i + 1][0] != "-":
            alpha = float(sys.argv[i + 1])
            i += 1
        elif (sys.argv[i] == "--epsilon" or sys.argv[i] == "-e") and sys.argv[i + 1][0] != "-":
            epsilon = float(sys.argv[i + 1])
            i += 1
        elif (sys.argv[i] == "--prediction" or sys.argv[i] == "-p") and sys.argv[i + 1][0] != "-":
            i += 1
            while i < len(sys.argv) and sys.argv[i][0].isnumeric():
                prediction.append(float(sys.argv[i]))
                i += 1
    # training
    (X, y) = load(training_set, feature_count(training_set))
    y = y.reshape(y.shape[0], 1)                  # convert y into column vector
    mu = np.mean(X, 0)                            # calculate means of features
    sigma = np.std(X, 0)                          # calculate standard dev. of features
    X = normalize(X, mu, sigma)                   # feature scaling
    X = np.append(np.ones((X.shape[0], 1)), X, 1) # add ones as constant term
    init_theta = np.empty((X.shape[1], 1))        # init theta to random values
    theta = grad_descent(X, init_theta, y, X.shape[0], alpha, epsilon)
    print("Theta: \n" + str(theta))
    # hypothesis prediction
    prediction = np.array([prediction])                                      # convert prediction to row vector
    prediction = normalize(prediction, mu, sigma)                            # normalize prediction
    prediction = np.append(np.ones((prediction.shape[0], 1)), prediction, 1) # add a column of 1's for the constant term in theta
    y_hat = np.matmul(prediction, theta)
    print("Prediction: \n" + str(y_hat))
    return y_hat
    

# helpers
def feature_count(training_set):
    with open(training_set, "r") as file:
        for line in file:
            return len(line.split(DEF_DELIMITER)) - 1
    return 0

if __name__ == "__main__":
    main()