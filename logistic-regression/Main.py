import matplotlib.pyplot as plt
import numpy as np
from ReadData import get_first_data, get_MNIST_data

np.random.seed(12345)


# The logistic function.
# Should return a vector/matrix with the same dimensions as x.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The cost function, the price we are paying for beeing wrong.
def cost(X, W, b, Y):
    Yhat = sigmoid(np.dot(X, W) + b)
    #return -1 / (np.shape(Y)[0]) * np.sum( -1 * np.log(Yhat) * Y - np.log(1-Yhat) * (1-Y) )
    return -1 / (np.shape(Y)[0]) * np.sum( Y * np.log(Yhat) + (1-Y) * np.log(1-Yhat))


# The derivative of the cost function at the
# Should return a tuple of (d cost)/(d W) and (d cost)/(d b).
# This function should work both when W and b are vectors or matrices.
def d_cost(X, W, b, Y):
    Yhat = sigmoid(np.dot(X, W) + b)

    dW = -1.0 / (np.shape(Y)[0]) * np.dot(X.T , (Y - Yhat))
    db = -1.0 / (np.shape(Y)[0]) * np.sum(Y - Yhat)

    return (dW, db)


# Normalize each column ("attribute") of a given matrix.
# The normalization is done so that each column of the output matrix
# have 0 mean and unit standard diviation.
def featureNormalization(x):
    return (x - np.mean(x, 0)) / np.std(x, 0)


# Preform gradient descent given the input data X and
# the expected output Y.
# alpha is the learning rate.
# Returns the value of W and b and a list of the cost for each iteration.
def gradient_descent(X, Y, alpha, iterations=1000):

    cost_over_time = []


    W = np.zeros((X.shape[1], Y.shape[1]))
    b = np.zeros((1, Y.shape[1]))


    for i in xrange(iterations):

        cost_over_time.append(cost(X, W, b, Y))

        dW, db = d_cost(X, W, b, Y)
        W = W - alpha * dW
        b = b - alpha * db


    return (W, b, cost_over_time)


# Classify each output depending on if it is smaller or larger than 0.5
def classify(X, W, b):
    if (sigmoid(np.dot(X, W) + b) <= 0.5): return 0
    elif (sigmoid(np.dot(X, W) + b) > 0.5): return 1


def learn_digits():

    # Train model
    X, Y = get_MNIST_data()
    W, b, cost_over_time = gradient_descent(X, Y, 0.1, 2500)

    # Create matrix to plot
    mp = (map(lambda x: np.reshape(W[:,x], (28,28)), range(10)))
    m1 = reduce(lambda x, y: np.concatenate((x, y), 1), mp[ :5])
    m2 = reduce(lambda x, y: np.concatenate((x, y), 1), mp[5:])

    m = np.concatenate((m1, m2), 0)


    # Plot the learnt weights in a grid
    plt.matshow(m)
    plt.plot([0, m.shape[1]], [m.shape[0]/2, m.shape[0]/2], 'k-')

    for i in range(1,5):
        plt.plot([28*i, 28*i], [0, m.shape[0]-1], 'k-')

    plt.axis([0, m.shape[1]-1, m.shape[0]-1,0])

    plt.show()




def classify_zeros():


    X, Y = get_first_data()
    X = featureNormalization(X)


    # Split dataset into a training and testing set
    testing = np.random.uniform(low=0, high=1, size=X.shape[0]) > 0.1
    training = np.logical_not(testing)

    X_test = X[testing]
    Y_test = Y[testing]
    X = X[training]
    Y = Y[training]

    alpha_rates = [50, 10, 1, 0.1, 0.01, 0.001]

    for a in alpha_rates:
        alpha = a

        W, b, cost_over_time = gradient_descent(X, Y, alpha)

        # Plot the achived result
        f, (ax1, ax2) = plt.subplots(1, 2)

        f.canvas.set_window_title("Learning Rate: " + str(alpha))
        ax1.plot(cost_over_time)
        ax1.set_title("Cost over time")

        X_l = np.asarray([np.linspace(np.min(X), np.max(X), 1000)]).T
        Yhat = sigmoid(np.dot(X_l,W) + b)
        ax2.plot(X, Y, 'bo' , X_l, Yhat, 'r-')
        ax2.set_title("The fitted logistic function")

        plt.show()

        # Report the classification error of the test set

        # You should put your own code for calculating the percentage of
        # correctly classified examples in the test set stored in X_test.

        # If all previous steps are correctly implemented calling classify_zeros()
        # would first train the logistic regression model and then plot the
        # decrease in cost over time and the fitted sigmoid function.

        # The percentage of correctly classified 0s would be around 91%

        print("++++++++++++++++++++" + '\n' + "Results for Learning Rate " + str(alpha) + '\n')

        length = len(X_test)
        all_zeros = 0
        all_ones = 0
        predicted_zeros = 0
        predicted_ones = 0
        Y_predicted = []

        for i in xrange(length):
            prediction = classify(X_test[i], W, b)
            Y_predicted.append(prediction)
            answer = Y_test[i]
            if (answer == 0):
                all_zeros = all_zeros + 1
                if (answer == prediction):
                    predicted_zeros = predicted_zeros + 1
            elif (answer == 1):
                all_ones = all_ones + 1
                if (answer == prediction):
                    predicted_ones = predicted_ones + 1

        print("Number of Zeros in Test Set: " + str(all_zeros))
        print("Number of Zeros predicted correctly: " + str(predicted_zeros))
        print("Successful prediction percentage: " + str(float(predicted_zeros) / float(all_zeros) * 100) + " %")
        print("====================")
        print("Number of Ones in Test Set: " + str(all_ones))
        print("Number of Ones predicted correctly: " + str(predicted_ones))
        print("Successful prediction percentage: " + str(float(predicted_ones) / float(all_ones) * 100) + " %")
        print("====================")
        print("Number of Records in Test Set: " + str(length))
        print("Number of Records predicted correctly: " + str(predicted_ones + predicted_zeros))
        print("Successful prediction percentage: " + str(float(predicted_ones + predicted_zeros) / float(length) * 100) + " %" + '\n')

# To test your implementatio run
classify_zeros()
learn_digits()