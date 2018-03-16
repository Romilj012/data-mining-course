import numpy as np

# Read the MNIST DATA
def get_MNIST_data(file_name="MNIST.txt", max_numbers=1000):

    X = []
    Y = []



    def parse_line(l):
        values = map(int, l.split(","))
        y = np.zeros(10)
        y[values[0]] = 1
        return map(lambda x: 1 if x / 255.0 > 0.5 else 0, values[1:]), y


    with open(file_name) as f:

        n = 0

        for line in f:
            x,y = parse_line(line)
            X.append(x)
            Y.append(y)

            n+= 1
            if n == max_numbers:
                break


    return np.asarray(X), np.asarray(Y)


# Get a dataset of the number of pixels and for the digits n1 and n2
# as well ass the class.
# The class for digit n1 will be 0 and the class for n2 will be 1.
def get_first_data():

    n1 = 1
    n2 = 0

    X, Y = get_MNIST_data()
    Y = np.argmax(Y, 1)
    X = X[np.logical_or(Y == n1, Y == n2),:]
    Y = np.asarray([Y[np.logical_or(Y == n1, Y == n2)]]).T
    X = np.sum(X, 1, keepdims=True)
    Y = (Y-n1) / (n2-n1)

    return X, Y
