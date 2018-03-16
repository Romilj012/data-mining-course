import numpy as np
from sklearn import datasets
import sys
import matplotlib.pyplot as plt

np.random.seed(12345)

# Function for loading the iris data
# load_data returns a 2D numpy array where each row is an example
#  and each column is a given feature.
def load_data():
    iris = datasets.load_iris()
    return iris.data

# Euclidian distance between two points
def distance(p1, p2):
    return np.sum(np.absolute(np.subtract(p1,p2))) / len(p1)

# Assign labels to each example given the center of each cluster
def assign_labels(X, centers):
    labels = []
    for i in range(0, len(X)):
        previous_distance = sys.maxint
        assigned_label  = 0
        for j in range(0, len(centers)):
            d = distance(X[i], centers[j])
            if d < previous_distance:
                previous_distance = d
                assigned_label = j
        labels.append(assigned_label)
    return labels

# Calculate the center of each cluster given the label of each example
def calculate_centers(X, labels, K):
    clusters = []
    members = []
    centers = []

    for i in range(0, K):
        clusters.append(0)
        members.append(0)
        centers.append(0)

    for i in range(0, len(X)):
        point = X[i]
        label = labels[i]
        clusters[label] = np.add(point, clusters[label])
        members[label] += 1

    for i in range(0, len(clusters)):
        try:
            centers[i] = clusters[i] / members[i]
        except (ValueError):
            pass
    return centers

# Test if the algorithm has converged
# Should return a bool stating if the algorithm has converged or not.
def test_convergence(old_centers, new_centers):
    return distance(old_centers, new_centers) < 0.001

# Evaluate the performance of the current clusters
# This function should return the total mean squared error of the given clusters
def evaluate_performance(X, labels, centers):
    all_errors = []
    for i in range(0, len(X)):
        point = X[i]
        label = labels[i]
        center = centers[label]
        squared_error = np.subtract(point, center) ** 2
        all_errors.append(squared_error)
    return 1.0 / len(X) * np.sum(all_errors)

# Algorithm for preforming K-means clustering on the give dataset
def k_means(X, K):
    data = X.copy()
    np.random.shuffle(data)
    new_centers = []
    for i in range(0, K):
        new_centers.append(data[i])
    converged = False
    labels = []
    counter = 0
    while (converged == False) and (counter < 200):
        labels = assign_labels(X, new_centers)
        old_centers = new_centers[:]
        new_centers = calculate_centers(X, labels, K)
        converged = test_convergence(old_centers, new_centers)
        counter += 1
    mse = evaluate_performance(X, labels, new_centers)
    return (labels, mse, new_centers)

# Evaluating kmeans, finding best value for k by applying the elbow method. Try different values for k.
X = load_data()
k = 20
mse = []
for i in range(1, k+1):
    mse.append(0)
    labels, mean, centers = k_means(X, i)
    mse[i - 1] = mean
print(mse)
mse = [None] + mse

plt.plot(mse)
plt.xlabel('k')
plt.title('Mean Squared Error')
plt.xticks(range(0, k+1, 1))
plt.show()
