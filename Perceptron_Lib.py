import numpy as np
import math
import random

in_dim = 785  # input dimension
out_dim = 10  # number of classes (0-9)
eta = 1  # Learning rate. You might try different rates (between 0.001 and 1) to maximize the accuracy


def Weight_update(feature, label, weight_i2o):
    ##
    # Update the weights for a train feature.
    # Inputs:
    # feature: feature vector (ndarray) of a data point with 785 dimensions. Here, the feature represents a handwritten digit
    # of a 28x28 pixel grayscale image, which is flattened into a 785-dimensional vector (include bias)
    # label: Actual label of the train feature
    # weight_i2o: current weights with shape (in_dim x out_dim) from input (feature vector) to output (digit number 0-9)
    # Return: updated weight
    ##
    # "*** YOUR CODE HERE ***"

    # First, multiply feature vector and weight
    # matrix to get vector with scores of each digit class
    var = np.dot(np.transpose(feature), np.transpose(weight_i2o))
    temp = np.argmax(var)
    if temp == label:
        return weight_i2o
    else:
        t = np.zeros(10, )
        t[int(temp)] = 1
        y = np.zeros(10, )
    y[label] = 1
    return weight_i2o == weight_i2o + np.dot(eta, np.transpose(np.dot(feature, (t - y))))


def get_predictions(dataset, weight_i2o):
    # """
    # Calculates the predicted label for each feature in dataset.
    # Inputs:
    # dataset: a set of feature vectors with shape
    # weight_i2o: current weights with shape (in_dim x out_dim)
    # Return: list (or ndarray) of predicted labels from given dataset
    # """
    # "*** YOUR CODE HERE ***"
    # initialize array for keeping predicted label values
    predictions = np.array([])
    # iterate through dataset to get each feature
    for feature in dataset:
        # as in weight_update function, we follow the same steps to get our predicted label and append them to the array
        predicted_label_vector = np.matmul(feature.transpose(), weight_i2o)
        predictLabel = np.argmax(predicted_label_vector)
        predictions = np.append(predictions, predictLabel)
    return predictions


def train(train_set, labels, weight_i2o):
    # """
    # Train the perceptron until convergence.
    # Inputs:
    # train_set: training set (ndarray) with shape (number of data points x in_dim)
    # labels: list (or ndarray) of actual labels from training set
    # weight_i2o:
    # Return: the weights for the entire training set
    # """
    for i in range(0, train_set.shape[0]):
        weight_i2o = Weight_update(train_set[i, :], labels[i], weight_i2o)
    return weight_i2o
