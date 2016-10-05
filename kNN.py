
# coding: utf-8

import numpy as np
import math
import operator

# Load training data
train_data = np.loadtxt('train.txt', delimiter=',')
print 'Train set: ' + repr(len(train_data))

#Load test data
test_data = np.loadtxt('test.txt', delimiter=',')
print 'Test set: ' + repr(len(test_data))


# Return mean of numbers

# numbers is a vector
def mean(numbers):
    return sum(numbers)/float(len(numbers))

# Return standard deviation of numbers

# numbers is a vector
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

# Return mean and standard deviation of all the attributes in dataset
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    del summaries[0]
    return summaries

# Generate summary for the training set
summary = summarize(train_data)


# Return standardized data

# dataset is a matrix
# summary is a list of mean and standard deviation for each attribute
def normalize(dataset, summary):
    for i in range(0,len(dataset)):
        for j in range(1,len(dataset[0])-1):
            mu, sigma = summary[j - 1]
            dataset[i][j] = (dataset[i][j] - mu) / sigma
    print dataset[0]
    return dataset

# Standardize training set and testing set with same parameters
train_data_normalized = normalize(train_data,summary)
test_data_normalized = normalize(test_data,summary)


# Return square of Euclidean distance between two data points

# instance1 and instance2 are two vectors of the same length
def l2norm(instance1, instance2, length):
    res = 0
    for x in range(1,length):
        res += pow((instance1[x] - instance2[x]), 2)
    return res

# Return Manhattan distance between two data points

# instance1 and instance2 are two vectors of the same length
def l1norm(instance1, instance2, length):
    res = 0
    for x in range(1,length):
        res += abs((instance1[x] - instance2[x]))
    return res

# Return k nearest Neighbors of the testInstance as per L2 norm

# trainingSet is the normalized matrix of training set
# testInstance is a single row from test set
def getL2Neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = l2norm(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Return k nearest Neighbors of the testInstance as per L1 norm

# trainingSet is the normalized matrix of training set
# testInstance is a single row from test set
def getL1Neighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = l1norm(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Return k nearest Neighbors as per L1 norm of the left out training instance during Leave One Out Cross Validation

# trainingSet is the normalized matrix of training set
# index is the index of left out training instance in trainingSet
def getL1NeighborsLOO(trainingSet, index, k):
    testInstance = trainingSet[index]
    distances = []
    length = len(testInstance)-1
    for x in range(index):
        dist = l1norm(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    for x in range(index+1, len(trainingSet)):
        dist = l1norm(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Return k nearest Neighbors as per L2 norm of the left out training instance during Leave One Out Cross Validation

# trainingSet is the normalized matrix of training set
# index is the index of left out training instance in trainingSet
def getL2NeighborsLOO(trainingSet, index, k):
    testInstance = trainingSet[index]
    distances = []
    length = len(testInstance)-1
    for x in range(index):
        dist = l2norm(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    for x in range(index+1, len(trainingSet)):
        dist = l2norm(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Return the class to which an instance should be classified

# neighbors is the list of k nearest neighbors of the instance we want to classify
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# Return the accuracy of model on test data

# testSet is the labeled dataset of instances we classified
# predictions is a vector of the classification decisions we have made for each instance in testSet
def getTestAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (float(correct)/float(len(testSet)) * 100.0)

# Return the accuracy of model on training data for L2 norm

# trainSet is the training dataset
# k ins the number of nearest neighbors we are taking into account
def getTrainAccuracyL2(trainSet, k):
    correct = 0
    for x in range(len(trainSet)):
        l2neighbors = getL2NeighborsLOO(trainSet, x, k)
        result = getResponse(l2neighbors)
        if trainSet[x][-1] == result:
            correct += 1
    return (float(correct)/float(len(trainSet)) * 100.0)

# Return the accuracy of model on training data for L1 norm

# trainSet is the training dataset
# k ins the number of nearest neighbors we are taking into account
def getTrainAccuracyL1(trainSet, k):
    correct = 0
    for x in range(len(trainSet)):
        l1neighbors = getL1NeighborsLOO(trainSet, x, k)
        result = getResponse(l1neighbors)
        if trainSet[x][-1] == result:
            correct += 1
    return (float(correct)/float(len(trainSet)) * 100.0)


# generate predictions

# predictions with Manhattan distance
predictions1=[]
# predictions with Euclidean ditance
predictions2=[]

for k in [1, 3, 5, 7]:
    for x in range(len(test_data)):
        l2neighbors = getL2Neighbors(train_data_normalized, test_data_normalized[x], k)
        l1neighbors = getL1Neighbors(train_data_normalized, test_data_normalized[x], k)
        result1 = getResponse(l1neighbors)
        result2 = getResponse(l2neighbors)
        predictions1.append(result1)
        predictions2.append(result2)
    train_accuracy1 = getTrainAccuracyL1(train_data, k)
    train_accuracy2 = getTrainAccuracyL2(train_data, k)
    test_accuracy1 = getTestAccuracy(test_data, predictions1)
    test_accuracy2 = getTestAccuracy(test_data, predictions2)
    print('Accuracy: ' + repr(train_accuracy1) + '%')
    print('Accuracy: ' + repr(train_accuracy2) + '%')
    print('Accuracy: ' + repr(test_accuracy1) + '%')
    print('Accuracy: ' + repr(test_accuracy2) + '%')