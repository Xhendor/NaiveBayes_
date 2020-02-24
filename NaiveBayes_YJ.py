import math

import scipy.io
from scipy import stats
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def separateByClass(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

def mean(numbers):
    return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers))
	return np.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
     summaries[classValue] = summarize(instances)
    return summaries

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*exponent

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
    for i in range(len(classSummaries)):
        mean, stdev, lenz = classSummaries[i]
        x = inputVector[i]
        probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

def predict(summaries, row):
	probabilities = calculateClassProbabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0


def main():
    filename = 'datos_wdbc.mat'
    mat=scipy.io.loadmat(filename)
    ptrain=0.70
    ptest=0.30
    trn=mat['trn']
    y=trn['y'][0,0]
    xc=trn['xc'][0,0]
    xc[xc == 0]=0.000001
    row=len(xc[0:,:])
    colum=len(xc[0])
    #BOXCOX
    data=np.empty([row,colum])
    for x in range(0,colum):
     train_data, fitted_lambda = stats.yeojohnson(xc[:,x])
     data[:,x]=train_data
#Agregando clases
    kio=np.append(data,y.astype(int),axis=1)
    xc=kio
    roro=math.floor(row*ptrain)
#Separar el porcentaje de datos de prueba y entramiento
    trainingSet=xc[0:roro,:]
    testSet=xc[roro+1:,:]
    # use lambda value to transform test data
    #YEO-JONHSON
    # (optional) plot train & test


    # prepare model
    summaries = summarizeByClass(trainingSet)
# test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Rendimiento: {0}%'.format(accuracy))

main()