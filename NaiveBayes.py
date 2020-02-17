import math

import scipy.io
from scipy import stats
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    spc=mat['spc']
    y=trn['y'][0,0]
    xc=trn['xc'][0,0]
    kio=np.append(xc,y.astype(int),axis=1)
    xc=kio
    print(xc)
    row=len(xc[0:,:])
    colum=len(xc[0])
    roro=math.floor(row*ptrain)
    splitRatio = 0.67
    trainingSet=xc[0:roro,:]
    testSet=xc[roro+1:,:]
    trainigSetPositive=trainingSet[trainingSet > 0]
    #BOXCOX
    train_data, fitted_lambda = stats.boxcox(trainigSetPositive[:,0])
    # use lambda value to transform test data
    test_data = stats.boxcox(testSet[testSet>0], fitted_lambda)
    #YEO-JONHSON
    # (optional) plot train & test
    fig, ax = plt.subplots(1, 3)
    sns.distplot(train_data, ax=ax[0])
    sns.distplot(test_data, ax=ax[1])
    xt, lmbda= stats.yeojohnson(train_data)
    prob = stats.probplot(xt, dist=stats.norm,plot=ax[2])
    plt.show()

    # prepare model
    summaries = summarizeByClass(trainingSet)
# test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%'.format(accuracy))

main()