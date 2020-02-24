import math
from random import randrange
from tkinter import *
from pprint import pprint

import scipy.io
from scipy import stats
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#START Utilerias
def arrayToList(arr):
    if type(arr) == type(np.array([])):
        return arrayToList(arr.tolist())
    elif type(arr) == type([]):
        return [arrayToList(a) for a in arr]
    else:
        return arr

def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")

def prettyArray(arr):
    pprint(arrayToList(arr))
#END Utilerias

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
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers))
    return np.sqrt(variance)

#Resume el dataset agregando los datos (Media, ,longitud
def summarize(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries

#Unificar por clase
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#Calcular la probabilidad
def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

#Calcular probabilidad por clase
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
    for i in range(len(classSummaries)):
        mean, stdev, lenz = classSummaries[i]
        x = inputVector[i]
        probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities

#Genera predicciones
def predict(summaries, row):
    probabilities = calculateClassProbabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

#Obtener predicciones
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

#Obtener Precision
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0



def main():
    #Cargar set de datos Matlab
    filename = 'datos_wdbc.mat'
    mat = scipy.io.loadmat(filename)
    #Porcentaje de datos entrenamiento
    ptrain = 0.80
    #Porcentaje de datos prueba
    ptest = 0.20
    #Carga datos entrenamiento
    trn = mat['trn']
    #Carga el set de clasificacion
    y = trn['y'][0, 0]
    #Carga el conjunto de datos
    xc = trn['xc'][0, 0]
    #Los valores 0 se remplazaron un valor minimo debido a que BOXCOX requeria valores  mayores a 0
    xc[xc == 0] = 0.000001
    row = len(xc[0:, :])
    colum = len(xc[0])
    print("Set de Datos:")
    matprint(xc)
    plt.figure(1)
    plt.subplot(211)
    plt.title("Set Datos y BOXCOX Aplicado")
    plt.plot(xc)


    # Se aplica BOXCOX al dataset
    data = np.empty([row, colum])
    for x in range(0, colum):
        train_data = stats.boxcox(xc[:, x],0.8)
        data[:, x] = train_data
    plt.subplot(212)
    plt.plot(data)
    print("Set de datos con BOXCOX aplicado")
    matprint(data)
    # Agregando clases
    kio = np.append(data, y.astype(int), axis=1)
    xc = kio
    roro = math.floor(row * ptrain)
    trainingSet = xc[0:roro, :]
    testSet = xc[roro + 1:, :]
    print("Set de datos con su clases")
    matprint(xc)
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Rendimiento: {0}%'.format(accuracy))
    plt.show()
main()

