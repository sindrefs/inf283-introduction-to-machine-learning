import csv
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.utils import to_categorical
from keras import models
from keras import layers
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dense
import random
from datetime import datetime


def read_input():
    '''
    Reads input from hard coded path, returns x dataset and y dataset
    '''
    with open('handwritten_digits_images.csv', 'r') as pictures_file:
        pictures_reader = csv.reader(pictures_file)
        pictures_list = list(pictures_reader)

    with open('handwritten_digits_labels.csv', 'r') as labels_file:
        labels_reader = csv.reader(labels_file)
        labels_list = list(labels_reader)

    x_data = np.array(pictures_list).astype(np.int)
    y_data = np.array([int(i[0]) for i in labels_list])

    return x_data, y_data


def main():
    '''
    Main function in program
    '''

    #Reshaping the data
    print("Started")
    x_data, y_data = read_input()

    #Splitting to train set and test set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_train = x_train.reshape(x_train.shape[0], 784).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')

    x_train = x_train/255.0
    x_test = x_test/255.0

    print("Read, shuffle and split complete")

    #FOR TESTING
    #x_train = x_train[:2000]
    #y_train = y_train[:2000]

    #USES CROSS VAL FOR FINDING BEST PARAMETERS
    find_best_params(x_train, y_train, x_test, y_test)

    params = {'n_layers': 2, 'epochs': 20, 'batch_size': 80, 'n_neurons': 250} #Grid of parameters for scoring
    time_best_start = datetime.now()
    print("Score for best model", model_score(x_train, y_train, x_test, y_test, params, timing=True))


def find_best_params(x_train, y_train, x_test, y_test):
    '''
    Grid of parameters is hard coded. Function does cross validation with all combinations, and writes reports to file
    '''
    # param_grid = {'n_layers': [2, 6, 10, 14], 'epochs': [10, 20, 30], 'batch_size': [80, 100, 120],
    #              'n_neurons': [10, 80, 150, 250]}

    param_grid = {'n_layers': [10, 20], 'epochs': [10, 20], 'batch_size': [80, 120], 'n_neurons': [100, 250]}

    # param_grid = {'n_layers': [14, 2], 'epochs': [10], 'batch_size': [80], 'n_neurons': [10]}

    params = {}
    f = open("customnn_scores.txt", "a")
    best_score = 0
    best_params = {}
    time_crossval_start = datetime.now()

    f.write("Model testing started " + str(time_crossval_start) + "\n")
    f.write("Testing for following parameters\n")
    f.write(str(param_grid) + "\n\n")
    for lay in param_grid['n_layers']:
        for epo in param_grid['epochs']:
            for bat in param_grid['batch_size']:
                for neu in param_grid['n_neurons']:
                    params = {'n_layers': lay, 'epochs': epo, 'batch_size': bat, 'n_neurons': neu}
                    print("one cross val done")
                    f.write("Cross val. three fold for\n")
                    f.write(str(params) + "\n")
                    score = cross_val(x_test, y_test, params) #Cross validation with given parameters
                    f.write("Gives score of " + str(score) + "\n\n")

                    if score > best_score: #Saves best parameter variables
                        best_score = score
                        best_params = params

    f.write("\nBest params was\n")
    f.write(str(best_params) + "\n")
    f.write("with score of " + str(best_score) + "\n")
    f.write("Used time " + str((datetime.now() - time_crossval_start)) + "\n\n")
    time_test_start = datetime.now()

    #Does normal testing with best parameters (just found)
    test_score = model_score(x_train, y_train, x_test, y_test, best_params
    f.write("Testing with test set gave score of " + str(test_score) + "\n")
    f.write("Used following time " + str((datetime.now() - time_test_start)) + "\n\n\n\n\n")

    print("Done")


def cross_val(x_train, y_train, params, n_folds = 3):
    """
    Performs cross validation on model with given  parameters, returns score
    """

    #Shaping of data
    x = np.array(x_train)
    y = np.array(y_train)
    x_split = np.array_split(x, 3)
    y_split = np.array_split(y, 3)

    total = 0
    for i in range(n_folds): #All different iterations of splitting
        x_here_train = np.concatenate([x_split[j] for j in range(len(x_split)) if i != j])
        y_here_train = np.concatenate([y_split[j] for j in range(len(y_split)) if i != j])
        x_here_validation = x_split[i]
        y_here_validation = y_split[i]

        #Does model scoring
        score = model_score(x_here_train, y_here_train, x_here_validation, y_here_validation, params)
        total += score

    mean = total/3

    return mean


def model_score(x_train, y_train, x_test, y_test, params, timing = False):
    """
    Performs a normal scoring (training on train set then predicts on test set)
    """
    model = Sequential()  # Initialize
    model.add(Dense(784, input_dim=784, activation='relu'))  # Input layer

    for i in range(params['n_layers']):
        model.add(Dense(params['n_neurons'], activation='relu'))

    model.add(Dense(10, activation='sigmoid'))  # Output layer
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    time_train_start = datetime.now()
    model.fit(x_train, y_train, epochs = params['epochs'], batch_size = params['batch_size'], verbose=0)

    if timing: print("Time fit", datetime.now()-time_train_start)
    time_evaluate_start = datetime.now()
    scores = model.evaluate(x_test, y_test)
    if timing: print("Time evaluate", datetime.now()-time_evaluate_start)
    return scores[1]


main()
