import csv
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

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
    print("Started")
    x_data, y_data = read_input()

    #Splitting to train set and test set
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    print("Read, shuffle and split complete")


    #FOR TESTING
    #x_train = x_train[:2000]
    #y_train = y_train[:2000]

    '''param_grid = {'n_estimators': [10, 60, 500], 'max_features': ['auto', 'log2', None],
                    'max_depth': [1, 50, None],
                     'bootstrap': [True, False], 'oob_score': [False]}'''


    param_grid = {'n_estimators': [10, 30], 'max_features': ['auto', None], 'max_depth': [ 50, None],
                   'bootstrap': [True, False], 'oob_score': [False]} # parameter grid

    #param_grid={}
    time_start = datetime.now()


    #The GridSearchCV object does all cross validation for all parameter combinations
    clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy', n_jobs=1)
    clf.fit(x_train, y_train)

    f = open("randomdtree_scores.txt", "a")
    f.write("Time started " + str(time_start)+"\n")
    f.write("Time ended " + str(datetime.now()) + "\n")
    f.write("Time elapsed " + str(datetime.now()-time_start) + "\n")
    f.write("Best score " + str(clf.best_score_) + " with following parameters " + str(clf.best_params_) + "\n")
    f.write(str(clf.best_estimator_) + "\n")
    f.write("All tests\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        f.write("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        f.write("\n")
    model = RandomForestClassifier(**clf.best_params_)
    time_fitstart = datetime.now()
    model.fit(x_train, y_train)
    f.write("\nData for best model:")
    f.write("\nTrain data fit time " + str(datetime.now()-time_fitstart) + " train data length " + str(len(x_train)) + "\n")
    time_predictstart = datetime.now()
    pred = model.predict(x_test)
    f.write("Test data predict time " + str(datetime.now()-time_predictstart) + " test data length " + str(len(x_test)) +"\n")
    f.write("Accuracy was " + str(accuracy_score(y_test, pred)))
    f.write("\nReport\n" + str(classification_report(y_test, pred)))
    f.write("\n\n\n")
    print("Done")


main()