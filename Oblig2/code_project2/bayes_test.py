import csv
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



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
    #x_train = x_train[:5000]
    #y_train = y_train[:5000]
    #k_range = [1]

    alpha_range = [10 ** i for i in range(-6, 6)]
    param_grid = dict(alpha=alpha_range) #Parameter grid
    param_grid['fit_prior'] = [True, False]

    time_start = datetime.now()

    #param_grid = {}

    #The GridSearchCV object does all cross validation for all parameter combinations
    clf = GridSearchCV(MultinomialNB(), param_grid, cv=3, scoring='accuracy', n_jobs=1)
    clf.fit(x_train, y_train)


    #Logging data from GridSeasrchCV testing
    f = open("bayes_scores.txt", "a")
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
    model = MultinomialNB(**clf.best_params_)
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

