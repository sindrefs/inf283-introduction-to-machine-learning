import numpy as np
from node import Tree
import random
from sklearn import tree #ONLY USED FOR COMPARING MY SOLUTION WITH SK LEARN DESICION TREE

def read_from_file(file):
    """
    Reads data from text file
    :param file: location of file as string
    :return: array of each item in array as list of Tree (node representation)
    """
    text_data = open(file, "r")
    X = []
    Y = []
    for line in text_data:
        line = line.strip("\n").split(",")
        if "?" in line:
            continue
        Y.append(line[0])
        X.append(line)
    return X, Y


def entropy(col):
    """
    Returns entropy measurement for an attribute collum
    :param col: collum of attributes as list
    :return: entropy measurement as double
    """
    elem, vals = np.unique(col, return_counts=True) #Find unique attribute values in collum
    tot = 0
    for val in vals:
        tot += val
    entropy = 0
    for c in vals:
        entropy += (c/tot)*np.log2((c/tot))
    return -entropy


def gini(col):
    """
    Returns gini measurement for an attribute collum
    :param col: collum of attributes as list
    :return: gini measurement as double
    """
    elem, vals = np.unique(col, return_counts=True)#Find unique attribute values in collum
    tot = 0
    for val in vals:
        tot += val
    gini = 0
    for c in vals:
        gini += pow((c/tot), 2)
    return 1-gini


def ig(x, y, feature, impurity_measure):
    """
    Returns information gain for given feature
    :param x: dataset of items as list of list
    :param y: target values for items as list
    :param feature: feature to calculate split on as integer
    :param impurity_measure: impurity measure entropy or gini as string
    :return: information gain for feature as double
    """
    x = np.array(x)

    if impurity_measure == "entropy":
        impurity_parent = entropy(y)
    elif impurity_measure == "gini":
        impurity_parent = gini(y)
    else: return None

    impurity_feature = 0

    elem, count = np.unique(x[:, feature], return_counts=True)#Find unique attribute values in collum for feature
    tot = np.sum(count)
    for i in range(len(elem)): #Loops through all unique attribute values
        # Returns a collum of target values for the items where the attribute value is equal to the given attribute
        # we are adding for
        class_for_feature = [item[0] for item in x if item[feature] == elem[i]]
        if impurity_measure == "entropy":
            impurity_feature += (count[i] / tot) * entropy(class_for_feature)
        elif impurity_measure == "gini":
            impurity_feature += (count[i] / tot) * gini(class_for_feature)

    return impurity_parent-impurity_feature


def max_ig(x, y, feature_list, impurity_measure):
    """
    Returns the feature which gives the largest information gain
    :param x: dataset of items as list of list
    :param y: target values for items as list
    :param feature_list: available features for each iteration as list
    :param impurity_measure: impurity measure entropy or gini as string
    :return: index for largest information gain as integer
    """
    best = -1.0
    best_index = None
    for feature in feature_list:
        val = ig(x, y, feature, impurity_measure)
        if val > best:
            best = val
            best_index = feature
    return best_index


def id3(x, y, feature_list, impurity_measure="entropy", parent_node=None):
    """
    Creates a decision tree, with the id3 alogoritm, based on learning data
    :param x: dataset of items as list of list
    :param y: target values for items as list
    :param feature_list: available features for each iteration as list
    :param impurity_measure: impurity measure entropy or gini as string
    :param parent_node: parent for this node iteration as Tree (node representation), default None
    :return: root node for tree as type Tree (node representation)
    """
    x = np.array(x)
    uniques, counts = np.unique(y, return_counts=True)#Find unique target values

    if len(uniques) == 1:  # Set is pure, only one target value left in set
        return Tree(uniques[0], label=uniques[0])

    if len(feature_list) == 0:#No more attributes to split on
        if counts[0] > counts[1]:
            return Tree(uniques[0], label=uniques[0])
        else:
            return Tree(uniques[1], label=uniques[1])

    best_attribute = max_ig(x, y, feature_list, impurity_measure)

    vals, counts = np.unique(x[:, best_attribute], return_counts=True)#Finds unique attribute values
    this_node = Tree(best_attribute, parent_node)

    n_children = len(vals)

    feature_list.remove(best_attribute)
    children_labels = []
    for child in range(n_children): #Creates each children, based on unique attribute values
        x_for_this_child = []
        y_for_this_child = []
        for item in range(len(x)): #Loops through all items, and passes on all items with correct attribute value
            if x[item][best_attribute] == vals[child]: # Items where the attribute value is equal to the decision branch
                x_for_this_child.append(x[item])
                y_for_this_child.append(y[item])

        # Continues to grow this child node
        new_child = id3(x_for_this_child, y_for_this_child, feature_list, impurity_measure, parent_node=this_node)
        new_child.set_was_split_on(vals[child]) #Saves what attribute value this child was split on
        this_node.add_child(new_child)
        children_labels.append(new_child.label)

    label_list, label_count = np.unique(children_labels, return_counts=True) #Lists of children labels
    label = label_list[np.argmax(label_count)]
    this_node.set_label(label)#This node get most popular label among children

    return this_node


def add_error(x, y, node):
    """
    Adds error values for each node
    :param x: dataset of items as list
    :param y: target values for items as list
    :param node: top node in tree as Tree (node representation)
    :return: void, returns nothing
    """
    if node.label != y:
        node.inc_error()
    if node.is_leaf():
        return
    children = node.get_children()
    for child in children:
        if child.split_on_attribute == x[node.data]:
            add_error(x, y, child)


def prune(node):
    """
    Does pruning of tree i.e. removes redundent nodes
    :param node: top node for tree as Tree (node representation)
    :return: error for top node as integer
    """
    if node.is_leaf():
        return node.error

    children = node.get_children()
    total_error = 0
    can_delete_children = True

    for child in children:
        total_error += prune(child)
        if not child.is_leaf():
            can_delete_children = False
    if total_error > node.error and can_delete_children:
        node.children = []
        node.data = node.label
    return node.error


def learn(X, Y, impurity_measure="entropy", pruning=False, shuffle=False, split_ratio=0.4):
    """
    Creates a decision tree for a dataset
    Pruning is done in a 60/40 manner unless otherwise is specified
    :param X: dataset of items as list of list
    :param Y: target values for items as list
    :param impurity_measure: use entropy or gini as string
    :param pruning: true or false for pruning as boolean
    :param shuffle: true or false for shuffling of dataset before pruning as boolean
    :param split_ratio: value bigger than zero, smaller than one to split dataset in before pruning as double
    :return: top node for tree as Tree (node representation)
    """
    assert 0 < split_ratio < 1 #split ratio must be between 0 and 1
    f_list = list(range(1, len(X[0])))  # list of available* features, from start all features
    if pruning:
        if shuffle: #Shuffels X and Y together
            x_and_y = list(zip(X, Y))
            random.shuffle(x_and_y)
            X[:], Y[:] = zip(*x_and_y)
        l = len(X)
        split_on = (int(l * split_ratio))
        X_learn = X[split_on:]
        Y_learn = Y[split_on:]
        X_prune = X[:split_on]
        Y_prune = Y[:split_on]

        root = id3(X_learn, Y_learn, f_list, impurity_measure)

        for i in range(len(X_prune)):
            add_error(X_prune[i], Y_prune[i], root)
        prune(root)
    else:
        root = id3(X, Y, f_list, impurity_measure)
    return root


def predict(x, tree):
    """
    Predicts the target of an item
    :param x: item as list
    :param tree: top node of tree as Tree (node representation)
    :return: target value as string
    """
    if tree.is_leaf():
        return tree.data
    children = tree.get_children()
    for child in children:
        if child.split_on_attribute == x[tree.data]: #There is a node representing the decision given the feature to split on
            return predict(x, child)
    return tree.label


def test_model(model, X_test, Y_test):
    """
    Takes a tree, then does testing on it, and returns the accuracy of the tree
    :param model: root node for decision tree as Tree (node representation)
    :param X_test: dataset of items as list of list
    :param Y_test: target values for items as list
    :return:
    """
    n_sucsess =0
    n_tests= 0
    for i in range(len(X_test)): #Tests each item, and compares predicted value with correct target value
        predicted_val = predict(X_test[i], model)
        n_tests +=1
        if predicted_val == Y_test[i]:
            n_sucsess +=1
    return (n_sucsess/n_tests)


def cross_val(X, Y, iteration, impurity_measure="entropy", pruning=False, shuffle=False):
    """
    Creates a tree (of specified type) for iteration number of times, then for each tree performs tests of accuracy for test items
    :param X: dataset of items as list of list
    :param Y: target values for items as list
    :param iteration: number of chucks we split the dataset in as integer
    :param impurity_measure: use entropy or gini as string
    :param pruning: true or false for pruning as boolean
    :param shuffle: true or false for shuffling of dataset before pruning as boolean
    :return: accuracy for model between 0 and 1 as double
    """
    x_and_y = list(zip(X, Y))
    random.shuffle(x_and_y)
    X[:], Y[:] = zip(*x_and_y)

    X = np.array(X)
    Y = np.array(Y)
    X_split = np.array_split(X, iteration)
    Y_split = np.array_split(Y, iteration)

    n_sucsess_tot = 0
    n_tests_tot = 0

    for i in range(iteration): #All different iterations of splitting
        X_here_train = np.concatenate([X_split[j] for j in range(len(X_split)) if i != j])
        Y_here_train = np.concatenate([Y_split[j] for j in range(len(Y_split)) if i != j])
        X_here_validation = X_split[i]
        Y_here_validation = Y_split[i]

        tree = learn(X_here_train, Y_here_train, impurity_measure, pruning, shuffle)

        n_success= 0
        n_tests = len(X_here_validation)
        for j in range(n_tests): #Tests each item, and compares predicted value with correct target value
            val = predict(X_here_validation[j], tree)
            if val == Y_here_validation[j]:
                n_success = n_success + 1
        n_sucsess_tot += n_success
        n_tests_tot += n_tests

    return n_sucsess_tot/n_tests_tot


def get_sk_tree_accuracy(X_train,  Y_train, X_test, Y_test):
    """
    Creates a decision tree with sk learns implementation, using the test data. It then tests the tree with the
    test data. The function returns a measurement of the precisions.

    :param X_train: train dataset of items as list of list
    :param Y_train: train target values for items as list
    :param X_test: test dataset of items as list of list
    :param Y_test: testtarget values for items as list
    :return: a precision score between 0 and 1
    """
    sk_tree = tree.DecisionTreeClassifier()
    # Converting to numpy arrays and converting all chars to ints (because sklearn only accept int decisions)
    X_train = np.array(X_train)
    X_train = [[ord(x2) for x2 in x1] for x1 in X_train]

    Y_train = np.array(Y_train)
    Y_train = [ord(y1) for y1 in Y_train]

    X_test = np.array(X_test)
    X_test = [[ord(x2) for x2 in x1] for x1 in X_test]

    Y_test = np.array(Y_test)
    Y_test = [ord(y1) for y1 in Y_test]
    # converting end

    sk_tree.fit(X_train, Y_train)

    n_sucsess = 0
    n_tests = 0
    for i in range(len(X_test)):
        predicted_val = sk_tree.predict([X_test[i]])
        n_tests += 1
        if predicted_val[0] == Y_test[i]:
            n_sucsess += 1
    return (n_sucsess / n_tests)


def main():
    """
    Main method void
    """
    X, Y = read_from_file("agaricus-lepiota_data.txt")

    #Shuffels input data
    x_and_y = list(zip(X, Y))
    random.shuffle(x_and_y)
    X[:], Y[:] = zip(*x_and_y)

    l = len(X)
    split_on = (int(l * 0.6))
    X_train = X[:split_on]
    Y_train = Y[:split_on]
    X_test = X[split_on:]
    Y_test = Y[split_on:]

    #Performs cross val tests with different models
    print("Cross val test, entropy                  ", cross_val(X_train, Y_train, 3))
    print("Cross val test, gini                     ", cross_val(X_train, Y_train, 3, impurity_measure="gini"))
    print("Cross val test, entropy, pruning         ", cross_val(X_train, Y_train, 3, pruning=True))
    print("Cross val test, gini, pruning            ", cross_val(X_train, Y_train, 3, impurity_measure="gini", pruning=True, shuffle=True))
    print("Cross val test, entropy, pruning, shuffle", cross_val(X_train, Y_train, 3))
    print("Cross val test, gini, pruning, shuffle   ", cross_val(X_train, Y_train, 3, impurity_measure="gini", pruning=True, shuffle=True))

    #Creates a model of desired type which are to be tested with test data
    model = learn(X_train, Y_train)
    accuracy = test_model(model, X_test, Y_test)
    print("\nAccuracy for my model (entropy, pruning)", accuracy)

    #Calculates the accuray of the sk tree
    sk_tree_accuracy = get_sk_tree_accuracy(X_train, Y_train, X_test, Y_test)
    print("\nAccuracy for sk learn tree implementation", sk_tree_accuracy)


#Run main
main()
