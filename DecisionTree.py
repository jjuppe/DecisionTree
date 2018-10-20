import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from time import time


class Node:

    def __init__(self, threshold=None, attribute=None, classes=None, leaf=False):
        self.threshold = threshold
        self.attribute = attribute
        self.leaf = leaf
        self.left = None
        self.right = None
        self.classes = classes
        self.predict = None


def calculate_gini_index(data, threshold, attribute, y_attribute):
    classes = data[y_attribute].unique()
    y1 = data.loc[data[attribute] <= threshold, y_attribute].values
    y2 = data.loc[data[attribute] > threshold, y_attribute].values
    n_instances = len(data)

    gini = 0.0
    for group in [y1, y2]:
        size = len(group)
        if size == 0:
            continue

        score = 0.0
        for class_val in classes:
            proportion = np.count_nonzero(group == class_val) / size
            score += proportion ** 2
        gini += (1 - score) * (size / n_instances)

    return gini


def calculate_best_threshold(data, attribute, y_attribute):
    col = sorted(data[attribute].values)
    col = np.unique(col)

    best_threshold = 10e14
    best_gini = 1
    for idx, candidate in enumerate(col[1:]):
        thresh = (col[idx] + candidate) / 2
        gini_index = calculate_gini_index(data, thresh, attribute, y_attribute)
        print('Attribute: {}, gini: {}, treshold: {}'.format(attribute, gini_index, thresh))
        if gini_index < best_gini:
            best_threshold = thresh
            best_gini = gini_index
    print('Best threshold found: {} with gini-index of {} for attribute {}'.format(best_threshold,
                                                                                   round(best_gini, 3), attribute))
    return best_threshold, best_gini


class Tree:
    NR_NODES = 0

    def __init__(self, data, max_depth):
        self.root = None
        self.nr_nodes = 0
        self.depth = 0
        self.data = data
        self.max_depth = max_depth
        self.num_instances_per_class = len(data[' z'].unique())
        self.classes = sorted(data[' z'].unique())
        self.build_tree(data)

    def calc_num_instances_per_class(self, data):
        counter = np.zeros((self.num_instances_per_class))
        for idx, class_val in enumerate(self.classes):
            counter[idx] = data.loc[data[' z'] == class_val, ' z'].count()
        return counter

    def build_tree(self, data):
        attributes = list(data.keys())[:-1]
        instances_in_classes = self.calc_num_instances_per_class(data)
        # if there is already a pure distribution create node leaf
        if np.count_nonzero(instances_in_classes) <= 1:
            leaf = Node(None, None, leaf=True, classes=instances_in_classes)
            leaf.predict = self.classes[np.argmax(instances_in_classes)]
            Tree.NR_NODES += 1
            return leaf

        if self.depth >= self.max_depth:
            leaf = Node(None, None, leaf=True, classes=instances_in_classes)
            leaf.predict = self.classes[np.argmax(instances_in_classes)]
            Tree.NR_NODES += 1
            return leaf

        attr, thresh, _ = self.choose_attribute(attributes, data)
        node = Node(threshold=thresh, attribute=attr, classes=instances_in_classes)
        self.depth += 1
        if Tree.NR_NODES == 0:
            self.root = node
        Tree.NR_NODES += 1
        node.left = self.build_tree(data[data[attr] <= thresh])
        node.right = self.build_tree(data[data[attr] > thresh])
        self.depth -= 1
        return node

    def print_tree(self):
        root = self.root
        self.print_tree_recursive(root)

    def print_tree_recursive(self, node, depth=0):
        if not node.leaf:
            print(depth * " " + "{} <= {} ".format(node.attribute, node.threshold))
            self.print_tree_recursive(node.left, depth + 1)
            self.print_tree_recursive(node.right, depth + 1)
        else:
            print(depth * " " + str(node.predict))

    def choose_attribute(self, attributes, data):
        best_attribute = None
        best_gini = 1
        best_threshold = 10e14

        for attribute in attributes:
            threshold, gini = calculate_best_threshold(data, attribute, " z")
            if gini < best_gini:
                best_threshold = threshold
                best_gini = gini
                best_attribute = attribute
        print("Best attribute: {} with a gini of {} and threshold value {}".format(best_attribute, best_gini,
                                                                                   best_threshold))
        return best_attribute, best_threshold, best_gini

    def predict(self, data):
        result = list()
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        for idx, row in data.iterrows():
            result.append(self.predict_single_row(row, self.root))
        return result

    def predict_single_row(self, row, node):
        if node.leaf:
            return node.predict
        if row[node.attribute] <= node.threshold:
            return self.predict_single_row(row, node.left)
        if row[node.attribute] > node.threshold:
            return self.predict_single_row(row, node.right)

    def predict_probability(self, data):
        root = self.root
        pred = self.predict(data)
        if isinstance(data, dict):
            data = pd.DataFrame(data)
        result = list()
        for idx, row in data.iterrows():
            result.append(self.predict_prob_recursive(row, root, pred[idx]))
        return result

    def predict_prob_recursive(self, row, node, classification):
        if node.leaf:
            return node.classes[classification] / sum(node.classes)
        if row[node.attribute] <= node.threshold:
            return self.predict_prob_recursive(row, node.left, classification)
        if row[node.attribute] > node.threshold:
            return self.predict_prob_recursive(row, node.right, classification)


if __name__ == '__main__':
    data = pd.read_csv('01_homework_dataset.csv', sep=",")
    dt = Tree(data, max_depth=2)
    print("-" * 10 + "Printing tree" + 10 * "-")
    dt.print_tree()
    print("-" * 20 + "\n")
    start = time()
    pred = dt.predict(data)
    end = time()
    print("Prediction took {} seconds".format(end - start))
    print("Confusion Matrix: \n" + str(confusion_matrix(data[' z'], pred)))
    print('Overall accuracy: {}'.format(round(accuracy_score(data[' z'], pred), 3)))
    input = {'x1': [4.1, 6.1], ' x2': [-0.1, 0.4], ' x3': [2.2, 1.3]}
    prob = dt.predict_probability(input)
    pred_ex = dt.predict(input)
    print('Probability: {}, prediction: {}'.format(prob, pred_ex))

    # Comparison to sklearn kit
    print("\n" + "-" * 10 + "Comparison to DTClassifier from sklearn" + 10 * "-")
    dt_class = DecisionTreeClassifier(max_depth=2)
    dt_class.fit(data.loc[:, :' x3'], data[' z'])
    start_sk = time()
    pred_sk = dt_class.predict(data.loc[:, :' x3'])
    end_sk = time()
    # print(tree.export_graphviz(dt_class))
    print("Prediction took {} seconds".format(end_sk - start_sk))
    print('Overall accuracy sklearn: {}'.format(round(accuracy_score(data[' z'], pred_sk), 3)))
