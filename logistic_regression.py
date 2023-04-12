import pickle

import numpy as np
import pylab
from sklearn.linear_model import LogisticRegression
from path import *
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score, auc

RANDOM_STATE = 45


def train_model():
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TRAIN_MATRIX}", "rb") as file:
        train_matrix = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TRAIN}", "rb") as file:
        y_train = pickle.load(file)

    clf = LogisticRegression(random_state=RANDOM_STATE)
    clf.fit(train_matrix.toarray(), y_train.toarray().ravel())
    with open(f"1__logistic_regression__v2.pkl", "wb") as file:
        pickle.dump(clf, file)


def test_model():
    with open(f"1__logistic_regression__v2.pkl", "rb") as file:
        clf = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TEST_MATRIX}", "rb") as file:
        test_matrix = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TEST}", "rb") as file:
        y_test = pickle.load(file)
    predicted = clf.predict(test_matrix)
    print("Accurancy: ", np.mean(predicted == y_test))
    print(clf.coef_)


def evalaute_model():
    with open(f"1__logistic_regression__v2.pkl", "rb") as file:
        clf = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TEST_MATRIX}", "rb") as file:
        test_matrix = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TEST}", "rb") as file:
        y_test = pickle.load(file)
    predicted = clf.predict(test_matrix)
    cm = confusion_matrix(y_test.toarray(), predicted)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    pylab.show()
    accuracy = accuracy_score(y_test.toarray(), predicted)
    precision = precision_score(y_test.toarray(), predicted)
    recall = recall_score(y_test.toarray(), predicted)
    f1 = f1_score(y_test.toarray(), predicted)


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)


if __name__ == "__main__":
    # train_model()
    # test_model()
    evalaute_model()
