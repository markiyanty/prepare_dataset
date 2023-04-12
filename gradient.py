import pickle

import pylab
from sklearn.ensemble import GradientBoostingClassifier
from path import *
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, \
    f1_score


def train_model():
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TRAIN_MATRIX}", "rb") as file:
        train_matrix = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TRAIN}", "rb") as file:
        y_train = pickle.load(file)

    clf = GradientBoostingClassifier(n_estimators=20, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)
    clf.fit(train_matrix.toarray(), y_train.toarray().ravel())
    with open(f"1__gradient__v1.pkl", "wb") as file:
        pickle.dump(clf, file)



def evalaute_model():
    with open(f"1__gradient__v1.pkl", "rb") as file:
        clf = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_TEST_MATRIX}", "rb") as file:
        test_matrix = pickle.load(file)
    with open(f"{DATA_FOLDER}{FINAL_DATA_FOLDER}{WORK_Y_TEST}", "rb") as file:
        y_test = pickle.load(file)
    predicted = clf.predict(test_matrix.toarray())
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
    train_model()
    evalaute_model()
