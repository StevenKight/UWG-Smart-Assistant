import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

__author__ = "Steven Kight"
__version__ = "1.0"
__pylint__ = "2.14.4"

def get_known_info():
    """
    Imports known encodings and names.

    ### Return
        A list of all encodings and a dictionary with names as keys and encodings as values.
    """

    header = ["Frequency mean","Frequency mode","Frequency sd","Frequency skew",
            "Frequency kurt","Frequency Confidence Interval Low",
            "Frequency Confidence Interval High","Amplitude mean","Amplitude sd",
            "Amplitude skew","Amplitude kurt","Amplitude Confidence Interval Low",
            "Amplitude Confidence Interval High","Name"]

    dataframe = pd.read_csv("smart_assistant/recognition/audio/Models/data.csv",
        names=header)

    labels = dataframe.pop('Name')
    labels.pop(0)
    labels = labels.values.tolist()

    header.pop(len(header)-1)
    numeric_features = dataframe[header]
    numeric_features = numeric_features.iloc[1: , :]
    numeric_features = numeric_features.values.tolist()

    binary_labels = []
    for name1 in labels:

        if name1 == "Steven-Kight":
            binary_labels.append(1)
        else:
            binary_labels.append(0)

    return numeric_features, binary_labels


def train():
    """
    Uses a list of different Binary Models to test for best accuracy
    across mulitple models to find best one to use for each individual
    """

    x_train, y_train = get_known_info()

    classifiers = [
        SVC(kernel="linear", C=0.025), # Potential
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000), # Potential
        AdaBoostClassifier()
    ]

    models = []
    for clf in classifiers:
        models.append(clf.fit(x_train, y_train))

    print("models created")

    return models


if __name__ == "__main__":
    models = train()

    test = [[1536.99,200.11,2307.65,14.67,306.65,1520.78,
        1553.21,188.91,692.39,7.32,74.48,185.46,192.35]]
    for model in models:
        print(model.predict(test))
