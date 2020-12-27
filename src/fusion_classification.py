import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report

from fusion import FeatureFusion

from color import Color
from daisy import Daisy
from edge import Edge
from HOG import HOG

from DB import Database, DatabaseType
from evaluate import (
    infer,
    AP
)



# variable
d_type = 'd1' # d1 is the method to calculate distances
depth  = 5 # 5 closest neighbors


def evaluate_dataset(samples_train, samples_validation, classes):
    # confusion matrix init
    y_true = []
    y_pred = []

    for query in samples_validation:
        # infer the image and return the $depth closest neighbors
        _, results = infer(query, samples=samples_train, depth=depth, d_type=d_type) # results is an array with images distance and class (we don't use AP from the infer method because it doesn't use the weigthed KNN algorithm to predict class, it only predict that neighbors are the good ones)

        # weighted class (knn algorithm on the $depth closest neighbors)
        pred = weighted_mode(
            [sub["cls"] for sub in results],
            np.reciprocal([sub["dis"] for sub in results]),
        )

        # add elements to calculate metrics later
        y_true.append(query["cls"])
        y_pred.append(pred[0][0])



    # calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # calculate recall
    recall = recall_score(y_true, y_pred, average=None)

    # calculate precision
    precision = precision_score(y_true, y_pred, average=None)

    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(classes))

    # make report
    report = classification_report(y_true, y_pred, target_names=list(classes))

    return accuracy, recall, precision, conf_matrix, report # return average precision and precision for each class & confusion matrix


def evaluate_image(samples_train, query):
    # infer the image and return the $depth closest neighbors
    _, results = infer(query, samples=samples_train, depth=depth, d_type=d_type) # results is an array with images distance and class (we don't use AP from the infer method because it doesn't use the weigthed KNN algorithm to predict class, it only predict that neighbors are the good ones)

    # weighted class (knn algorithm on the $depth closest neighbors)
    pred = weighted_mode(
        [sub["cls"] for sub in results],
        np.reciprocal([sub["dis"] for sub in results]),
    )

    # exctract predicted class
    predicted_class = pred[0][0]

    # return image's predicted class
    return predicted_class


if __name__ == "__main__":
    # pick datasets
    print("=== Preparing datasets ===")
    print("Preparing dataset objects.")
    db_train = Database(DatabaseType.TRAIN)
    db_validation = Database(DatabaseType.VALIDATION)
    db_test = Database(DatabaseType.TEST)



    # train
    print("=== Training ===")
    print("Training...")
    start_time = time.time()
    fusion = FeatureFusion(features=['color', 'daisy'])
    samples_train = fusion.make_samples(db_train)
    train_time = round((time.time() - start_time) / 60, 3) # get train time in minute
    print("Trained in %smins." % (train_time))



    # predict one image
    print("=== Predict one image ===")
    filename = "database\\test\\obj_car\\29031.jpg"
    expected_cls = "obj_car"
    print("Performing image: %s" % filename)
    print("Expected class: %s" % expected_cls)
    query = fusion.make_samples_image(filename)[0]
    predicted_class = evaluate_image(samples_train, query)
    print("Predicted class: %s" % predicted_class)



    # evaluate model on dataset
    print("=== Model evaluation ===")
    print("Validation dataset in progress...")
    samples_validation = fusion.make_samples(db_validation, cache=False)
    accuracy, _, _, conf_matrix, report = evaluate_dataset(samples_train, samples_validation, db_validation.classes)

    # show model metrics
    print(report)
    print("Model accuracy: %f" % accuracy)

    # render confusion matrix
    dataframe = pd.DataFrame(conf_matrix, index=[i for i in db_validation.classes], columns=[i for i in db_validation.classes])
    plt.figure(figsize=(8, 8))
    sn.heatmap(dataframe, annot=True, fmt='d')
    plt.show()