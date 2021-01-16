import numpy as np
import time
import os
import shutil
from sklearn.utils.extmath import weighted_mode
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

from fusion import FeatureFusion

from color import Color
from daisy import Daisy
from edge import Edge
from HOG import HOG

from DB import Database, DatabaseType
from plot_utils import render_confusion_matrix
from evaluate import (
    infer,
    AP
)



# variable
d_type = 'd1' # d1 is the method to calculate distances
depth  = 4 # 4 closest neighbors

path_results = os.path.join("result", "fusion_classification")


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

    # calculate F1 score
    f1 = f1_score(y_true, y_pred, average=None)

    # calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(classes))

    # make report
    report = classification_report(y_true, y_pred, target_names=list(classes))

    return accuracy, recall, precision, f1, conf_matrix, report # return average precision and precision for each class & confusion matrix


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
    db_train      = Database(DatabaseType.TRAIN)
    db_validation = Database(DatabaseType.VALIDATION)
    db_test       = Database(DatabaseType.TEST)



    # train
    print("=== Training ===")
    print("Training...")
    start_time = time.time()
    fusion = FeatureFusion(features=['color', 'daisy'])
    samples_train = fusion.make_samples(db_train)
    train_time = round((time.time() - start_time) / 60, 3) # get train time in minute
    print("Trained in %smin." % (train_time))



    # predict a random image
    print("=== Predict a random image ===")
    start_time = time.time()
    filename = np.random.choice(db_validation.get_data()['img']) # take a random image to process in the test dataset (high probability of bad classification)
    expected_cls = filename.split(os.sep)[-2] # get random image class
    print("Performing image: %s" % filename)
    print("Expected class: %s" % expected_cls)
    query = fusion.make_samples_image(filename)[0]
    predicted_class = evaluate_image(samples_train, query)
    print("Predicted class: %s" % predicted_class)
    execution_time = round((time.time() - start_time) / 60, 3) # get execution time in minute
    print("Predicted in %smin." % (execution_time))



    # evaluate model on dataset
    print("=== Model evaluation ===")
    print("Validation dataset in progress...")
    start_time = time.time()
    samples_validation = fusion.make_samples(db_validation, cache=False)
    labels = sorted(db_validation.classes) # sort labels to improve the figure rendering
    accuracy, recall, precision, f1, conf_matrix, report = evaluate_dataset(samples_train, samples_validation, labels)
    execution_time = round((time.time() - start_time) / 60, 3) # get execution time in minute
    seconds_per_image = (execution_time/len(db_validation.get_data()['img']))
    imgs_per_second = 1/seconds_per_image
    print("Evaluated %d images in %smin (%fs/img or %fimgs/s)." % (len(db_validation.get_data()['img']), execution_time, seconds_per_image, imgs_per_second))


    # show model metrics
    print("- Classes metrics:")
    print(report)

    print("- Model metrics:")
    print("    * Model accuracy:  %f" % accuracy)
    print("    * Model recall:    %f" % np.mean(recall))
    print("    * Model precision: %f" % np.mean(precision))
    print("    * Model f1:        %f" % np.mean(f1))



    # predict on all images and move in result folder
    print("=== File saving evaluation ===")
    print("Performing evaluation with file copy in result directory...")
    # create result dir
    if not os.path.exists(path_results):
        os.makedirs(path_results)    
    
    # create class result dirs
    for cls in labels:
        path_cls = os.path.join(path_results, cls)
        if not os.path.exists(path_cls):
            os.makedirs(path_cls)

    # make a classification for each file in test dataset
    for filename in db_validation.get_data()['img']:
        query = fusion.make_samples_image(filename)[0]
        predicted_class = evaluate_image(samples_train, query)

        # copy file after classification
        shutil.copy(filename, os.path.join(path_results, predicted_class, os.path.basename(filename)))

    print("You can look images in details in ", path_results)


    # render confusion matrix to the user
    render_confusion_matrix(conf_matrix, labels)