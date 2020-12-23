import numpy as np
import time
from collections import defaultdict
from sklearn.utils.extmath import weighted_mode

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


def evaluate_dataset(samples_train, samples_validation):
    # array to calculate classes' AP
    tmp_class_aps = defaultdict(lambda: 0)
    tmp_class_img_nb = defaultdict(lambda: 0)

    for query in samples_validation:
        # infer the image and return the $depth closest neighbors
        _, results = infer(query, samples=samples_train, depth=depth, d_type=d_type) # results is an array with images distance and class (we don't use AP from the infer method because it doesn't use the weigthed KNN algorithm to predict class, it only predict that neighbors are the good ones)

        # weighted class (knn algorithm on the $depth closest neighbors)
        pred = weighted_mode(
            [sub["cls"] for sub in results],
            np.reciprocal([sub["dis"] for sub in results]),
        )

        # make result query for that pic to calculate AP (average precision)
        validation_query = [{
            'cls': pred[0][0],
            'dis': pred[1][0]
        }]

        # evaluate if the class is the good one or not
        image_ap = float(AP(query["cls"], validation_query, sort=False)) # calculate precision on the image (here 0 if it's not the good class, otherwise 1) => we calculate a new precision only on the class result from the KNN algorithm

        # add to array to be able to make a confusion matrix
        tmp_class_img_nb[query["cls"]] += 1 # increments the number of image waited in the class
        tmp_class_aps[query["cls"]] += image_ap # add the image precision result


    # calculate for each class the average precision
    classes_aps = {}

    total_aps = 0
    total_img = 0
    for cls in tmp_class_aps.keys():
        classes_aps[cls] = tmp_class_aps[cls] / tmp_class_img_nb[cls]
        
        # add elements in acumulators make average precision later
        total_aps += tmp_class_aps[cls]
        total_img += tmp_class_img_nb[cls]

    # calculate average precision on all classes
    ap = total_aps / total_img

    return ap, classes_aps # return average precision and precision for each class


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
    samples_validation = fusion.make_samples(db_validation)
    ap, precision_classes = evaluate_dataset(samples_train, samples_validation)

    # show precision results to user
    print("* Precision results :")
    for cls in precision_classes.keys():
        print("    - %s: %f" % (cls, precision_classes[cls]))
    
    print("\n* Classification average precision (AP): %f" % ap) # sample_analysis