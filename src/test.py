from fusion import FeatureFusion

from color import Color
from daisy import Daisy
from edge import Edge
from HOG import HOG

from DB import Database, DatabaseType
from evaluate import (
    distance, 
    evaluate_class,
    infer
)

# variable
d_type = 'd1'
depth    = 5


if __name__ == "__main__":
    db_train = Database(DatabaseType.TRAIN)
    db_validation = Database(DatabaseType.VALIDATION)
    db_test = Database(DatabaseType.TEST)

    # train all algorithms (put samples in cache for next usage)
    print("=== Making sample for each algorithm ===")
    Color().make_samples(db_train)
    Daisy().make_samples(db_train)
    # Edge().make_samples(db_train)
    # HOG().make_samples(db_train)

    # fusion
    # evaluate