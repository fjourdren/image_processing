import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import os

result_path = os.path.join("result")
if not os.path.exists(result_path):
    os.makedirs(result_path)

def render_confusion_matrix(conf_matrix, labels):
    dataframe = pd.DataFrame(conf_matrix, index=[i for i in labels], columns=[i for i in labels]) # prapare data
    plt.figure(figsize=(8, 8)) # make figures
    sn.heatmap(dataframe, annot=True, fmt='d') # config seaborn

    # save plot on disk
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    print("confusion_matrix.png saved")

    # show confusion matrix to user
    plt.show()