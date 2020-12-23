# -*- coding: utf-8 -*-

from __future__ import print_function

import pandas as pd
import os
from enum import Enum


class DatabaseType(Enum):
  TRAIN      = "train"
  VALIDATION = "validation"
  TEST       = "test"

class Database(object):

  def __init__(self, suffix=None, dataset_dir="database"):
    if suffix != None:
      self.dir = os.path.join(dataset_dir, suffix.value)
      self.csv = "data_{}.csv".format(suffix.value)
    else:
      self.dir = dataset_dir
      self.csv = "data.csv"

    self._gen_csv()
    self.data = pd.read_csv(self.csv)
    self.classes = set(self.data["cls"])

  def _gen_csv(self):
    if os.path.exists(self.csv):
      return
    with open(self.csv, 'w', encoding='UTF-8') as f:
      f.write("img,cls")
      for root, _, files in os.walk(self.dir, topdown=False):
        cls = root.split(os.sep)[-1]
        for name in files:
          if not name.endswith('.jpg'):
            continue
          img = os.path.join(root, name)
          f.write("\n{},{}".format(img, cls))

  def __len__(self):
    return len(self.data)

  def get_class(self):
    return self.classes

  def get_data(self):
    return self.data


if __name__ == "__main__":
  db = Database()
  data = db.get_data()
  classes = db.get_class()

  print("DB length:", len(db))
  print(classes)
