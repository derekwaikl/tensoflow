import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

californa_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/calfornia_housing_train.csv", sep=",")

print('californa_housing_dataframe')
# california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
# california_housing_dataframe["median_house_value"]/=1000.0
# california_housing_dataframe
# california_housing_dataframe.describe()

## Define the input feature : total rooms.
# my_feature = california_housing_dataframe[["total_rooms"]]

## configure a numeric feature column for total rooms
#feature_columns =[tf.feature_column.numeric_column("total_rooms")]

#define the label
#targets = calfornia_housing_dataframe["median_house_value"]

