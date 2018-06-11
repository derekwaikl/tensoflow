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

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"]/=1000.0
california_housing_dataframe
california_housing_dataframe.describe()

# Define the input feature: total rooms.
my_feature = california_housing_dataframe[["total_rooms"]]

# configure a numeric feature column for total rooms.
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

#define the label
targets = california_housing_dataframe["median_house_value"]

# Use gradient descent as the optimizer  for training the model.
my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)

#Configure the liner regression model with our feature columns and optimizer
#Set a learning rate of 0.0000001 for Gradient descent

linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns,optimizer=my_optimizer)

def my_input_fn(features,targets,batch,batch_size=1,shuffle=True,num_epochs=None):
    '''Train a linear regression model of one feature

    Args:
    features:pandas Dataframe of features
    targets: pandas Dataframe of targets
    batch_size: Size of batches to be passed to the model
    shuffle: True or False. Whether to shuffle the data.
    num_epochs:Number of epochs for which data should be repeated. None = Repeat indefinitely
    Returns:
    Tuple of  (features,labels) for next data batch
    '''
    #Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key, value in dict(features).items()}

    #Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices ((features, targets)) #warning, 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    #Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)

    # Return the next batchof data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

    _ = linear_regressor.train(input_fn = lambda: my_input_fn(my_feature,targets),steps=100)


