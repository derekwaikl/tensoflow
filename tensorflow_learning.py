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

def my_input_fn(features,targets,batch_size=1,shuffle=True,num_epochs=None):
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

_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature,targets),steps=100)

# Create an input function for predictions
# Note: Since we are making just one prediction for each example, we
# don't need to repeat or shuffle the data here.
prediction_input_fn = lambda: my_input_fn (my_feature, targets, num_epochs=1,shuffle= False)

#Call predict() on the linear_regressor to make predictions.
predictions = linear_regressor.predict(input_fn=prediction_input_fn)

# Format predictions as a Numpy array, so we can calculate error metrics.
predictions = np.array([item['predictions'][0] for item in predictions])

#Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error(on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error(on training data): %0.3f" % root_mean_squared_error)

min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print ("Min. Median House value: %0.3f" % min_house_value)
print( "Max. Median House valye: %0.3f" % max_house_value)
print ("Difference between min. and max.: %0.3f" % min_max_difference)
print( "Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pd.DataFrame()
calibration_data ["predictions"] = pd.Series(predictions)
calibration_data ["targets"] = pd.Series(targets)
calibration_data.describe()

sample = california_housing_dataframe.sample(n=300)

# Get the min and max total rooms values.
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

# Retrieve the final weight and bias generated during training
weight - linear_regressor.get_variable_value('linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

# Get the predicted median_house_values for the min and max total_rooms values
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0,y_0) to (x_1,y_1)
plt.plot([x_0,x_1],[y_0,Y_1], c='r')

#Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")

# Plot a scatter plot from our data sample
plt.scatter(sample["total_rooms"],sample["median_house_value"])

# Display graph
plt.show()
