import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# General Settings
seed = 123

# Load training data
df = pd.read_csv("etl_data.csv")
X = df.drop('median_ces', axis=1)
Y = df['median_ces']

# convert to numpy array
X_array = np.array(X)
Y_array = np.array(Y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size=0.2, random_state=seed)

model_crack_dir = 'bestdnn.h5'
def piven_loss():
  # Define a dummy function to load model
  return None
model = tf.keras.models.load_model(model_crack_dir, custom_objects={"piven_loss": piven_loss})

predictions = model.predict(X_test, verbose=0)
# predictions format [y_u, y_l, y_v]
y_u_pred = predictions[:,0]
y_l_pred = predictions[:,1]
y_v_pred = predictions[:,2]

y_piven = y_v_pred * y_u_pred + (1 - y_v_pred) * y_l_pred

# MSE performance
def mse(y, yhat):
  return sum((y-yhat)**2)/len(y_test)

mse_dnn_pivon = mse(y_test, y_piven)
print(f'MSE of SVR is {mse_dnn_pivon}')

# Interval estimation performance
K_u = y_u_pred > y_test
K_l = y_l_pred < y_test
Prob_dnn = np.mean(K_u * K_l)
Len_dnn = round(np.mean(y_u_pred - y_l_pred), 3)

print('Probability of containing target:', Prob_dnn)
print('Average estimate interval length:', Len_dnn)