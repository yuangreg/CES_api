import pickle
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import multiprocessing
from xgboostlss.model import *
from sklearn.model_selection import train_test_split

# General Settings
seed = 123
n_cpu = multiprocessing.cpu_count()

# Load training data
df = pd.read_csv("etl_data.csv")
X = df.drop('median_ces', axis=1)
Y = df['median_ces']

# convert to numpy array
X_array = np.array(X)
Y_array = np.array(Y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size=0.2, random_state=seed)
print(f'Test data: {X_test.shape}, target: {y_test.shape}')

# SVR Model
filename = 'svr_model.pickle'
SVRregressor = pickle.load(open(filename, "rb"))

predictions_SVR = SVRregressor.predict(X_test)
print(predictions_SVR)


# DNN model
def piven_loss():
  # Define a dummy function to load model
  return None
model = tf.keras.models.load_model('bestdnn.h5', custom_objects={"piven_loss": piven_loss})

predictions = model.predict(X_test, verbose=0)
# predictions format [y_u, y_l, y_v]
y_u_pred = predictions[:,0]
y_l_pred = predictions[:,1]
y_v_pred = predictions[:,2]

y_middle = 0.5 * y_u_pred + 0.5 * y_l_pred
y_piven = y_v_pred * y_u_pred + (1 - y_v_pred) * y_l_pred
print(y_piven)

filename = "xgblss_model.pickle"
xgblss = pickle.load(open(filename, "rb"))

# Convert to stream
dtest = xgb.DMatrix(X_test, nthread=n_cpu)

# Set seed for reproducibility
torch.manual_seed(123)

# Number of samples to draw from predicted distribution
n_samples = 1000
quant_sel = [0.05, 0.95] # Quantiles to calculate from predicted distribution

# Calculate quantiles from predicted distribution
pred_quantiles = xgblss.predict(dtest,
                                pred_type="quantiles",
                                n_samples=n_samples,
                                quantiles=quant_sel)

print(pred_quantiles)

####################################################################
# Compare MSE
def mse(y, yhat):
  return sum((y-yhat)**2)/len(y_test)

# MSE of SVR
mse_svr = mse(y_test, predictions_SVR)
print(f'MSE of SVR is {mse_svr}')

# MSE of DNN
mse_dnn_pivon = mse(y_test, y_piven)
print(f'MSE of DNN is {mse_dnn_pivon}')

####################################################################
# Compare Interval estimation

# DNN Model
K_u = y_u_pred > y_test
K_l = y_l_pred < y_test
Prob_dnn = np.mean(K_u * K_l)
Len_dnn = round(np.mean(y_u_pred - y_l_pred), 3)

print('Probability of containing target of DNN:', Prob_dnn)
print('Average estimate interval length of DNN:', Len_dnn)

# Generative Model xgblss_model
xgblss_result = np.array(pred_quantiles)
xgb_l_pred = xgblss_result[:,0]
xgb_u_pred = xgblss_result[:,1]
K_u = xgb_u_pred > y_test
K_l = xgb_l_pred < y_test
Prob_xgb = np.mean(K_u * K_l)
Len_xgb = round(np.mean(xgb_u_pred - xgb_l_pred), 3)

print('Probability of containing target of XGB:', Prob_xgb)
print('Average estimate interval length of XGB:', Len_xgb)