from sklearn.model_selection import train_test_split
import multiprocessing
from xgboostlss.model import *
import pandas as pd
import pickle
import numpy as np

filename = "xgblss_model.pickle"
xgblss = pickle.load(open(filename, "rb"))
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

dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
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
xgblss_result = np.array(pred_quantiles)