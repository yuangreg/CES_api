# Reference: https://github.com/StatMixedML/XGBoostLSS
# pip install git+https://github.com/StatMixedML/XGBoostLSS.git
# pip install git+https://github.com/dsgibbons/shap.git

from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from xgboostlss.distributions.Mixture import *

from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
import pandas as pd

# Set seed of cpu for experiment reproductivity.
seed = 123
n_cpu = multiprocessing.cpu_count()
n_trials = 10

# Load training data
df = pd.read_csv("etl_data.csv")
X = df.drop('median_ces', axis=1)
Y = df['median_ces']

# convert to numpy array
X_array = np.array(X)
Y_array = np.array(Y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size=0.2, random_state=seed)

# Prepair input stream
dtrain = xgb.DMatrix(X_train, label=y_train, nthread=n_cpu)
dtest = xgb.DMatrix(X_test, nthread=n_cpu)

# Specifies a mixture of Gaussians. See ?Mixture for an overview.
xgblss = XGBoostLSS(
    Mixture(
        Gaussian(response_fn="softplus"),
        M = 5,
        tau=1.0,
        hessian_mode="individual",
    )
)

# Parameter optimization
param_dict = {
    "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True}],
    "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False}],
    "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True}],
    "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False}],
    "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True}],
    "booster":          ["categorical", ["gbtree"]],
}

np.random.seed(seed)
opt_param = xgblss.hyper_opt(param_dict,
                             dtrain,
                             num_boost_round=100,        # Number of boosting iterations.
                             nfold=5,                    # Number of cv-folds.
                             early_stopping_rounds=20,   # Number of early-stopping rounds
                             max_minutes=60,             # Time budget in minutes, i.e., stop study after the given number of minutes.
                             n_trials=n_trials,          # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                             silence=True,               # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
                             seed=123,                   # Seed used to generate cv-folds.
                             hp_seed=None                # Seed for random number generator used in the Bayesian hyperparameter search.
                            )

opt_params = opt_param.copy()
n_rounds = opt_params["opt_rounds"]
del opt_params["opt_rounds"]

# Train Model with optimized hyperparameters
xgblss.train(opt_params,
             dtrain,
             num_boost_round=n_rounds,
             )

# Save model
import pickle
filename = "xgblss_model.pickle"
pickle.dump(xgblss, open(filename, "wb"))