# Run the following to install xgboost packages
# pip install git+https://github.com/StatMixedML/XGBoostLSS.git
# pip install git+https://github.com/dsgibbons/shap.git

from xgboostlss.model import *
from xgboostlss.distributions.Gaussian import *
from xgboostlss.distributions.Mixture import *
from xgboostlss.distributions.mixture_distribution_utils import MixtureDistributionClass
from sklearn.model_selection import train_test_split
import multiprocessing

import numpy as np
import pandas as pd

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


mix_dist_class = MixtureDistributionClass()

candidate_distributions = [
    Mixture(Gaussian(response_fn="softplus"), M = 2),
    Mixture(Gaussian(response_fn="softplus"), M = 3),
    Mixture(Gaussian(response_fn="softplus"), M = 4),
    Mixture(Gaussian(response_fn="softplus"), M = 5),
    Mixture(Gaussian(response_fn="softplus"), M = 6),
]

dist_nll = mix_dist_class.dist_select(target=y_train, candidate_distributions=candidate_distributions, max_iter=50, plot=True, figure_size=(8, 5))
print(dist_nll)