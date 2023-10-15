import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

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


# Training
SVRregressor = SVR(kernel = 'rbf')
SVRregressor.fit(X_train, y_train)


# Save model
import pickle
filename = "svr_model.pickle"
pickle.dump(SVRregressor, open(filename, "wb"))

# Validation
y_hat = SVRregressor.predict(X_test)
mse = sum((y_test-y_hat)**2)/len(y_test)
print(mse)




