import pickle
import numpy as np

filename = "svr_model.pickle"
SVRregressor = pickle.load(open(filename, "rb"))

img_array = [0.17031873,  0.60573858,  0.90266271, -0.30333629, -0.30378221, -0.34127123, -0.26558142, -0.04156759,  1., 0., 0., 0., 0.]
x = np.expand_dims(img_array, axis=0)
predictions = SVRregressor.predict(x)

print(predictions)