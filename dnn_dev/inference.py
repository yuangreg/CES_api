import tensorflow as tf
import numpy as np
from lossfunction import piven_loss

model_crack_dir = 'bestdnn.h5'
model = tf.keras.models.load_model(model_crack_dir, custom_objects={"piven_loss": piven_loss})

# Single input case
img_array = [0.7031873,  0.60573858,  0.90266271, -0.30333629, -0.30378221, -0.34127123, -0.26558142, -0.04156759,  1., 0., 0., 0., 0.]
x = np.expand_dims(img_array, axis=0)
predictions = model.predict(x, verbose=0)

y_u_pred = predictions[:,0]
y_l_pred = predictions[:,1]
y_v_pred = predictions[:,2]

y_piven = y_v_pred * y_u_pred + (1 - y_v_pred) * y_l_pred
print(y_piven)