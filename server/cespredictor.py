import tensorflow as tf
import os
import pandas as pd
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class CESPred():
    def __init__(self):
        model_crack_dir = './models/bestdnn.h5'
        def piven_loss():
            # Define a dummy function to load model
            return None
        self.model = tf.keras.models.load_model(model_crack_dir, custom_objects={"piven_loss": piven_loss})
        self._load_dict()

    def _load_dict(self):
        self.mean = dict(pd.read_csv("./models/mean.csv", index_col=False).values)
        self.std = dict(pd.read_csv("./models/std.csv", index_col=False).values)
        self.min_val = dict(pd.read_csv("./models/min.csv", index_col=False).values)
        self.max_val = dict(pd.read_csv("./models/max.csv", index_col=False).values)

    def normalize(self, input_list):
        output_list = list(input_list)
        value_list = [
            'longitude',
            'latitude',
            'housing_median_age',
            'total_rooms',
            'total_bedrooms',
            'population',
            'households',
            'median_income'
        ]
        for i in range(2):
            key = value_list[i]
            max_v = self.max_val[key]
            min_v = self.min_val[key]
            output_list[i] = (input_list[i] - min_v) / (max_v - min_v)

        for i in range(2, 8):
            key = value_list[i]
            mean_val = self.mean[key]
            std_val = self.std[key]
            output_list[i] = (input_list[i] - mean_val) / std_val

        return output_list

    def denormalize(self, predictions):
        prediction = predictions[0]
        mean_val = self.mean['median_ces']
        std_val = self.std['median_ces']

        y_u_pred = prediction[0]
        y_l_pred = prediction[1]
        y_v_pred = prediction[2]

        y_piven = y_v_pred * y_u_pred + (1 - y_v_pred) * y_l_pred

        y_est = y_piven * std_val + mean_val
        y_upper = y_u_pred * std_val + mean_val
        y_lower = y_l_pred * std_val + mean_val

        return y_est, y_upper, y_lower

    def predict(self, input_list):
        input_data = self.normalize(input_list)
        x = np.expand_dims(input_data, axis=0)
        predictions = self.model.predict(x, verbose=0)
        y_est, y_upper, y_lower = self.denormalize(predictions)
        return y_est, y_upper, y_lower

