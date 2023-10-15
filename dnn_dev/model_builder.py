import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# General Settings
seed = 123

def build_model(dropout_rate):
    # model structure
    inputs = tf.keras.layers.Input(shape=(13,))
    # Feature layers
    x = tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=seed))(
        inputs)

    x = tf.keras.layers.Dense(200, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=seed))(x)

    x = tf.keras.layers.Dense(200, activation='relu',
                             kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=seed))(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(100, activation='relu',
                              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=seed))(x)

    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    pi = tf.keras.layers.Dense(2, activation='linear', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.2, seed=seed))(x)
    v = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))(x)
    piven_out = tf.keras.layers.Concatenate(name='piven_out')([pi, v])

    model = tf.keras.models.Model(inputs=inputs, outputs=[piven_out], name='piven_model')

    return model