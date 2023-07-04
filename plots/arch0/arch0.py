import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, node_lenght,dropout):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.LSTM(node_lenght, input_shape=(node_lenght, 1), activation="LeakyReLU"),
            layers.Dropout(dropout),
            layers.Dense(32, activation="LeakyReLU"),
            layers.Dropout(dropout),
            layers.Dense(16, activation="LeakyReLU"),
            layers.Dense(16, activation="LeakyReLU")])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="LeakyReLU"),
            layers.Dense(16, activation="LeakyReLU"),
            layers.Dropout(dropout),
            layers.Dense( 32, activation="LeakyReLU"),
            layers.Dropout(dropout),
            layers.LSTM(node_lenght, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

