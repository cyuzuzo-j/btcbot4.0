import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import keras_tuner
import keras_nlp
NBATCHES = 421
BOTTELNECK = 20


def load_dataset(year=None):
    if year is None:
        dataframe = pd.read_csv("reinforcemnt learning/dataset/BTC-Hourly.csv")
    else:
        dataframe = pd.read_csv(f"reinforcemnt learning/dataset/BTC-{year}min.csv")
    dataframe = dataframe.sort_values(by=['unix'])
    # dataframe.iloc[:,3:] = fftn(dataframe.iloc[:,3:])
    batches = np.split(dataframe, NBATCHES)
    train_data, test_data, _, _ = train_test_split(batches, batches, test_size=0.2)
    return train_data, test_data


def normalize_batches(batches):
    normalized_batches = []
    for batch in batches:
        values_dataset = np.log10(batch.iloc[:, 3:] + 1)
        time_dataset = batch.iloc[:, :3]  # and symbol
        values_dataset = (values_dataset - values_dataset.min()) / (values_dataset.max() - values_dataset.min())
        normalized_batches.append(pd.concat([time_dataset, values_dataset], axis=1))

    return normalized_batches


def nullize_batch(batches):
    # voegt een aantal nullen toe aangezien in de ppo het neuraal network ook een hoop waardes krijgt die nul zijn
    new_batches = []
    new_batches_values = []

    for batch in batches:
        place = random.randint(0, 70)
        new_batches.append(
            pd.concat([batch[:place], pd.concat([batch.iloc[place:, :3], batch.iloc[place:, 3:] * 0], axis=1)], axis=0))
        new_batches_values.append(pd.concat([batch.iloc[:place, 3:], batch.iloc[place:, 3:] * 0], axis=0))
    return new_batches, new_batches_values


def visualize_dataset(dataset):
    for batch in dataset:
        plt.grid()
        plt.plot(batch["unix"], batch.iloc[:, 3])
        plt.plot(batch["unix"], batch.iloc[:, 4])
        plt.plot(batch["unix"], batch.iloc[:, 5])
        plt.plot(batch["unix"], batch.iloc[:, 6])
        plt.plot(batch["unix"], batch.iloc[:, 7])
        plt.title("vibes")
    plt.show()


train_batches, test_batches = load_dataset()
train_batches = normalize_batches(train_batches)
test_batches = normalize_batches(test_batches)
train_values_batches = np.array([data.iloc[:,3:]  for data in train_batches])
test_values_batches = np.array([data.iloc[:,3:]  for data in test_batches])
test_values_batches_out = test_values_batches[:, :, 0]
train_values_batches_out = train_values_batches[:, :, 0]


class Autoencoder(Model):
    def __init__(self, hp):
        super(Autoencoder, self).__init__()
        self.base_noise = tf.keras.layers.GaussianNoise(hp.Int("units", min_value=1, max_value=100, step=10))

        # basenoise positional encode
        self.query_lstm1 = tf.keras.layers.LSTM(2, hp.Choice("activationLstm", ["LeakyReLU", "relu", "sigmoid", "tanh"]), input_shape=(2, 79), return_sequences=True, )
        self.key_lstm1 = tf.keras.layers.LSTM(2, hp.Choice("activationLstm", ["LeakyReLU", "relu", "sigmoid", "tanh"]), input_shape=(2, 79), return_sequences=True)

        self.flat1 = tf.keras.layers.Flatten()
        self.attention1 = tf.keras.layers.AdditiveAttention()  # prijs
        self.query1 = tf.keras.layers.Dense(79,hp.Choice("activationDense1",["LeakyReLU", "relu","sigmoid","tanh"]))
        self.key1 = tf.keras.layers.Dense(79,hp.Choice("activationDense2",["LeakyReLU", "relu","sigmoid","tanh"]))
        self.output1= tf.keras.layers.Dense(79,hp.Choice("activationDense3",["LeakyReLU", "relu","sigmoid","tanh"]))

        self.query_lstm1 = tf.keras.layers.LSTM(2, hp.Choice("activationLstm", ["LeakyReLU", "relu", "sigmoid", "tanh"]), input_shape=(2, 79), return_sequences=True, )
        self.key_lstm1 = tf.keras.layers.LSTM(2, hp.Choice("activationLstm", ["LeakyReLU", "relu", "sigmoid", "tanh"]), input_shape=(2, 79), return_sequences=True)

        self.flat2 = tf.keras.layers.Flatten()
        self.attention2 = tf.keras.layers.AdditiveAttention()  # prijs
        self.query2 = tf.keras.layers.Dense(79,hp.Choice("activationDense1",["LeakyReLU", "relu","sigmoid","tanh"]))
        self.key2 = tf.keras.layers.Dense(79,hp.Choice("activationDense2",["LeakyReLU", "relu","sigmoid","tanh"]))
        self.output2= tf.keras.layers.Dense(79,hp.Choice("activationDense3",["LeakyReLU", "relu","sigmoid","tanh"]))

    def call(self, input):
        x = input[:, :, 0]*0
        x = self.base_noise(x,training = True)
        print(x.shape)


        a = self.query_lstm1(tf.stack([x, input[:, :, 0]], 2))
        b = self.key_lstm1(tf.stack([x, input[:, :, 0]], 2))
        a = self.flat1(a)
        b= self.flat1(b)
        a = self.query1(a)
        b= self.key1(b)
        x = self.attention1([a,b,x])
        x = self.output1(x)

        a = self.query_lstm2(tf.stack([x, input[:, :, 0]], 2))
        b = self.key_lstm2(tf.stack([x, input[:, :, 0]], 2))
        a = self.flat2(a)
        b = self.flat2(b)
        a = self.query2(a)
        b = self.key2(b)
        x = self.attention2([a, b, x])
        x = self.output2(x)
        return x


n = 8


tuner = keras_tuner.BayesianOptimization(
    lambda x:Autoencoder(x),
    objective='val_loss',
    max_trials=50)

tuner.search(train_values_batches[0:len(train_values_batches)], train_values_batches[0:len(train_values_batches)], epochs=600,
             validation_data=(test_values_batches, test_values_batches))
autoencoder = tuner.get_best_models()[0]
autoencoder.compile("adam","mse")
history = autoencoder.fit(train_values_batches[0:len(train_values_batches)], train_values_batches_out[0:len(train_values_batches)],
                          epochs=2000,
                          validation_data=(test_values_batches, test_values_batches_out),
                          shuffle=False)
autoencoder.summary()

plt.subplot(1, 2, 1)
plt.plot(history.history["val_loss"], label=f"Validation Loss")
plt.plot(history.history["loss"], label=f"train Loss")
plt.legend()
test_values_batches_sample = test_values_batches[0].reshape(
    (1, test_values_batches.shape[1], test_values_batches.shape[2]))
predictiaaann = autoencoder.predict(test_values_batches_sample)
plt.subplot(1, 2, 2)
plt.plot(test_batches[0]["unix"], predictiaaann.reshape(len(test_batches[0])), label=f"aiii")
real_vals = np.delete(test_values_batches[1], np.s_[1:6], 1)
plt.plot(test_batches[0]["unix"], real_vals, label="real")
plt.legend()
plt.show()

for i,batch in enumerate(test_batches):
    batch_sample = test_values_batches[i].reshape((1,test_values_batches.shape[1],test_values_batches.shape[2]))
    predictiaaann = autoencoder.predict(batch_sample)
    plt.plot(batch["unix"], predictiaaann.reshape(len(test_batches[0])), label=f"aiii")
    real_vals = np.delete(test_values_batches[i], np.s_[1:6], 1)
    plt.plot(batch["unix"], real_vals, label="real")
    plt.legend()
    plt.show()