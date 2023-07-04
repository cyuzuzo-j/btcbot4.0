
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
import keras_tuner
NBATCHES = 421
BOTTELNECK=20

def load_dataset(year=None):
    if year is None:
        dataframe = pd.read_csv("reinforcemnt learning/dataset/BTC-Hourly.csv")
    else:
        dataframe = pd.read_csv(f"reinforcemnt learning/dataset/BTC-{year}min.csv")
    dataframe = dataframe.sort_values(by=['unix'])
    #dataframe.iloc[:,3:] = fftn(dataframe.iloc[:,3:])
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
    #voegt een aantal nullen toe aangezien in de ppo het neuraal network ook een hoop waardes krijgt die nul zijn
    new_batches = []
    new_batches_values = []

    for batch in batches:
        place= random.randint(0,70)
        new_batches.append(pd.concat([batch[:place],pd.concat([batch.iloc[place:, :3],batch.iloc[place:, 3:]*0],axis=1)],axis=0))
        new_batches_values.append(pd.concat([batch.iloc[:place, 3:],batch.iloc[place:, 3:]*0],axis=0))
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
train_batches, train_values_batches  = nullize_batch(train_batches)
test_batches, test_values_batches  = nullize_batch(test_batches)
train_values_batches = np.array([data.values for data in train_values_batches])
test_values_batches = np.array([data.values for data in test_values_batches])

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # basenoise positional encode
        self.query_lstm1 = tf.keras.layers.LSTM(6, input_shape=(6, 79), return_sequences=True)
        self.key_lstm1 = tf.keras.layers.LSTM(6, input_shape=(6, 79), return_sequences=True)
        self.value = tf.keras.layers.LSTM(3, input_shape=(6, 79), return_sequences=True)
        self.attention = tf.keras.layers.AdditiveAttention()

        #self.lstm1 = tf.keras.layers.LSTM(3,activation="LeakyReLU", input_shape=(6, 79), return_sequences=True)
        self.drop1 = layers.Dropout(0.2)
        self.lstm2 = tf.keras.layers.LSTM(20, activation="LeakyReLU", input_shape=(3, 79))
        self.drop1 = layers.Dropout(0.2)
        self.latens_space = layers.Dense(14,activation="LeakyReLU")
        self.reshape = layers.Reshape((14,1))
        self.lstm_out1 = tf.keras.layers.LSTM(6, input_shape=(14, 1),activation="LeakyReLU",return_sequences=True )
        self.drop1 = layers.Dropout(0.2)
        self.conv1 = layers.Conv1DTranspose(6, (79 + 1) -14,activation="LeakyReLU")
        self.lstm_out2 = layers.LSTM(6, activation="LeakyReLU", return_sequences=True)



    def call(self, input):
        print(input.shape)

        a = self.query_lstm1(input)
        b = self.key_lstm1(input)
        c = self.value(input)
        x = self.attention([a,b,c])

        print(x.shape)
        x = self.drop1(x)
        print(x.shape)
        x = self.lstm2(x)
        print(x.shape)
        x = self.drop1(x)
        print(x.shape)
        x = self.latens_space(x)
        print(x.shape)
        x = self.reshape(x)
        print(x.shape)
        x = self.lstm_out1(x)
        print(x.shape)
        x = self.drop1(x)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)

        x = self.lstm_out2(x)
        print(x.shape)


        return x


autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(train_values_batches, train_values_batches,
                          epochs=100,
                          batch_size=80,
                          validation_data=(test_values_batches, test_values_batches),
                          shuffle=True)
autoencoder.summary()

plt.subplot(1, 2, 1)
plt.plot(history.history["val_loss"], label=f"Validation Loss{(i + 1) / 10}")
plt.plot(history.history["loss"], label=f"train Loss{(i + 1) / 10}")
plt.legend()
test_values_batches_sample = test_values_batches[0].reshape(
    (1, test_values_batches.shape[1], test_values_batches.shape[2]))
predictiaaann = autoencoder.predict(test_values_batches_sample)

plt.subplot(1, 2, 2)
predictiaaann = np.delete(predictiaaann, np.s_[1:6], 2)
plt.plot(test_batches[0]["unix"], predictiaaann.reshape(len(test_batches[0])), label=f"aiii{(i + 1) / 10}")