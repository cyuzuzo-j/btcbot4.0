
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
    def __init__(self, nodeshape, bottelneck,dropout,hp):
        super(Autoencoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.LSTM(3, input_shape=nodeshape, activation=hp.Choice("activationLSTM",["LeakyReLU", "relu","sigmoid","tanh"]), return_sequences=True),
            layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, sampling="linear")),
            layers.LSTM(hp.Int("units", min_value=1, max_value=40, step=1), activation=hp.Choice("activationLSTM",["LeakyReLU", "relu","sigmoid","tanh"])),
            layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, sampling="linear")),
            layers.Dense(hp.Int('bottel', min_value=1, max_value=20, step=1), activation=hp.Choice("activationDense",["LeakyReLU", "relu","sigmoid","tanh"]))])

        self.decoder = tf.keras.Sequential([
            layers.LSTM(nodeshape[1],input_shape=(hp.Int('bottel', min_value=1, max_value=20, step=1),1), activation=hp.Choice("activationLSTM",["LeakyReLU", "relu","sigmoid","tanh"]),return_sequences=True),
            layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, sampling="linear")),
            layers.Conv1DTranspose(nodeshape[1], (nodeshape[0]+1)-hp.Int('bottel', min_value=1, max_value=20, step=1), activation=hp.Choice("activationCONV",["LeakyReLU", "relu","sigmoid","tanh"])),
            layers.Dropout(hp.Float("dropout", min_value=0, max_value=1, sampling="linear")),
            layers.LSTM(6, activation=hp.Choice("activationOUT",["LeakyReLU", "relu","sigmoid","tanh"]), return_sequences=True),])
        self.compile(optimizer='adam', loss='mse')

    def call(self, x):
        print(x.shape)
        encoded = self.encoder(x)
        print(encoded.shape)
        decoded = self.decoder(encoded)
        print(decoded.shape)
        return decoded





for i in range(0, 1):
    tuner = keras_tuner.BayesianOptimization(
        lambda x: Autoencoder(np.array(train_values_batches[0]).shape, BOTTELNECK,0.2,x),
        objective='val_loss',
        max_trials=1)
    tuner.search(train_values_batches,train_values_batches, epochs=2, validation_data=(test_values_batches, test_values_batches))
    autoencoder = tuner.get_best_models()[0]

    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    history = autoencoder.fit(train_values_batches,train_values_batches,
                              epochs=1,
                              batch_size=80,
                              validation_data=(test_values_batches,test_values_batches),
                              shuffle=True)

    plt.subplot(1, 2, 1)
    plt.plot(history.history["val_loss"], label=f"Validation Loss{(i+1)/10}")
    plt.plot(history.history["loss"], label=f"train Loss{(i+1)/10}")
    plt.legend()
    test_values_batches_sample = test_values_batches[0].reshape((1,test_values_batches.shape[1],test_values_batches.shape[2]))
    predictiaaann = autoencoder.predict(test_values_batches_sample)

    plt.subplot(1, 2, 2)
    predictiaaann = np.delete(predictiaaann, np.s_[1:6], 2)
    plt.plot(test_batches[0]["unix"], predictiaaann.reshape(len(test_batches[0])), label=f"aiii{(i+1)/10}")
real_vals = np.delete(test_values_batches[0], np.s_[1:6], 1)
plt.plot(test_batches[0]["unix"], real_vals, label="real")
plt.legend()
plt.show()
for i,batch in enumerate(test_batches):

    batch_sample = test_values_batches[i].reshape((1,test_values_batches.shape[1],test_values_batches.shape[2]))
    predictiaaann = autoencoder.predict(batch_sample)
    predictiaaann = np.delete(predictiaaann, np.s_[1:6], 2)
    plt.plot(batch["unix"], predictiaaann.reshape(len(test_batches[0])), label=f"aiii")
    real_vals = np.delete(test_values_batches[i], np.s_[1:6], 1)
    plt.plot(batch["unix"], real_vals, label="real")
    plt.legend()
    plt.show()

"""
latent_space = autoencoder.encoder.predict([test_values_batches[0]])
for i in range(BOTTELNECK):
    latent_space_new = list(map(lambda space: list(space)[:i] + [list(space)[i] + 6] + list(space)[i+1:],latent_space))
    latent_space_new = tf.cast(latent_space_new, tf.float32)
    reconstruction = autoencoder.decoder.predict(latent_space_new)
    plt.subplot(3,BOTTELNECK,i+1)
    plt.plot(test_batches[0]["unix"], reconstruction.reshape(len(test_batches[0])), label=f"feauture {(i + 1)}")
    plt.plot(test_batches[0]["unix"], autoencoder.decoder.predict(latent_space).reshape(len(test_batches[0])), label="real")
    plt.legend()
plt.show()
"""