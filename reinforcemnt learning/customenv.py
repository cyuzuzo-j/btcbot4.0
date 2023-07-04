import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.model_selection import train_test_split

NBATCHES = 421


def load_dataset(year=None):
    if year is None:
        dataframe = pd.read_csv("dataset/BTC-Hourly.csv")
    else:
        dataframe = pd.read_csv(f"dataset/BTC-{year}min.csv")
    dataframe = dataframe.sort_values(by=['unix'])
    batches = np.split(dataframe, NBATCHES)
    return batches


def normalize_batches(batches):
    normalized_batches = []
    normalized_values_batches = []
    for batch in batches:
        values_dataset = np.log10(batch.iloc[:, 3:] + 1)
        time_dataset = batch.iloc[:, :3]  # and symbol
        values_dataset = (values_dataset - values_dataset.min()) / (values_dataset.max() - values_dataset.min())
        normalized_batches.append(pd.concat([time_dataset, values_dataset], axis=1))
        normalized_values_batches.append(values_dataset)

    return normalized_batches, normalized_values_batches


def getbatches():
    train_batches = load_dataset()
    train_batches, train_values_batches = normalize_batches(train_batches)
    train_values_batches = np.array([data.values for data in train_values_batches]) #todo:zeer inoptimaal
    return train_values_batches

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.testareas = getbatches()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        ...

    def close(self):
        ...
'''