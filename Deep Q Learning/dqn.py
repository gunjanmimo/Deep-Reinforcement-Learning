import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
import keras.callbacks
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque

REPLAY_MEMORY_SIZE = 50000


class DqnAgent:
    def __init__(self):
        # main model # get trained every step
        self.model = self.create_model()
        # target model # this is what we predict against every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(256, (3, 3),
                         input_shape=env.OBSERVATION_SPACE_VALUES))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Conv2D(256, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse",
                      optimizer=Adam(lr=0.001),
                      metrics=["accuracy"])
        return model
