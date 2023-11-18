from keras import Model
from keras.src.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.models import Sequential
from datetime import datetime
import random
import os
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping
from PIL import Image
from tensorflow.keras import layers

from ai.Utils import Utils
from ai.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from ai.SumTree import SumTree
import pickle
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split

TOTAL_STACKED_IMAGE = 3
BATCH_SIZE = 6

HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete(['0.0001', '0.01']))
# HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete(['1']))
# HP_CONV1 = hp.HParam('conv_1', hp.Discrete([16, 128]))
HP_CONV1 = hp.HParam('conv_1', hp.Discrete([32]))
HP_CONV1_KERNEL = hp.HParam('conv_1_kernel', hp.Discrete([11]))
HP_CONV1_STRIDES = hp.HParam('conv_1_strides', hp.Discrete([4]))
HP_CONV2 = hp.HParam('conv_2', hp.Discrete([32]))
HP_CONV2_KERNEL = hp.HParam('conv_2_kernel', hp.Discrete([11]))
HP_CONV2_STRIDES = hp.HParam('conv_2_strides', hp.Discrete([4]))
HP_DENSE1 = hp.HParam('dense_1', hp.Discrete([128]))
HP_DENSE2 = hp.HParam('dense_2', hp.Discrete([128]))
HP_DENSE3 = hp.HParam('dense_3', hp.Discrete([128]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([2, 4]))
HP_LOSS = hp.HParam('loss', hp.Discrete(['huber_loss']))
HP_OPT = hp.HParam('opt', hp.Discrete(['Adam']))

METRIC_ACCURACY = 'accuracy'
METRIC_LEARNING_TIME = 'learning_time'
METRIC_LOSS = 'loss'


with open('training.pickle', 'rb') as f:
    memory_with_priority = pickle.load(f)

def generate_data():
    return train_test_split(memory_with_priority.sample(100), test_size=0.25, shuffle=True)

X, y = generate_data()

session_num = 0
log_dir = "logs-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/hparam_tuning"

def train(hparams, indexed_experiences):
    m = []
    m.append(Conv2D(hparams[HP_CONV1], (hparams[HP_CONV1_KERNEL], hparams[HP_CONV1_KERNEL]), strides=(hparams[HP_CONV1_STRIDES], hparams[HP_CONV1_STRIDES]), activation='relu', kernel_initializer='zeros', input_shape=(TOTAL_STACKED_IMAGE, 125, 175, 1)))
    m.append(Conv2D(hparams[HP_CONV2], (hparams[HP_CONV2_KERNEL], hparams[HP_CONV2_KERNEL]), strides=(hparams[HP_CONV2_STRIDES], hparams[HP_CONV2_STRIDES]), activation='relu', kernel_initializer='zeros'))
    # m.append(Conv2D(96, (11, 11), strides=(3, 3), activation='relu', kernel_initializer='zeros'))
    # m.append(Conv2D(64, (3, 3), strides=(1, 1), activation='relu', kernel_initializer='zeros'))
    m.append(Flatten())
    m.append(Dense(hparams[HP_DENSE1], activation='relu', kernel_initializer='zeros'))
    m.append(Dense(hparams[HP_DENSE2], activation='relu', kernel_initializer='zeros'))
    m.append(Dense(hparams[HP_DENSE3], activation='relu', kernel_initializer='zeros'))
    m.append(Dense(3, activation='linear', kernel_initializer='zeros'))
    model = Sequential(m)
    # display(model.summary())
    if hparams[HP_OPT] == 'Adam':
        opt = Adam(learning_rate=float(hparams[HP_LEARNING_RATE]))
    elif hparams[HP_OPT] == 'RMSprop':
        opt = RMSprop(learning_rate=float(hparams[HP_LEARNING_RATE]))
    elif hparams[HP_OPT] == 'SGD':
        opt = SGD(learning_rate=float(hparams[HP_LEARNING_RATE]))

    model.compile(loss=hparams[HP_LOSS], optimizer=opt, metrics=['accuracy'])

    states, q_values, errors = _calculate(model, indexed_experiences)

    print('Learning started')
    s = time.time()
    model.fit(states, q_values, batch_size=hparams[HP_BATCH_SIZE], epochs=50, verbose=1, validation_split=0.2, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1),  # log metrics
        hp.KerasCallback(log_dir, hparams),  # log hparams,
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ])
    learning_time = time.time() - s
    print(f'Learning time: {learning_time}')

    return model, learning_time

def test(model, y):
    experiences = [o[1] for o in y]

    states, actions, rewards, _, _ = map(np.array, zip(*experiences))

    prediction = model.predict(states, verbose=0)

    correct_prediction = 0
    for i in range(len(states)):
        predicted_action = np.argmax(prediction[i])
        if rewards[i] > 0 and actions[i] == predicted_action:
            correct_prediction += 1
    return (correct_prediction / len(states)) * 100

def _calculate(model, indexed_experiences):
    gamma = 0.95

    experiences = [o[1] for o in indexed_experiences]

    states, actions, rewards, next_states, dones = map(np.array, zip(*experiences))

    states = preprocess_image(states)
    next_states = preprocess_image(next_states)

    q_values = model.predict(states, verbose=0)
    next_q_values = model.predict(next_states, verbose=0)
    errors = np.zeros(len(experiences))

    for i in range(len(experiences)):
        old_value = q_values[i][actions[i]]
        if dones[i]:
            target = rewards[i]
        else:
            target = rewards[i] + gamma * np.max(next_q_values[i])
        q_values[i][actions[i]] = target

        errors[i] = abs(old_value - target)

    return states, q_values, errors

def preprocess_image(states):
    new_states = []
    for state in states:
        new_images = []
        for image in state:
            new_image = np.array(Image.fromarray(np.reshape(image, (125, 175)), mode='L').point(lambda pixel: 0 if pixel < 128 else 1, '1')).astype(np.uint8)
            new_image = np.reshape(new_image, (125, 175, 1))
            new_images.append(new_image)
        new_states.append(new_images)
    return np.array(new_states)



with tf.summary.create_file_writer(log_dir).as_default():
    hp.hparams_config(
        hparams=[HP_LEARNING_RATE, HP_CONV1, HP_CONV1_KERNEL, HP_CONV1_STRIDES, HP_CONV2, HP_CONV2_KERNEL, HP_CONV2_STRIDES, HP_DENSE1, HP_DENSE2, HP_DENSE3, HP_BATCH_SIZE, HP_LOSS, HP_OPT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy'), hp.Metric(METRIC_LEARNING_TIME, display_name='learning_time')],
    )

for learning_rate in HP_LEARNING_RATE.domain.values:
    for conv1 in HP_CONV1.domain.values:
        for conv1_kernel in HP_CONV1_KERNEL.domain.values:
            for conv1_strides in HP_CONV1_STRIDES.domain.values:
                for conv2 in HP_CONV2.domain.values:
                    for conv2_kernel in HP_CONV2_KERNEL.domain.values:
                        for conv2_strides in HP_CONV2_STRIDES.domain.values:
                            for dense1 in HP_DENSE1.domain.values:
                                for dense2 in HP_DENSE2.domain.values:
                                    for dense3 in HP_DENSE3.domain.values:
                                        for batch_size in HP_BATCH_SIZE.domain.values:
                                            for loss in HP_LOSS.domain.values:
                                                for opt in HP_OPT.domain.values:
                                                    hparams = {
                                                        HP_LEARNING_RATE: learning_rate,
                                                        HP_CONV1: conv1,
                                                        HP_CONV1_KERNEL: conv1_kernel,
                                                        HP_CONV1_STRIDES: conv1_strides,
                                                        HP_CONV2: conv2,
                                                        HP_CONV2_KERNEL: conv2_kernel,
                                                        HP_CONV2_STRIDES: conv2_strides,
                                                        HP_DENSE1: dense1,
                                                        HP_DENSE2: dense2,
                                                        HP_DENSE3: dense3,
                                                        HP_BATCH_SIZE: batch_size,
                                                        HP_LOSS: loss,
                                                        HP_OPT: opt,
                                                    }
                                                    run_name = "run-%d" % session_num
                                                    print('--- Starting trial: %s' % run_name)
                                                    print({h.name: hparams[h] for h in hparams})
                                                    run_dir = log_dir + '/' + run_name

                                                    with tf.summary.create_file_writer(run_dir).as_default():
                                                        hp.hparams(hparams)
                                                        model, learning_time = train(hparams, X)
                                                        accuracy = test(model, y)
                                                        print(f'Accuracy: {accuracy}')
                                                        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
                                                        tf.summary.scalar(METRIC_LEARNING_TIME, learning_time, step=1)

                                                    session_num += 1

#%%

#%%
