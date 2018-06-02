from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Dense, Flatten, Convolution2D
from sklearn.utils import shuffle
import numpy as np
import cv2

DATASET_PATH = '../dataset'

def image_generator(samples, batch_size=32):
  '''
  Python generator to generate/feed multiple batches of training-dataset.
  :param batch_size: Batch size of training-samples to generate.
  :return: Batch of features and labels, forming a part of the training-samples.
  '''
  num_samples = len(samples)
  while 1:
    shuffle(samples)
    for offset in range(0, num_samples, batch_size):
      batch_samples = samples[offset:offset+batch_size]
      images = []
      angles = []
      for batch_sample in batch_samples:
        center_image_path = DATASET_PATH + '/IMG/' + batch_sample[0].split('/')[-1]
        center_image = cv2.cvtColor(cv2.imread(center_image_path), cv2.COLOR_BGR2RGB)
        center_angle = float(batch_sample[3])
        images.append(center_image)
        angles.append(center_angle)
      X_train = np.array(images)
      y_train = np.array(angles)
      yield shuffle(X_train, y_train)

def get_nvidia_end_to_end_model():
  '''
  Builds and returns the CNN model as discussed nVidia's end-to-end pipeline paper.
  '''
  # Set up cropping2D layer
  model = Sequential()
  # Cropping 60 pixel-rows from top and 20 pixel-rows from bottom.
  model.add(Cropping2D(cropping=((70,25),(0,0)), input_shape=(160,320,3)))  # Image dimension cropped down to 65x320.
  model.add(Lambda(lambda x : x / 127.5 - 1.))
  model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2,2)))
  model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2,2)))
  model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2,2)))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(Convolution2D(64, 3, 3, activation='relu'))
  model.add(Flatten())
  model.add(Dense(100))
  model.add(Dense(50))
  model.add(Dense(10))
  model.add(Dense(1))
  return model


