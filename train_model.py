import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

data = pd.read_csv(r'data/train.csv')
data = data.append(pd.read_csv(r'data/Dig-MNIST.csv'))
x = data.drop(['label'], axis=1)
y = np.asarray(data.label)

x = np.asarray(x).astype(np.float32)
x = x.reshape(x.shape[0], 1, 28, 28)
x = x/255

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1,
                                                  shuffle=True, random_state=10)
# CREATE MORE IMAGES VIA DATA AUGMENTATION
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        data_format='channels_first')

# CREATE VALIDATION GENERATOR
val_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,  
        zoom_range = 0.10,  
        width_shift_range=0.1, 
        height_shift_range=0.1,
        data_format='channels_first')

# DECREASE LEARNING RATE EACH EPOCH
annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# EARLY STOPPING
stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

with tf.device('/device:GPU:0'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation='relu', data_format='channels_first'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=50), epochs=100, 
                        steps_per_epoch=x_train.shape[0]//50, 
                        validation_data=val_gen.flow(x_val, y_val, batch_size=50), 
                        validation_steps=x_val.shape[0]//50, callbacks=[annealer, stop],
                        verbose=2)

    loss, acc = model.evaluate(x_val, y_val, batch_size=128) 
  
del(x,y,x_train,x_val,y_train,y_val)
model.save(f'model_{loss}_{acc}.h5')




















