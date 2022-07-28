import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.io import wavfile
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils
from keras.layers import Layer, Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


!wget 'http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip'
!unzip mini_speech_commands.zip
!rm '/content/mini_speech_commands/README.md'
data_dir = '/content/mini_speech_commands/'

def load_data(data_dir):
    labels=os.listdir(data_dir)
    labels=["yes", "no", "up", "down", "left", "right", "stop", "go"]
    return labels

def preprocess_data(labels):
    #Mengambil sample data berdasarkan label
    all_wave = []
    all_label = []
    num_labels = len(labels)

    for label in labels:
        print(label)
        waves = [f for f in os.listdir(data_dir + label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(data_dir + label + '/' + wav, sr = 16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if(len(samples)== 8000) : 
                all_wave.append(samples)
                all_label.append(label)

    #mengubah labels ke integer
    encoder = LabelEncoder()
    y=encoder.fit_transform(all_label)
    classes= list(encoder.classes_)

    #mengubah integer ke one-hot vector 
    y=np_utils.to_categorical(y, num_classes=num_labels)

    #reshape dari 2d array ke 3d array untuk conv1d
    all_wave = np.array(all_wave).reshape(-1,8000,1)
    return all_wave,y

def prepare_dataset(all_wave,y):
    x_train, x_val, y_train, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    return x_train, x_val, y_train, y_val

def build_model1(inputs):
    K.clear_session()

    #First Conv1D layer
    conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)

    #Flatten layer
    conv = Flatten()(conv)

    #Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    #Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)

    outputs = Dense(num_labels, activation='softmax')(conv)

    model_conv1d = Model(inputs, outputs)
    model_conv1d.summary()
    return model_conv1d

def build_model2(inputs):
    K.clear_session()
    norm_layer = layers.Normalization()
    model_conv2d = models.Sequential([
        inputs,
        #Downsample input
        layers.Resizing(32, 32),
        # Normalization
        norm_layer,
        #First Conv2d Layer
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model_conv2d.summary()
    return model_conv2d

def plot_history(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    return plt.show()

def predict(audio):
    prob_text=model_conv1d.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob_text[0])
    return classes[index]

def main():
    #load data
    labels = load_data(data_dir)
    
    #preprocess data
    allwave,y = preprocess_data(labels)

    #generate train, validation set
    x_train, y_train, x_val, y_val = prepare_dataset(all_wave,y)

    #membuat model 1 - conv1d
    inputs = Input(shape=(8000,1))
    model_conv1d = build_model1(inputs)

    #membuat model 2 - conv2d
    inputs = Input(shape=(8000,1,3))
    model_conv2d = build_model2(inputs)

    #compile model
    model_conv1d.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model_conv2d.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #mencegah overfitting dengan EarlyStopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 

    #menyimpan model dengan parameter terbaik menggunakan ModelCheckpoint
    mc = ModelCheckpoint('best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    #train model 1 dan 2 
    his_conv1d = model_conv1d.fit(x_train, y_train,epochs=29, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))
    his_conv2d = model_conv2d.fit(x_train, y_train,epochs=29, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

    # plot accuracy/loss untuk training/validation set 
    plot_history(his_conv1d)
    plot_history(his_conv2d)

    #melakukan evaluasi dataset mengambil file audio random pada dataset untuk diprediksi
    index=random.randint(0,len(x_val)-1)
    samples=x_val[index].ravel()
    print("Audio:",classes[np.argmax(y_val[index])])
    ipd.Audio(samples, rate=8000)
    
    print("Text:",predict(samples))

if __name__ == "__main__":
    main()
