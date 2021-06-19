#!/usr/bin/env python
# coding= UTF-8
#
# Author: Fing
# Date  : 2017-12-03
#

import glob
import os
import librosa
import pandas as pd
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
from keras import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from keras.utils import np_utils


def features_extractor(file_name):
#     #mfcc
#     audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
#     mfccs_features = librosa.feature.melspectrogram(y=audio, sr=sample_rate,n_mels=128)
#     mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

    #mel
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
#     y_trimmed, index = librosa.effects.trim(audio, top_db=12, frame_length=2)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate,n_mels=128)
    mel_spec = np.mean(mel_spec.T,axis=0)
    
    return mel_spec



def main():
    audio_dataset_path=r'D:/raees assignments and study materials/8th Sem Final Year/Final year projectr/Final Year(AUDIO CLASS)/UrbanSound8K/audio/'
    metadata=pd.read_csv(r'D:/raees assignments and study materials/8th Sem Final Year/Final year projectr/Final Year(AUDIO CLASS)/UrbanSound8K/metadata/dataset_alter.csv')
    ### Now we iterate through every audio file and extract features 
    ### using Mel-Frequency Cepstral Coefficients
    extracted_features=[]
    for index_num,row in tqdm(metadata.iterrows()):
        file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
        final_class_labels=row["class"]
        data=features_extractor(file_name)
        extracted_features.append([data,final_class_labels])

    ### converting extracted_features to Pandas dataframe
    extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])

    ### Split the dataset into independent and dependent dataset
    X=np.array(extracted_features_df['feature'].tolist())
    y=np.array(extracted_features_df['class'].tolist())

    # #Performing oversampling for imbalancy

    oversample = SMOTE(random_state=0)
    X, y = oversample.fit_resample(X, y)

    

    ### Train Test Split
    # from sklearn.model_selection import train_test_split
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

    #stratified for balancing
    from sklearn.model_selection import StratifiedShuffleSplit 

    splitter=StratifiedShuffleSplit(n_splits=2,random_state=12)
    for train,test in splitter.split(X,y):     #this will splits the index
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
    #reshaping to shape required by CNN 1D
    rows=128
    cols=1
    X_train=X_train.reshape(X_train.shape[0],rows,cols,1)
    X_test=X_test.reshape(X_test.shape[0],rows,cols,1)
    X_train.shape,X_test.shape

    ### No of classes
    num_labels=13
    #model build
    
    model=Sequential()
    model.add(Conv1D(64,kernel_size=5,strides=1,padding="Same",activation="relu",input_shape=(128,1)))
    model.add(MaxPooling1D(padding="same"))

    model.add(Conv1D(128,kernel_size=5,strides=1,padding="same",activation="relu"))
    model.add(MaxPooling1D(padding="same"))
    model.add(Conv1D(256,kernel_size=5,strides=1,padding="same",activation="relu"))
    model.add(MaxPooling1D(padding="same"))

    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_labels,activation="softmax"))
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


    ### Training my model


    num_epochs = 12
    num_batch_size = 128

    checkpointer = ModelCheckpoint(filepath='saved_models/cnn1d.hdf5', 
                                verbose=1, save_best_only=True)
    start = datetime.now()
    ###ANN
    # model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)
    ###CNN
    model.fit(X_train,y_train,batch_size=50,epochs=30,validation_data=(X_test,y_test))
    duration = datetime.now() - start
    print("Training completed in time: ", duration)

    #train and test loss and scores respectively
    train_loss_score=model.evaluate(X_train,y_train)
    test_loss_score=model.evaluate(X_test,y_test)
    print(train_loss_score)
    print(test_loss_score)




    labels = ['Air Conditioner','Dog bark', 'Car Horn', 'Children Playing',
        'Drilling', 'Engine Idling', 'Jackhammer', 'Siren','Gunshot',
        'Street Music','glass break','scream']
    sel_labels = ['dogbark','gunshot','glassbreak','scream']
    print ("Showing Confusion Matrix")
    y_prob = model.predict(X_test, verbose=0)

    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(y_test, 1)
    cm = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(cm, labels, labels)
    plt.figure(figsize = (16,8))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 14}, fmt='g', linewidths=.5)

    path = "model"
    model_json = model.to_json()
    with open("modelfull"+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("modelfull"+".h5")


if __name__ == '__main__': main()
