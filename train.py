# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt


# %%
import pandas as pd
import os
import librosa

checkpoint_weight_path = 'weights.best.basic_cnn15.hdf5'
weight_path = 'models/weights.basic_cnn15.hdf5'
model_name = 'models/model3.h5'
logs_path = './logs/cnn1/'
num_epochs = 500
num_batch_size = 2

# Set the path to the full audio dataset 
fulldatasetpath = r'./images/'

metadata = pd.read_csv(r'./metadatas/matadata.csv')

features = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row["category"]), 'Case ' + str(row["number"]))
    mri = []
    for i in range(1, 29):
        im = image.load_img(file_name + r'/' + str(i) + r'.jpg', target_size = (320, 320), color_mode = 'grayscale')
        im = img_to_array(im)
        # print(im.shape)
        mri.append(im)
    
    class_label = row["category"]
    features.append([mri, class_label])
# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')


# %%
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# Convert features and corresponding classification labels into numpy arrays
# print(featuresdf.feature[0])
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
print(X[0])
# print(X[0].shape)
# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# %%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

a, num_channels, num_rows, num_columns, b = X.shape
print(X[0].shape)
x_train = x_train.reshape(x_train.shape[0],num_channels, num_rows, num_columns, 1)
x_test = x_test.reshape(x_test.shape[0],num_channels, num_rows, num_columns, 1)

num_labels = yy.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv3D(filters=32, kernel_size=2, input_shape=(num_channels, num_rows, num_columns, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=2))

model.add(Conv3D(filters=32, kernel_size=2, activation='relu'))
model.add(AveragePooling3D(pool_size=2))

model.add(Conv3D(filters=64, kernel_size=2, activation='relu'))
model.add(AveragePooling3D(pool_size=2))

model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# %%

# Display model architecture summary 
model.summary()
model.save(model_name)
# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1, batch_size=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)


# %%
# ===================Training=========================

from keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime 


checkpointer = ModelCheckpoint(filepath=checkpoint_weight_path, 
                               verbose=1, save_best_only=True)
tbCallBack = TensorBoard(log_dir=logs_path,
                 histogram_freq=0,
                 write_graph=True, 
                 write_grads=True, 
                 write_images=True,
                 embeddings_freq=0, 
                 embeddings_layer_names=None, 
                 embeddings_metadata=None)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.2, callbacks=[checkpointer, tbCallBack], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save_weights(filepath=weight_path)

# %%

# score = model.evaluate(x_train, y_train, verbose=0)
# print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, batch_size=num_batch_size, verbose=1)
print("Testing Accuracy: ", score[1])

# %%
print("Finish Weight")
model1 = model
model1.load_weights(weight_path)
score1 = model1.evaluate(x_test, y_test, batch_size=1, verbose=1)
print("Testing Accuracy: ", score1[1])
for i in range(len(x_test)):
    print(model1.predict(x_test[i:i+1]), y_test[i:i+1])
# %%
print("Check Point Weight")
model2 = model
model2.load_weights(checkpoint_weight_path)
score2 = model2.evaluate(x_test, y_test, batch_size=1, verbose=1)
print("Testing Accuracy: ", score2[1])
for i in range(len(x_test)):
    print(model2.predict(x_test[i:i+1]), y_test[i:i+1])
# %%
