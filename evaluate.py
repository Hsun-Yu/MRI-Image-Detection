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
        print(im.shape)
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
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
print(X[0].shape)
# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y)) 

# split the dataset 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

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
model.load_weights('models/weights.best.basic_cnn1.hdf5')
score = model.evaluate(X, yy, batch_size=1, verbose=1)
accuracy = 100 * score[1]

print("Accuracy: %.4f%%" % accuracy)
