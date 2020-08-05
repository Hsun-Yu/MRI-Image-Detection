# %%
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa

# %%
fulldatasetpath = r'./images/'

metadata = pd.read_csv(r'./metadatas/matadata.csv')

data = []

# Iterate through each sound file and extract the features 
for index, row in metadata.iterrows():
    file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row["category"]), 'Case ' + str(row["number"]))
    mri = []
    for i in range(1, 29):
        im = image.load_img(file_name + r'/' + str(i) + r'.jpg', target_size = (320, 320), color_mode = 'grayscale')
        im = img_to_array(im)
        # print(im.shape)
        mri.append(im)
    data.append([mri, row["category"], row["number"], row["isReal"], row["from"]])
    
dataframe = pd.DataFrame(data, columns=["feature", "category", "number", "isReal", "from"])
print('Finished data extraction from ', len(dataframe), ' files')

real_data_frame = dataframe[dataframe["isReal"] == 1]
print('Number of real data:' , len(real_data_frame))

# %%
from sklearn.model_selection import train_test_split
real_train, real_test = train_test_split(real_data_frame, test_size=0.2, random_state = 42)

print("len of original training data:", len(real_train))
print("len of original testing data:", len(real_test))

# %%
training_data = pd.DataFrame(columns=["feature", "category", "number", "isReal", "from"])
for row in real_train.iloc():
    data = dataframe[(dataframe["category"] == row["category"]) & (dataframe["from"] == row["number"])]
    training_data = training_data.append(data, ignore_index=True)
testing_data = pd.DataFrame(columns=["feature", "category", "number", "isReal", "from"])
for row in real_test.iloc():
    data = dataframe[(dataframe["category"] == row["category"]) & (dataframe["from"] == row["number"])]
    testing_data = testing_data.append(data, ignore_index=True)

print("len of Training Data:", len(training_data), " len of Testing Data:", len(testing_data))

# %%
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()

x_train = np.array(training_data["feature"].tolist())
y_train = to_categorical(le.fit_transform(training_data["category"].tolist()))

x_test = np.array(testing_data["feature"].tolist())
y_test = to_categorical(le.fit_transform(testing_data["category"].tolist()))

# %%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

a, num_channels, num_rows, num_columns, b = x_train.shape

x_train = x_train.reshape(x_train.shape[0],num_channels, num_rows, num_columns, 1)
x_test = x_test.reshape(x_test.shape[0],num_channels, num_rows, num_columns, 1)

num_labels = y_train.shape[1]
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv3D(filters=32, kernel_size=4, input_shape=(num_channels, num_rows, num_columns, 1), activation='relu'))
model.add(MaxPooling3D(pool_size=2))

model.add(Conv3D(filters=64, kernel_size=3, activation='relu'))
model.add(AveragePooling3D(pool_size=2))

model.add(Conv3D(filters=64, kernel_size=2, activation='relu'))
model.add(AveragePooling3D(pool_size=2))

model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()
# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1, batch_size=1)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)
# %%

from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 5000
num_batch_size = 1

checkpointer = ModelCheckpoint(filepath='models/weights.best.basic_cnn8.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save_weights(filepath='models/weights.best.basic_cnn8.hdf5')

# %%
score = model.evaluate(x_test, y_test, batch_size=1, verbose=1)
print("Testing Accuracy: ", score[1])

# %%
for i in range(len(x_test)):
    print(model.predict(x_test[i:i+1])[0])

# %%
