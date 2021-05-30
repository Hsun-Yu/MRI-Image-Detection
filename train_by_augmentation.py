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

checkpoint_weight_path = 'models/weights.best.basic_cnn99.hdf5'
weight_path = 'models/weights.basic_cnn99.hdf5'
model_name = 'models/model3.h5'
num_epochs = 500
num_batch_size = 1

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
        mri.append(im)
    data.append([mri, row["category"], row["number"], row["isReal"], row["from"]])
    
dataframe = pd.DataFrame(data, columns=["feature", "category", "number", "isReal", "from"])
print('Finished data extraction from ', len(dataframe), ' files')

real_data_frame = dataframe[dataframe["isReal"] == 1]
augmentation_data_frame = dataframe[dataframe["isReal"] == 0]
print('Number of real data:' , len(real_data_frame))
print('Number of augmentation data:' , len(augmentation_data_frame))

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

le = LabelEncoder()

x_train = np.array(augmentation_data_frame["feature"].tolist())
y_train = to_categorical(le.fit_transform(augmentation_data_frame["category"].tolist()))

x_test = np.array(real_data_frame["feature"].tolist())
y_test = to_categorical(le.fit_transform(real_data_frame["category"].tolist()))

# %%
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 

a, num_channels, num_rows, num_columns, b = x_train.shape

# x_train = x_train.reshape(x_train.shape[0],num_channels, num_rows, num_columns, 1)
# x_test = x_test.reshape(x_test.shape[0],num_channels, num_rows, num_columns, 1)

num_labels = y_train.shape[1]
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
model.summary()
model.save(model_name)
# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1, batch_size=2)
accuracy = 100 * score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)
# %%

from keras.callbacks import ModelCheckpoint 
from datetime import datetime 



checkpointer = ModelCheckpoint(filepath=checkpoint_weight_path, 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_split=0.2, callbacks=[checkpointer], verbose=1)

duration = datetime.now() - start
print("Training completed in time: ", duration)
model.save_weights(filepath=weight_path)

# %%
print("Finish Weight")
model1 = model
model1.load_weights(weight_path)
score1 = model1.evaluate(x_test, y_test, batch_size=1, verbose=1)
print("Testing Accuracy: ", score1[1])
for i in range(len(x_test)):
    print(model1.predict(x_test[i:i+1])[0])
# %%
print("Check Point Weight")
model2 = model
model2.load_weights(checkpoint_weight_path)
score2 = model2.evaluate(x_test, y_test, batch_size=1, verbose=1)
print("Testing Accuracy: ", score2[1])
for i in range(len(x_test)):
    print(model2.predict(x_test[i:i+1])[0])

# %%
