# %%
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, AveragePooling3D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import os
import xml.etree.ElementTree as et
# %%

def get_vertices(label_dataframe: pd.DataFrame):
    return {
        "xmax": max(label_dataframe.xmax),
        "xmin": min(label_dataframe.xmin),
        "ymax": max(label_dataframe.ymax),
        "ymin": min(label_dataframe.ymin),
        "zmax": max(label_dataframe.z),
        "zmin": min(label_dataframe.z)
    }


# Set the path to the full audio dataset 
fulldatasetpath = r'./images/'

metadata = pd.read_csv(r'./metadatas/matadata.csv')

features = []

file_name = os.path.join(os.path.abspath(fulldatasetpath), "Af", 'Case ' + "1")
mri = []

posterior_label = []
left_label = []
right_label = []

for i in range(1, 29):
    im = image.load_img(file_name + r'/' + str(i) + r'.jpg', color_mode = 'grayscale')
    im = img_to_array(im)

    xtree = et.parse(file_name + "/outputs/"+ str(i) + ".xml")
    xroot = xtree.getroot()
    xoutput = xroot.iter('item')
    for node in xoutput:
        name = node.find('name').text.replace(" ", "")
        bndbox = node.find("bndbox")
        label = {"name": node.find('name').text.replace(" ", ""),
                 "z": i,
                 "xmin": int(bndbox.find("xmin").text),
                 "ymin": int(bndbox.find("ymin").text),
                 "xmax": int(bndbox.find("xmax").text), 
                 "ymax": int(bndbox.find("ymax").text)}
        if name == "posterior":
            posterior_label.append(label)
        elif name == "right":
            left_label.append(label)
        elif name == "left":
            right_label.append(label)

    mri.append(im)

post_dataframe = pd.DataFrame(posterior_label)
post_vertices = get_vertices(post_dataframe)

left_dataframe = pd.DataFrame(left_label)
left_vertices = get_vertices(left_dataframe)

right_dataframe = pd.DataFrame(right_label)
right_vertices = get_vertices(right_dataframe)

# collect post
mri_post = []
for m in mri[post_vertices["zmin"]: post_vertices["zmax"]]:
    mri_post.append(m[post_vertices["ymin"]:post_vertices["ymax"],
                        post_vertices["xmin"]: post_vertices["xmax"]])

# collect right
mri_right = []
for m in mri[right_vertices["zmin"]: right_vertices["zmax"]]:
    mri_right.append(m[right_vertices["ymin"]:right_vertices["ymax"],
                        right_vertices["xmin"]: right_vertices["xmax"]])

# collect post
mri_left = []
for m in mri[left_vertices["zmin"]: left_vertices["zmax"]]:
    mri_left.append(m[left_vertices["ymin"]:left_vertices["ymax"],
                        left_vertices["xmin"]: left_vertices["xmax"]])

# print(file_name)
# # Iterate through each sound file and extract the features 
# for index, row in metadata.iterrows():
#     file_name = os.path.join(os.path.abspath(fulldatasetpath), str(row["category"]), 'Case ' + str(row["number"]))
#     mri = []
#     for i in range(1, 29):
#         im = image.load_img(file_name + r'/' + str(i) + r'.jpg', target_size = (320, 320), color_mode = 'grayscale')
#         im = img_to_array(im)
#         # print(im.shape)
#         mri.append(im)
    
#     class_label = row["category"]
#     features.append([mri, class_label])
# # Convert into a Panda dataframe 
# featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

# print('Finished feature extraction from ', len(featuresdf), ' files')

# %%
