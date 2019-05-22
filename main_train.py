#%%
import os
import json
import matplotlib.pyplot as plt 
from imutils.paths import list_images
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from generator.pathgenerator import DataGenerator
from augmentImage.aug_img import aug
from network.googlenet import GoogleNet
from keras.optimizers import Adam

import keras.backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
#%%
epochs = 5
batch = 64
input_shape = (224, 224, 3)
LR = 0.001
# get data paths
root_path = "D:\\python\\dataset\\plant-seedlings-classification"
totalpath = list(list_images(os.path.sep.join([root_path, 'train'])))
labels_str = [x.split('\\')[-2] for x in totalpath]
print(len(totalpath))
print(labels_str[:3])
#%%
#encode label
le = LabelEncoder()
labels = le.fit_transform(labels_str)
class_dict = {v:str(k) for (k,v) in zip(labels, labels_str)}
with open('.\\class_dict.json', 'w') as f:
    f.write(json.dumps(class_dict))
#%%
# split data
x_train, x_val, y_train, y_val = train_test_split(totalpath, labels,\
    stratify = labels, test_size= 0.2, random_state = 314234)
#%%
#make generator
def f_scale(x):
    return x / 127.5 -1

traingen = DataGenerator(img_list = x_train, label_list = y_train, scale_func=f_scale, \
        resize = True, aug = aug, batch_size=batch, dim=input_shape[:2], n_channels=input_shape[2],\
        n_classes=len(class_dict.keys()), shuffle=True)
valgen = DataGenerator(img_list = x_val, label_list = y_val, scale_func=f_scale, \
        resize = True, aug = None, batch_size=batch, dim=input_shape[:2], n_channels=input_shape[2],\
        n_classes=len(class_dict.keys()), shuffle=True)
#%%
model = GoogleNet.build(input_shape = (224, 224, 3), n_class = len(class_dict.keys()))
model.summary()
model.compile(optimizer = Adam(lr = LR, beta_1 = 0.9),\
    loss = {'aux1_softmax':'categorical_crossentropy',"aux2_softmax":'categorical_crossentropy',\
        "softmax_out":"categorical_crossentropy"}, loss_weights={'aux1_softmax':0.3, 'aux2_softmax':0.3,\
            'softmax_out':1.}, metrics = ['accuracy'])

lr_decay = ReduceLROnPlateau(monitor = 'val_softmax_out_loss', factor = 0.3, patience = 10, min_lr = 1e-8)
ckpt = ModelCheckpoint(filepath = "D:\\python\\models\\seed\\{epoch:03d}-{val_softmax_out_acc:.4f}-{val_softmax_out_loss:.4f}.hdf5",\
    save_best_only = True, monitor = 'val_softmax_out_acc')

train_step = int(len(x_train) / batch)
val_step = int(len(x_val) / batch)
H = model.fit_generator(traingen, steps_per_epoch= train_step, epochs = epochs,\
    verbose = 1, validation_data = valgen, validation_steps = val_step, callbacks = [lr_decay, ckpt],\
        workers = 1, use_multiprocessing = False)

#%%
from pandas import DataFrame
DataFrame(H.history).to_csv("./history_tmp.csv")
#%%