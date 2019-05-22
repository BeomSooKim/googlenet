#%%
import keras as K
from network.googlenet import GoogleNet
from keras.utils import plot_model
#%%
model = GoogleNet.build((224, 224, 3), 1000)

print(model.summary())
plot_model(model, to_file = './model.png')