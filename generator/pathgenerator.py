import keras as K
import numpy as np 
import imutils 
import cv2

class DataGenerator(K.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, img_list, label_list, scale_func, \
        resize, aug = None, batch_size=32, dim=(64,64), n_channels=3,\
        n_classes=17, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.image_list = img_list
        self.resize = resize
        self.label_list = label_list
        self.n_channels = n_channels
        self.aug = aug
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.scale = scale_func
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_img = [self.image_list[k] for k in indexes]
        batch_label = [self.label_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(batch_img, batch_label)
        
        X = self.scale(X)
        return X,{'aux1_softmax':y,'aux2_softmax':y,'softmax_out':y}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def resize_img(self, image, inter = cv2.INTER_AREA):
        #pdb.set_trace()
        resize = self.dim[0]
        (h, w) = image.shape[:2]
        dW = 0
        dH = 0

        if w < h:
            image = imutils.resize(image, width=resize, inter=inter)
            dH = int((image.shape[0] - resize) / 2.0)
        else:
            image = imutils.resize(image, height=resize, inter=inter)
            dW = int((image.shape[1] - resize) / 2.0)

        (h, w) = image.shape[:2]
        image = image[dH:h - dH, dW:w - dW]

        return cv2.resize(image, (resize, resize), interpolation=inter)
    
    def get_input(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def __data_generation(self, batch_img, batch_label):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype = int)

        # Generate data
        for i, (path, label) in enumerate(zip(batch_img, batch_label)):
            # Store sample
            img= self.get_input(path)
            label = K.utils.to_categorical(label, num_classes = self.n_classes)
            if self.resize:
                img = cv2.resize(img, self.dim)
            
            X[i,] = img
            y[i] = label

        if self.aug is not None:
            X = self.aug.augment_images(X)

        return X, y