#%%
from keras.layers import Dense, Conv2D, Concatenate, Softmax, Input, Activation, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Dropout, AveragePooling2D, Flatten
from keras.models import Model
from keras.regularizers import l2

class GoogleNet:
    @staticmethod
    def inception_v1(x, conv1x1, conv3x3_reduce, conv3x3, conv5x5_reduce, conv5x5, pool_proj , \
        prefix, init = 'glorot_normal', reg = 0.0005):
    # conv1x1 : # filters of 1x1 convolution
    # conv3x3_reduce : # filters of 1x1 convolution followed by 3x3 conv
    # conv3x3 : # filters of 3x3 convolution 
    # conv5x5_reduce : # filters of 1x1 convolution followed by 5x5 conv
    # conv5x5 : # filters of 5x5 convolution
    # pool_proj : # filters of 1x1 convolution following max pool layer

        l_conv1x1 = Conv2D(filters = conv1x1, kernel_size = (1, 1), strides = (1, 1),\
            padding = 'same', kernel_initializer =init, kernel_regularizer = l2(reg), \
                name = '{}_conv1x1'.format(prefix))(x)
        l_conv1x1 = Activation('relu', name = '{}_conv1x1_act'.format(prefix))(l_conv1x1)

        l_conv3x3_reduce = Conv2D(filters = conv3x3_reduce, kernel_size = (1, 1), strides = (1, 1),\
            padding = 'same', kernel_initializer =init, kernel_regularizer = l2(reg),\
                name = '{}_conv3x3_reduce'.format(prefix))(x)
        l_conv3x3_reduce = Activation('relu', name = '{}_conv3x3_reduce_act'.format(prefix))(l_conv3x3_reduce)
        l_conv3x3 = Conv2D(filters = conv3x3, kernel_size = (3, 3), strides = (1, 1), \
        padding = 'same', kernel_initializer =init, kernel_regularizer = l2(reg),\
            name = '{}_conv3x3'.format(prefix))(l_conv3x3_reduce)
        l_conv3x3 = Activation('relu', name = '{}_conv3x3_act'.format(prefix))(l_conv3x3)

        l_conv5x5_reduce =  Conv2D(filters = conv5x5_reduce, kernel_size = (1, 1), strides = (1, 1),\
            padding = 'same', kernel_initializer =init, kernel_regularizer = l2(reg),\
                name = '{}_conv5x5_reduce'.format(prefix))(x)
        l_conv5x5_reduce = Activation('relu', name = '{}_conv5x5_reduce_act'.format(prefix))(l_conv5x5_reduce)
        l_conv5x5 = Conv2D(filters = conv5x5, kernel_size = (5, 5), strides = (1, 1), \
        padding = 'same', kernel_initializer =init, kernel_regularizer = l2(reg),\
            name = '{}_conv5x5'.format(prefix))(l_conv5x5_reduce)
        l_conv5x5 = Activation('relu', name = '{}_conv5x5_act'.format(prefix))(l_conv5x5)

        l_pool_proj = MaxPooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same',\
            name = '{}_pool_proj_pooling'.format(prefix))(x)
        l_pool_proj = Conv2D(filters = pool_proj, kernel_size = (1, 1), strides = (1, 1),\
            padding = 'same', kernel_initializer =init, kernel_regularizer = l2(reg),\
                name = '{}_pool_porj_conv'.format(prefix))(l_pool_proj)
        l_pool_proj = Activation('relu', name = '{}_pool_proj_act'.format(prefix))(l_pool_proj)

        out = Concatenate(name = '{}_concat'.format(prefix))([l_conv1x1, l_conv3x3, l_conv5x5, l_pool_proj])
        
        return out
    @staticmethod
    def auxiliary_loss(x, init, reg, prefix, n_class):
        x = AveragePooling2D(pool_size = (5, 5), strides = (3, 3), padding = 'same',\
            name = '{}_avgpool'.format(prefix))(x)
        x = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', \
            kernel_regularizer = l2(reg), kernel_initializer = init, name = '{}_conv1d'.format(prefix))(x)
        x = Activation('relu', name = '{}_act1'.format(prefix))(x)
        
        x = Flatten(name = '{}_flatten'.format(prefix))(x)
        x = Dense(1024, name = '{}_dense1'.format(prefix))(x)
        x = Activation('relu', name = '{}_act2'.format(prefix))(x)
        x = Dropout(rate = 0.7, name = '{}_dropout'.format(prefix))(x)
        x = Dense(n_class, name = '{}_dense2'.format(prefix))(x)
        x = Softmax(name = '{}_softmax'.format(prefix))(x)

        return x

        



    @staticmethod
    def build(input_shape, n_class, init = 'glorot_normal', reg = 0.0005, include_aux = True):
        _input = Input(shape = input_shape, name = 'input')
        x = ZeroPadding2D((3, 3), name = 'input_padding')(_input)

        x = Conv2D(filters = 64, kernel_size = (7,7), strides = (2, 2), padding = 'valid',\
            kernel_initializer = init, kernel_regularizer = l2(reg), name = 'conv7x7/2')(x)
        x = Activation('relu', name = 'conv7x7/2_relu')(x)
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'conv7x7/2_pool')(x)

        x = Conv2D(filters = 192, kernel_size = (3,3), strides = (1,1), padding = 'same',\
            kernel_initializer = init, kernel_regularizer = l2(reg), name = 'conv3x3/1')(x)
        x = Activation('relu', name = 'conv3x3/1_relu')(x)
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'same', name = 'conv3x3/1_pool')(x)
    
        x = GoogleNet.inception_v1(x, 64, 96, 128, 16, 32, 32, prefix = 'inception_3a')
        x = GoogleNet.inception_v1(x, 128, 128, 192, 32, 96, 64, prefix = 'inception_3b')
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'maxpool1', padding = 'same')(x)

        x = GoogleNet.inception_v1(x, 192, 96, 208, 16, 48, 64, prefix = 'inception_4a')
        x = GoogleNet.inception_v1(x, 160, 112, 224, 24, 64, 64, prefix = 'inception_4b')
        if include_aux:
            aux1 = GoogleNet.auxiliary_loss(x, init, reg, prefix = 'aux1', n_class = n_class)

        x = GoogleNet.inception_v1(x, 128, 128, 256, 24, 64, 64, prefix = 'inception_4c')
        x = GoogleNet.inception_v1(x, 112, 144, 288, 32, 64, 64, prefix = 'inception_4d')
        x = GoogleNet.inception_v1(x, 256, 160, 320, 32, 128, 128, prefix = 'inception_4e')
        if include_aux:
            aux2 = GoogleNet.auxiliary_loss(x, init, reg, prefix = 'aux2', n_class = n_class)
        x = MaxPooling2D(pool_size = (3, 3), strides = (2, 2), name = 'maxpool2', padding = 'same')(x)

        x = GoogleNet.inception_v1(x, 256, 160, 320, 32, 128, 128, prefix = 'inception_5a')
        x = GoogleNet.inception_v1(x, 384, 192, 384, 48, 128, 128, prefix = 'inception_5b')
        x = GlobalAveragePooling2D()(x)

        x = Dropout(rate = 0.4)(x)
        x = Dense(n_class)(x)
        x = Softmax(name = 'softmax_out')(x)

        if include_aux:
            model = Model(_input, [aux1, aux2, x])
        else:
            model = Model(_input, x)
        return model

 