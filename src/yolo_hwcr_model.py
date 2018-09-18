from keras.models import Model
from keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, MaxPooling2D,  UpSampling2D, Concatenate

def create_model(input_shape):

    X_input = Input(input_shape, name = 'input_1')

    X = Conv2D(16, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_1')(X_input)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_1')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_1
    X = MaxPooling2D((2, 2), strides = (2, 2), name = 'max_pooling2d_1')(X)

    X = Conv2D(32, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_2')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_2')(X)
    X = LeakyReLU(alpha=0.1)(X) #leakyt_relu_2
    X = MaxPooling2D((2, 2), strides = (2, 2), name = 'max_pooling2d_2')(X)

    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_3')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_3')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_3
    X = MaxPooling2D((2, 2), strides = (2, 2),  name = 'max_pooling2d_3')(X)

    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_4')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_4')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_4
    X = MaxPooling2D((2, 2), strides = (2, 2),  name = 'max_pooling2d_4')(X)

    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_5')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_5')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_5
    D = X
    X = MaxPooling2D((2, 2), strides = (2, 2),  name = 'max_pooling2d_5')(X)

    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_6')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_6')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_6
    X = MaxPooling2D((2, 2), strides = (1, 1), padding = 'same', name = 'max_pooling2d_6')(X)

    X = Conv2D(1024, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_7')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_7')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_7

    X = Conv2D(256, (1, 1), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_8')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_8')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_8

    X = Conv2D(128, (1, 1), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_11')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_10')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_10

    X = UpSampling2D(size = (2, 2))(X) #up_sampling2d_1
    X = Concatenate()([X, D]) #concatenate_1

    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', use_bias = False, name = 'conv2d_12')(X)
    X = BatchNormalization(axis = -1, name = 'batch_normalization_11')(X)
    X = LeakyReLU(alpha=0.1)(X) #leaky_re_lu_11

    X = Conv2D(13, (1, 1), strides = (1, 1), padding = 'same', use_bias = True, name = 'conv2d_13')(X)

    model = Model(inputs = X_input, outputs = X, name = 'my_yolo_ocr_model')

    return model

