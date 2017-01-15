from keras import backend as K
from keras.layers import Input, Dense, merge, Flatten, Dropout, Lambda, normalization, Merge, Reshape, noise
from keras.models import Model
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

assert K.backend() == 'theano'
assert K.image_dim_ordering() == 'th'

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)

def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]

def attention_control(args):
    x,dense_2 = args
    find_att = K.reshape(x,(15,15,10))
    find_att = K.transpose(find_att[:,:,:])
    find_att = K.mean(find_att,axis=0)
    find_att = find_att/K.sum(find_att,axis=0)
    find_att = K.repeat_elements(find_att,32,axis=0)
    find_att = K.reshape(find_att,(1,32,15,15))
    return find_att

def no_attention_control(args):
    x,dense_2 = args
    find_att = K.ones(shape=(1,32,15,15))
    return find_att

def change_shape1(x):
    x = K.reshape(K.transpose(x),(15*15,32))
    return x

def att_shape(input_shape):
    return (input_shape[0][0],32,15,15)

def att_shape2(input_shape):
    return input_shape[0][0:4]

def minst_attention(inc_noise=False, attention=True):
    #make layers
    inputs = Input(shape=(1,image_size,image_size),name='input')

    conv_1a = Convolution2D(32, 3, 3,activation='relu',name='conv_1')
    maxp_1a = MaxPooling2D((3, 3), strides=(2,2),name='convmax_1')
    norm_1a = crosschannelnormalization(name="convpool_1")
    zero_1a = ZeroPadding2D((2,2),name='convzero_1')

    conv_2a = Convolution2D(32,3,3,activation='relu',name='conv_2')
    maxp_2a = MaxPooling2D((3, 3), strides=(2,2),name='convmax_2')
    norm_2a = crosschannelnormalization(name="convpool_2")
    zero_2a = ZeroPadding2D((2,2),name='convzero_2')

    dense_1a = Lambda(global_average_pooling,output_shape=global_average_pooling_shape,name='dense_1')
    dense_2a = Dense(10, activation = 'softmax', init='uniform',name='dense_2')

    #make actual model
    if inc_noise:
        inputs_noise = noise.GaussianNoise(2.5)(inputs)
        input_pad = ZeroPadding2D((1,1),input_shape=(1,image_size,image_size),name='input_pad')(inputs_noise)
    else:
        input_pad = ZeroPadding2D((1,1),input_shape=(1,image_size,image_size),name='input_pad')(inputs)

    conv_1 = conv_1a(input_pad)
    conv_1 = maxp_1a(conv_1)
    conv_1 = norm_1a(conv_1)
    conv_1 = zero_1a(conv_1)

    conv_2_x = conv_2a(conv_1)
    conv_2 = maxp_2a(conv_2_x)
    conv_2 = norm_2a(conv_2)
    conv_2 = zero_2a(conv_2)
    conv_2 = Dropout(0.5)(conv_2)

    dense_1 = dense_1a(conv_2)
    dense_2 = dense_2a(dense_1)

    conv_shape1 = Lambda(change_shape1,output_shape=(32,),name='chg_shape')(conv_2_x)
    find_att = dense_2a(conv_shape1)

    if attention:
        find_att = Lambda(attention_control,output_shape=att_shape,name='att_con')([find_att,dense_2])
    else:
        find_att = Lambda(no_attention_control,output_shape=att_shape,name='att_con')([find_att,dense_2])

    zero_3a = ZeroPadding2D((1,1),name='convzero_3')(find_att)
    apply_attention  = Merge(mode='mul',name='attend')([zero_3a,conv_1])

    conv_3 = conv_2a(apply_attention)
    conv_3 = maxp_2a(conv_3)
    conv_3 = norm_2a(conv_3)
    conv_3 = zero_2a(conv_3)

    dense_3 = dense_1a(conv_3)
    dense_4 = dense_2a(dense_3)

    model = Model(input=inputs,output=dense_4)

    return model

import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
image_size = np.shape(X_train[0])[1]
X_train.shape = (len(X_train),1,image_size,image_size)
X_test.shape = (len(X_test),1,image_size,image_size)

from keras.utils.np_utils import to_categorical
y_trainCAT = to_categorical(y_train)
y_testCAT = to_categorical(y_test)

model = minst_attention(inc_noise=False)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

model_history = model.fit(X_train, y_trainCAT,batch_size=1,validation_data=(X_test,y_testCAT),nb_epoch=1)

#import pickle
#pickle.dump( model_history.history['val_acc'], open( "./Performance/minst_att2_nonoise_performance.p", "wb" ) )
#score = model_history.history['val_acc'][-1]
#all_score = model_history.history['val_acc']
