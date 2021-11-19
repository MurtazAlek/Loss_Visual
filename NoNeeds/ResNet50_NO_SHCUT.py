import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import cifar10 as c
import numpy as np
import os
import keras.utils

batch_size = 100
epochs = 20
num_classes = 10

subtract_pixel_mean = True


x_train = c.load_cifar_10_data('cifar-10-batches-py')[0]
y_train = c.load_cifar_10_data('cifar-10-batches-py')[2]
x_test = c.load_cifar_10_data('cifar-10-batches-py')[3]
y_test = c.load_cifar_10_data('cifar-10-batches-py')[5]
label_names = c.load_cifar_10_data('cifar-10-batches-py')[6]


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1

    return lr


def block(X,f,filters, s):

    F = filters

    # First layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(s, s), padding='valid', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Third layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Fourth layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    #Fifth layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    #Sixth layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    #Seventh layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    #Eaght layer
    X = Conv2D(filters=F, kernel_size=(f, f), strides=(1, 1), padding='valid', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    return X


def ResNet_NO_SHCUT(input_shape=(32,32,3), classes=10):

    # CONV2D -> BATCHNORM -> RELU ->
    # block * 6 ->
    # AVGPOOL -> TOPLAYER

    X_input = Input(shape=input_shape)
    X = ZeroPadding2D((2,2))(X_input)
    X = Conv2D(filters=16,kernel_size=(1,1), strides=(1,1), name = 'conv1', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)

    X = block(X, 1, 16, s=1)
    X = block(X, 1, 32, s=2)
    X = block(X, 1, 64, s=2)
    X = block(X, 1, 128, s=2)
    X = block(X, 1, 512, s=2)
    X = block(X, 1, 1024,s=2)


    X=AveragePooling2D((2,2), name='avg_pool')(X)

    X=Flatten()(X)
    X=Dense(classes,activation='softmax',name='fc' + str(classes), kernel_initializer='he_normal')(X)
    model=Model(inputs=X_input, outputs=X, name='ResNet')

    return model

model=ResNet_NO_SHCUT(input_shape=(32,32,3),classes=10)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

tf.keras.utils.plot_model(model, "model_ResNet50_NO_SHCUT.png", show_shapes=True)

depth=50
model_type = 'ResNet_NO_SHCUT%d' % (depth)
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)