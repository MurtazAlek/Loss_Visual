import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import metrics
from keras import losses
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import cifar10 as c
import numpy as np
import os
import keras.utils

batch_size = 100
epochs = 2
num_classes = 10

x_train = c.load_cifar_10_data('cifar-10-batches-py')[0]
y_train = c.load_cifar_10_data('cifar-10-batches-py')[2]
x_test = c.load_cifar_10_data('cifar-10-batches-py')[3]
y_test = c.load_cifar_10_data('cifar-10-batches-py')[5]
label_names = c.load_cifar_10_data('cifar-10-batches-py')[6]


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
subtract_pixel_mean = True
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

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

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=1024).batch(batch_size)

def identity_block(X,f,filters, stage,block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1,F2,F3 = filters
    X_shortcut = X

    #First layer
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name = conv_name_base + '2a', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second layer
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1, 1), padding='same', name = conv_name_base + '2b',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third layer
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1, 1), padding='valid', name = conv_name_base + '2c', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    #Add shortcut value to F(X) + 'relu'
    X = keras.layers.add([X,X_shortcut])
    X = Activation('relu')(X)

    return X

def convolution_block(X, f, filters,stage, block, s):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    # First layer
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second layer
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third layer
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Add shortcut value to F(X) + 'relu'
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s),name=conv_name_base + '1', padding='valid', kernel_initializer='he_normal')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)
    X = keras.layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet(input_shape=(32,32,3), classes=10):
    # CONV2D -> BATCHNORM -> RELU ->
    # CONVBLOCK -> IDBLOCK * 2 ->
    # CONVBLOCK -> IDBLOCK * 3 ->
    # CONVBLOCK -> IDBLOCK * 5 ->
    # CONVBLOCK -> IDBLOCK * 2 ->
    # AVGPOOL -> TOPLAYER

    X_input = Input(shape=input_shape)
    X = ZeroPadding2D((2,2))(X_input)
    X = Conv2D(filters=64,kernel_size=(1,1), strides=(1,1), name = 'conv1', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)

    X = convolution_block(X, f=3,filters=[64,64,256],stage = 2, block='a',s=1)
    X = identity_block(X, f=3, filters=[64,64,256], stage=2, block='b')
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    X = convolution_block(X, f=3, filters=[128,128,512], stage = 3, block='a', s=2)
    X = identity_block(X, f=3, filters=[128, 128, 512], stage = 3, block='b')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage = 3, block='c')
    X = identity_block(X, f=3, filters=[128, 128, 512], stage = 3, block='d')

    X = convolution_block(X, f=3, filters=[256, 256, 1024],stage=4, block='a', s=2)
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    X = convolution_block(X, f=3, filters=[512, 512, 2048], stage=5,block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5,block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5,block='c')

    X=AveragePooling2D((2,2), name='avg_pool')(X)

    X=Flatten()(X)
    X=Dense(classes,activation='softmax',name='fc' + str(classes), kernel_initializer='he_normal')(X)
    model=Model(inputs=X_input, outputs=X, name='ResNet')

    return model

model=ResNet(input_shape=(32,32,3),classes=10)
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(lr=lr_schedule(0)),
#               metrics=['accuracy'])

def Graph(model_name):
    return tf.keras.utils.plot_model(model, "model_name", show_shapes=True)

with tf.device('/device:GPU:0'):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False, name="SGD")

    train_loss_list = []
    train_acc_list = []

    for epoch in range(epochs):

        train_loss = 0
        train_acc = 0
        loss_sum_epoch = []
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

            loss_sum_epoch.append(loss_value)

            train_acc_metric.update_state(y_batch_train, logits)

        avg_train_acc = train_acc_metric.result()

        avg_train_loss = np.sum(loss_sum_epoch) / len(train_dataset)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)

    loss_sum_test=[]
    for x_batch_test, y_batch_test in test_dataset:
        test_logits = model(x_batch_test, training=False)
        loss_value_test = loss_fn(y_batch_test, test_logits)
        loss_sum_test.append(loss_value_test)
        test_acc_metric.update_state(y_batch_test, test_logits)
    test_loss=np.sum(loss_sum_test)/len(test_dataset)
    test_acc = test_acc_metric.result()

# depth=50
# model_type = 'ResNet%d' % (depth)
# # Prepare model model saving directory.
# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# filepath = os.path.join(save_dir, model_name)
#
# # Prepare callbacks for model saving and for learning rate adjustment.
# checkpoint = ModelCheckpoint(filepath=filepath,
#                              monitor='val_accuracy',
#                              verbose=1,
#                              save_best_only=True)
#
# lr_scheduler = LearningRateScheduler(lr_schedule)
#
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
#                                cooldown=0,
#                                patience=5,
#                                min_lr=0.5e-6)
#
# callbacks = [checkpoint, lr_reducer, lr_scheduler]
#
# model.fit(x_train, y_train,
#               batch_size=batch_size,
#               epochs=epochs,
#               validation_data=(x_test, y_test),
#               shuffle=True,
#               callbacks=callbacks)
#
# scores = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])























