import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, ZeroPadding2D
from keras.layers import AveragePooling2D, Input, Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import metrics
from keras.models import Model
import tensorflow as tf
import cifar10 as c
import numpy as np
import os
import keras.utils
import keras.losses

num_classes = 10
batch_size = 100
epochs = 2

x_train = c.load_cifar_10_data('cifar-10-batches-py')[0]
y_train = c.load_cifar_10_data('cifar-10-batches-py')[2]
x_test = c.load_cifar_10_data('cifar-10-batches-py')[3]
y_test = c.load_cifar_10_data('cifar-10-batches-py')[5]
label_names = c.load_cifar_10_data('cifar-10-batches-py')[6]


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# x_train = np.reshape(x_train, (-1, len(x_train)))
# x_test = np.reshape(x_test, (-1, len(x_test)))

# y_train = tf.keras.utils.to_categorical(y_train, num_classes)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes)

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
    # AVGPOOL -> TOPLAYER

    X_input = Input(shape=input_shape)
    X = ZeroPadding2D((2,2))(X_input)
    X = Conv2D(filters=16,kernel_size=(1,1), strides=(1,1), name = 'conv1', kernel_initializer='he_normal')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)

    X = convolution_block(X, f=3,filters=[16,16,32],stage = 2, block='a',s=1)
    X = identity_block(X, f=3, filters=[16,16,32], stage=2, block='b')
    X = identity_block(X, f=3, filters=[16, 16, 32], stage=2, block='c')

    X=AveragePooling2D((2,2), name='avg_pool')(X)

    X=Flatten()(X)
    X=Dense(classes,activation='softmax',name='fc' + str(classes), kernel_initializer='he_normal')(X)
    model=Model(inputs=X_input, outputs=X, name='ResNet')

    return model

model=ResNet(input_shape=(32,32,3),classes=10)

def Graph(model_name):
    return tf.keras.utils.plot_model(model, "model_name", show_shapes=True)

with tf.device('/device:GPU:0'):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name="SGD")

    train_loss_list = []
    train_acc_list = []

    for epoch in range(epochs):

        train_loss = 0
        train_acc = 0
        loss_sum_epoch=[]
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            opt.apply_gradients(zip(grads, model.trainable_weights))

            loss_sum_epoch.append(loss_value)

            train_acc_metric.update_state(y_batch_train, logits)

        train_acc = train_acc_metric.result()

        avg_train_loss = np.sum(loss_sum_epoch) / len(train_dataset)

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(train_acc)

    loss_sum_test=[]
    for x_batch_test, y_batch_test in test_dataset:
        test_logits = model(x_batch_test, training=False)
        loss_value_test = loss_fn(y_batch_test, test_logits)
        loss_sum_test.append(loss_value_test)
        test_acc_metric.update_state(y_batch_test, test_logits)
    test_loss=np.sum(loss_sum_test)/len(test_dataset)
    test_acc = test_acc_metric.result()




print(model.trainable_variables)


