#coding:utf-8
from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from util import load_data

def model_config(size):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='valid', input_shape=(size, size, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Activation('tanh'))

    # Softmax分类
    model.add(Dense(label_size, kernel_initializer='he_normal', bias_initializer='zeros'))
    model.add(Activation('softmax'))

    return model

if __name__ == '__main__':
    size = 100

    test, true_labels = load_data("./data/test/",  size, False)
    label_size = 6
    # for categorical_crossentropy loss, labels should be one-hot encoding vector

    label_test = np_utils.to_categorical(true_labels, label_size)
    load_flag = True
    # load_flag = False

    if not load_flag:
        data, label = load_data("./data/train/", size, True)
        print(data.shape[0], ' samples')

        label_train = np_utils.to_categorical(label, label_size)

        model = model_config(size)
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical", metrics=['accuracy'])

        model.fit(data, label_train, batch_size=100, nb_epoch=10, shuffle=True, verbose=1, validation_split=0.1)
        model.save("model.h5")
    else:
        model = load_model('model.h5')

    test_results = model.evaluate(test, label_test, verbose=1)

    print("Test data")
    print(test_results)