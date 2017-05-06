#coding:utf-8

from __future__ import absolute_import
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model

from util import load_data

def model_config():
    model = Sequential()

    model.add(Conv2D(4, (5, 5), padding='valid', input_shape=(80, 80, 1)))
    model.add(Activation('tanh'))

    model.add(Conv2D(8, (3, 3), padding='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), padding='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, kernel_initializer='random_uniform', bias_initializer='zeros', input_shape=(16 * 4 * 4,)))
    model.add(Activation('tanh'))

    # Softmax分类
    model.add(Dense(label_size, kernel_initializer='random_uniform', bias_initializer='zeros'))
    model.add(Activation('softmax'))

    return model

data,label = load_data("./data/train/")
test, true_labels = load_data("./data/test/")

print(data.shape[0], ' samples')

label_size = 6
label = np_utils.to_categorical(label, label_size)
test_labels = np_utils.to_categorical(true_labels, label_size)
# load_flag = True
load_flag = False

if not load_flag:
    model = model_config()
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical", metrics=['accuracy'])

    model.fit(data, label, batch_size=100, nb_epoch=10, shuffle=True, verbose=1, validation_split=0.2)
    model.save("model.h5")
else:
    model = load_model('model.h5')

test_results = model.evaluate(test, test_labels, verbose=1)

print("Test data")
print(test_results)