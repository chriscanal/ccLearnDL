import sys
import numpy as np
import matplotlib.pyplot as plt
import time
np.random.seed(1337)  # for reproducibility

#variables to help with deleting of model
this = sys.modules[__name__]
c = dir()
c.append('c')
c.append('i')
c.append('n')
c.append('layers')
c.append('neurons')

layers = 2 #Change to for loop
neurons = 100 #Change to for loop

for i in range(2):
    for n in dir():
        if n[0]!='_' and (not n in c):
            print "Deleting ", n
            delattr(this, n)

    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers import advanced_activations
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils




    # the data, shuffled and split between train and test sets
    if not 'X_train' in c:
        a = dir()
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        im_width  = X_train.shape[1]
        im_height = X_train.shape[2]

        # change type to float
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # vectorize the images
        X_train = X_train.reshape(num_train, im_width*im_height)
        X_test = X_test.reshape(num_test, im_width*im_height)

        # normalize the range
        print('maximum of X_train:',np.max(X_train[:]))
        X_train /= 255.0;
        X_test /= 255.0;
        print('maximum of X_train:',np.max(X_train[:]))

        b = dir()
        c.extend(list(set(b) - set(a)))



    # convert class vectors to binary class matrices (one hot representation)
    nb_classes = np.unique(y_train).size
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # construct the network
    model = Sequential()
    model.add(Dense(100, input_shape=(im_width*im_height,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    for j in range(layers):
        model.add(Dense(neurons))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    batch_size = 1024
    nb_epoch = 1
    for k in range(15):
        start = time.time()
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test), validation_split=0.2)
        end = time.time()
        print(history.history)
        score = model.evaluate(X_test, Y_test, verbose=0)

        print 'Epoch number: ', k
        print 'Test score:   ', score[0]
        print 'Test accuracy:', score[1]
        print 'Time elapsed: ',(end - start), "seconds"
