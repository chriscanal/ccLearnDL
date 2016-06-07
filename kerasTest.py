import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import csv


#variables to help with deleting of model
this = sys.modules[__name__]
testData = []
c = dir()
c.append('c')
c.append('i')
c.append('n')
c.append('layers')
c.append('neurons')

layers = 2 #Change to for loop
neurons = 100 #Change to for loop

for i in range(1):
    for n in dir():
        if n[0]!='_' and (not n in c):
            print "Deleting ", n
            delattr(this, n)
    #np.random.seed(1337)  # for reproducibility, must be reset with each generation of network
                          # network or the results will differ slightly
    from keras.datasets import mnist
    from keras.models import Model, Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers import advanced_activations
    from keras.layers import Convolution2D, MaxPooling2D, Flatten
    from keras.layers import Input, LSTM, Embedding, merge
    from keras.optimizers import SGD, Adam, RMSprop
    from keras.utils import np_utils




    # the data, shuffled and split between train and test sets
    if not 'X_train' in c:
        a = dir()
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train[:600][:][:]
        y_train = y_train[:600][:][:]
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        im_width  = X_train.shape[1]
        im_height = X_train.shape[2]
        print 'num_train: ', num_train
        print 'num_test: ', num_test
        print 'X_train Shape: ', X_train.shape



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

    # construct the network
    model = Sequential()
    model.add(Dense(700, input_shape=(im_width*im_height,)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.05))
    model.add(Dense(700))
    model.add(Activation('relu'))
    # model.add(Dropout(0.05))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    batch_size = 100
    nb_epoch = 1
    testData.append(['Epoch number'])
    testData.append(['Loss'])
    testData.append(['Test accuracy'])
    testData.append(['Epoch time'])
    print testData
    csvName = "testResults"+str(i)+".csv"
    for k in range(15):
        start = time.time()
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test), validation_split=0.2)
        end = time.time()
        print(history.history)
        score = model.evaluate(X_test, Y_test, verbose=0)

        print 'Epoch number: ', k
        #print 'Loss:   ', score[0]
        print 'Test accuracy:', score[1]
        #print 'Time elapsed: ',(end - start), "seconds"

        testData[0+(i*4)].append(k)
        testData[1+(i*4)].append(score[0])
        testData[2+(i*4)].append(score[1])
        testData[3+(i*4)].append(end - start)


    with open(csvName, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(testData)
