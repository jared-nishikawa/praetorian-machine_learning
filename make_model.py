#!/usr/bin/python

import tools
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

if __name__ == '__main__':
    C = tools.read_data('train.csv')
    qty = 100
    C.set_common_words(qty=qty)
    C.set_vector_mapping(qty=qty)
    X_train = []
    Y_train = []
    for doc in C.documents:
        sample = doc.clean()
        X_train.append(sample)
        Y_train.append(tools.categories(doc.arch))

    # Must be np arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    #print X_train[0]
    #print Y_train[0]

    # Define model architecture
    print "Creating model..."
    model = Sequential()
    model.add(Dense(32, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(12, activation='sigmoid'))

    # Compile model
    print "Compiling model..."
    model.compile(loss='categorical_crossentropy',
            optimizer='adagrad',
            metrics=['accuracy'])

    # Fit model
    print "Fitting model..."
    model.fit(X_train, Y_train, batch_size=32, epochs=150, verbose=1)

    # Evaluate model
    print "Evaluating model..."
    scores = model.evaluate(X_train, Y_train)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Save model
    print "Saving model..."
    model.save("model.h5")



