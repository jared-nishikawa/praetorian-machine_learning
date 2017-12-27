#!/usr/bin/python

from keras.models import load_model
import tools
import numpy as np

def guess(values):
    """Take a list of 12 values, and return the index of
    the max value (as well as the actual maximum).
    This list is the output from the neural net.
    """
    V = list(values)
    M = max(V)
    ind = V.index(M)
    return tools.ARCHS[ind], M

if __name__ == '__main__':
    # Load model
    model = load_model('model.h5')

    # Read in train.csv for corpus
    C = tools.read_data('train.csv')
    qty = 100
    C.set_common_words(qty=qty)
    C.set_vector_mapping(qty=qty)

    # Read in test.csv using a dummy corpus
    X_test = []
    Y_test = []
    C_dummy = tools.read_data('test.csv')
    for doc in C_dummy.documents:
        doc.corpus = C
        sample = doc.clean()
        X_test.append(sample)
        Y_test.append(doc.arch)

    # Must be np arrays
    X_test = np.array(X_test)

    # Predictions
    Z = model.predict(X_test, batch_size=32, verbose=1)
    correct = 0
    total = 0
    for i,z in enumerate(Z):
        total += 1
        arch, conf = guess(z)
        print "Predicted:", arch
        print "Confidence:", conf*100
        print "True:", Y_test[i]
        if arch == Y_test[i]:
            correct += 1
    print "Accuracy:", "%.2f" % (float(correct)*100/total)


