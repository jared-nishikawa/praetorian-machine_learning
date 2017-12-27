#!/usr/bin/python

import base64
import json
import math
import numpy as np
import pandas as pd

ARCHS = ['avr', 'alphaev56', 'arm', 'm68k', 'mips', 'mipsel', \
        'powerpc', 's390', 'sh4', 'sparc', 'x86_64', 'xtensa']

GLOBAL_MAX = 64

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def categories(arch):
    sample_arch = [0]*12
    ind = ARCHS.index(arch)
    sample_arch[ind] = 1
    return sample_arch

def incorporate(dst_dict, tuples):
    for v,k in tuples:
        dst_dict[k] = dst_dict[k] + v if dst_dict.has_key(k) else v

def top(src_dict):
    tuples = [(src_dict[key], key) for key in src_dict]
    tuples.sort(reverse=True)
    return tuples

class Corpus(object):
    def __init__(self):
        self.documents = []
        self.word_in_doc = {}
        self.ngrams = {}

        # This will be a mapping of 1,2,3,4-letter words
        # to indexes of a vector of size 556
        self.vector_mapping = {}

    def common_words_by_arch(self, arch, w_size=2, qty=16):
        word_counts = {}
        for doc in self.documents:
            if doc.arch != arch:
                continue
            words = doc.most_tfidf_words(w_size, qty=qty)
            incorporate(word_counts, words)
            #for count,word in words:
            #    word_counts[word] = word_counts[word] + count if word_counts.has_key(word) else count
        tuples = top(word_counts)
        return tuples[:qty]

    def common_words(self, w_size=2, qty=100):
        all_common = {}
        for arch in ARCHS:
            tuples = self.common_words_by_arch(arch, w_size=w_size)
            incorporate(all_common, tuples)
        tuples = top(all_common)
        return tuples[:qty]

    def set_common_words(self, max_size=4, qty=100):
        for w_size in range(2, max_size+1):
            tuples = self.common_words(w_size=w_size, qty=qty)
            just_words = [tup[1] for tup in tuples]
            just_words.sort()
            self.ngrams[w_size] = just_words

    # This is not very flexible right now
    def set_vector_mapping(self, qty=100):
        for i in range(256):
            c = chr(i)
            self.vector_mapping[c] = i
        for n in range(2,5):
            ngrams = self.ngrams[n]
            incorporate(self.vector_mapping, [(256+qty*(n-2) + ind, ngram) \
                    for ind,ngram in enumerate(ngrams)])

class Document(object):
    def __init__(self, arch, binary, corpus, max_size=4):
        self.arch = arch
        self.binary = binary
        self.corpus = corpus
        self.check_all_words(max_size)

    # This is like a bag of words, but weighted by tfidf
    def clean(self, qty=100):
        # 256 possible one-byte words
        # Top 16 two-byte words for all twelve archs (pick top 100)
        # Top 16 three-byte words for all twelve archs (pick top 100)
        # Top 16 four-byte words for all twelve archs (pick top 100)
        # Total is 556
        new_array = [0]*(256 + 3*qty)
        W = set([])
        W = W.union(self.doc_words(1), self.doc_words(2), \
                self.doc_words(3), self.doc_words(4))
        for w in W:
            if self.corpus.vector_mapping.has_key(w):
                ind = self.corpus.vector_mapping[w]
                new_array[ind] = self.tfidf(w)

        # Testing something weird here
        M = max(new_array)
        # Add one for the bias
        new_array += [M]

        # Normalize out of M, because why not
        np_array = np.array(map(float, new_array))
        np_array /= M

        return np_array

    def most_tfidf_words(self, w_size, qty=256):
        word_tfidf = {}
        for word in self.doc_words(w_size):
            word_tfidf[word] = self.tfidf(word)
        tuples = top(word_tfidf)
        return tuples[:qty]

    def tfidf(self, word):
        tf = self.binary.count(word)/64.0
        num_docs = 1 + self.corpus.word_in_doc[word]

        # If a word shows up in EVERY document, then idf is zero
        idf = math.log(float(len(self.corpus.documents))/num_docs)
        return tf*idf

    def doc_words(self, w_size, as_list=False):
        unaltered = [self.binary[i:i+w_size] for i in range(len(self.binary) - w_size + 1)]
        if as_list:
            return unaltered
        return set([self.binary[i:i+w_size] for i in range(len(self.binary) - w_size + 1)])

    def check_words(self, w_size):
        words = self.doc_words(w_size)
        incorporate(self.corpus.word_in_doc, [(1,word) for word in words])

    def check_all_words(self, max_size=4):
        for w_size in range(1, max_size+1):
            self.check_words(w_size)



def sum_square_diff(A,B):
    assert len(A) == len(B)
    total = 0
    for i in range(len(A)):
        total += (A[i] - B[i])**2
    return total

def read_data(training_data):
    C = Corpus()
    df = pd.read_csv(training_data)
    for row in df.itertuples():
        binary = base64.b64decode(row.binary)
        arch = row.architecture
        D = Document(arch, binary, C)
        C.documents.append(D)
    return C
