#!/usr/bin/python

import pandas as pd
import base64
import math
import time

import ncd

TRAINING_SAMPLES = []
PROFILES = {}

class Sample(object):
    def __init__(self):
        self.architecture = None
        self.binary = None

class ArchProfile(object):
    def __init__(self, name):
        self.name = name
        self.singles = None

    def set_profile(self, blobs, n=0, k=0):
        self.singles = byte_averages(blobs)
        self.tfidf = tfidf_averages(blobs)
        self.top_grams = {}
        for i in range(2, n+1):
            ngrams = ngram_averages(blobs,i)
            self.top_grams[i] = top_ngrams(ngrams, k)

    # Lower is better
    def single_score(self, blob):
        profile = byte_profile(blob)
        S = sum_square_difference(profile, self.singles)
        return S

    # Lower is better
    def tfidf_score(self, blob):
        profile = tfidf_profile(blob)
        S = sum_square_difference(profile, self.tfidf)
        return S

    # Higher is better
    def n_score(self, blob, n, k):
        ngrams = ngram_analysis(blob, n)
        blob_top_ngrams = top_ngrams(ngrams, k)
        # Compare to self.top_grams[n]
        A = [g[0] for g in blob_top_ngrams]
        B = [g[0] for g in self.top_grams[n]]
        intersection = [a for a in A if a in B]
        return len(intersection)

# Lower is better
def sum_square_difference(A, B):
    assert len(A) == len(B)
    total = 0
    for i in range(len(A)):
        total += abs(A[i] - B[i])**2
    return total

def majority(A):
    counts = {}
    for a in A:
        counts[a] = counts[a] + 1 if counts.has_key(a) else 1
    L = [(counts[key], key) for key in counts.keys()]
    L.sort(reverse=True)
    return L[0][1]

# W is given in a list of tuples (score, name)
def weighted_majority(W, reverse=False):
    weights = {}
    for w in W:
        weights[w[1]] = weights[w[1]] + float(1)/(1+w[0]) if \
                weights.has_key(w[1]) else float(1)/(1+w[0])
    L = [(weights[key], key) for key in weights.keys()]
    L.sort(reverse=True)
    return L[0][1]
       
# Term frequency - inverse document frequency
def tfidf(word, binary):
    tf = binary.count(word)
    num_docs = 0
    for sample in TRAINING_SAMPLES:
        if word in sample.binary:
            num_docs += 1
    if num_docs == 0:
        return 0
    idf = math.log(float(len(TRAINING_SAMPLES))/num_docs)
    return tf*idf

def ngram_analysis(blob,n):
    ngrams = {}
    for i in range(len(blob)-n+1):
        ngram = blob[i:i+n]
        if not ngrams.has_key(ngram):
            ngrams[ngram] = 0
        ngrams[ngram] += 1
    return ngrams

def ngram_averages(blobs,n):
    ngram_avgs = {}
    for blob in blobs:
        ngrams = ngram_analysis(blob, n)
        for key in ngrams.keys():
            ngram_avgs[key] = ngram_avgs[key] + 1 if ngram_avgs.has_key(key) else 1
    return ngram_avgs
    for key in ngram_avgs:
        # This gives a ratio of number of each ngram per blob
        ngram_avgs[key] = round(float(ngram_avgs[key])/len(blobs),2)

    return ngram_avgs

def top_ngrams(ngrams,k):
    pairs = [(key, ngrams[key]) for key in ngrams.keys()]
    pairs.sort(key=lambda x: x[1])
    top_pairs = pairs[-k:]
    return top_pairs

def endian(blob):
    twograms = [blob[i:i+2] for i in range(len(blob)-1)]
    A = twograms.count('\x01\x00') # occurs more often in little endian
    B = twograms.count('\x00\x01') # occurs more often in big endian
    C = twograms.count('\xfe\xff') # occurs more often in little endian
    D = twograms.count('\xff\xfe') # occurs more often in big endian
    if A > B and C > D:
        return "little"
    if B > A and D > C:
        return "big"
    else:
        return "unknown"

def tfidf_profile(blob):
    S = [0]*256
    for b in set(blob):
        S[ord(b)] = round(tfidf(b, blob),2)
    return S

def tfidf_averages(blobs):
    all_blobs = [b for blob in blobs for b in blob]
    S = tfidf_profile(all_blobs)
    avgs = [round(float(s)/len(blobs),2) for s in S]
    return avgs

def byte_profile(blob):
    S = [0]*256
    for b in blob:
        S[ord(b)] += 1
    return S

def byte_averages(blobs):
    all_blobs = [b for blob in blobs for b in blob]
    S = byte_profile(all_blobs)
    # This gives a ratio of number of each byte per blob
    avgs = [round(float(s)/len(blobs),2) for s in S]
    return avgs

def knn(blob, k, targets=None, measure='ssd'):
    profile = byte_profile(blob)
    neighbors = []
    for sample in TRAINING_SAMPLES:
        sample_profile = byte_profile(sample.binary)

        # Sum square difference score
        if measure == 'ssd':
            score = sum_square_difference(profile, sample_profile)

        # Normalized compression distance score
        elif measure == 'ncd':
            score = ncd.ncd(blob, sample.binary)
        else:
            raise Exception("Unknown measure")

        if targets and sample.architecture not in targets:
            continue
        neighbors.append((score, sample.architecture))
    neighbors.sort()
    #maj = majority([neighbor[1] for neighbor in neighbors[:k]])
    maj = weighted_majority(neighbors[:k])
    return maj

def train(n=2,k=16):
    global TRAINING_SAMPLES
    global PROFILES
    df = pd.read_csv('train.csv')
    archs = {}
    for row in df.itertuples():
        binary = base64.b64decode(row.binary)
        if not archs.has_key(row.architecture):
            archs[row.architecture] = []
        archs[row.architecture].append(binary)

        # initialize TRAINING_SAMPLES
        sample = Sample()
        sample.architecture = row.architecture
        sample.binary = binary
        TRAINING_SAMPLES.append(sample)
    for arch in archs.keys():
        P = ArchProfile(arch)
        P.set_profile(archs[arch], n=n,k=k)
        PROFILES[arch] = P

def profile_match(blob, targets=None):
    guesses = []
    for profile in PROFILES.values():
        new_guess = (profile.single_score(blob), profile.name)
        if targets and new_guess[1] not in targets:
            continue
        guesses.append(new_guess)
    guesses.sort()
    return guesses[0][1]

def tfidf_match(blob, targets=None):
    guesses = []
    for profile in PROFILES.values():
        new_guess = (profile.tfidf_score(blob), profile.name)
        if targets and new_guess[1] not in targets:
            continue
        guesses.append(new_guess)
    guesses.sort()
    return guesses[0][1]

# n is ngram
# k is top k ngrams
def ngram_match(blob,n=2,k=16, targets=None):
    guesses = []
    for profile in PROFILES.values():
        new_guess = (profile.n_score(blob,n,k), profile.name)
        if targets and new_guess not in targets:
            continue
        guesses.append(new_guess)
    guesses.sort(reverse=True)
    return guesses[0][1]

def test():
    df = pd.read_csv('test.csv')
    total = 0
    correct = 0
    wrong = []
    tricky = []
    for row in df.itertuples():
        blob = base64.b64decode(row.binary)
        true_arch = row.architecture
        #guessed_archs = [profile_match(blob), tfidf_match(blob), \
        #        ngram_match(blob), ngram_match(blob, n=3)]
        #guessed_arch = majority(guessed_archs)
        guessed_arch_a = knn(blob, 16, measure='ssd')
        guessed_arch_b = knn(blob, 16, measure='ncd')
        #print "Guessed arch:", guessed_arch
        #print "True arch:", true_arch
        #raw_input()
        total += 1
        if true_arch == guessed_arch_a == guessed_arch_b:
            correct += 1
        else:
        #    wrong.append(guessed_arch)
            tricky.append(true_arch)

        if guessed_arch_a != guessed_arch_b:
            wrong.append((guessed_arch_a, guessed_arch_b, true_arch))


    return total, correct, wrong, tricky

def timeit(func, label=None, *args, **kwargs):
    start = time.time()
    result = func(*args,**kwargs)
    end = time.time()
    print "Function %s took %.2f seconds" % (label, end-start)
    return result

if __name__ == '__main__':
    timeit(train, "Train", n=4)
    t,c,w,y = timeit(test, "Test")
    print "Total:", t
    print "Correct:", c
    print "Pct correct: %d%%" % (float(c*100)/t)
    print "Wrong:", w
    print "Tricky:", y

