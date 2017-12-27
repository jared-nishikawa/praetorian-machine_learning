# Praetorian Machine Learning Challenge

## Introduction

This is my code for the solution to a challenge by [Praetorian Cybersecurity Solutions](https://www.praetorian.com/challenges/machine-learning).

From the website:
> The crux of the challenge is to build a classifier that can automatically identify and categorize the instruction set architecture of a random binary blob. Train a machine learning classifier to identify the architecture of a binary blob given a list of possible architectures. We currently support twelve architectures, including: avr, alphaev56, arm, m68k, mips, mipsel, powerpc, s390, sh4, sparc, x86_64, and xtensa.

The challenge implies that you need to get 500 correct classifications in a row in order to win, but my final solution certainly did not achieve that, which means I think the challenge was to get a very high accuracy (my final solution achieved 2 wrong in 502 guesses).

## My Solution

Praetorian provides some [basic hints](https://p16.praetorian.com/blog/machine-learning-tutorial) to get started.  This challenge is essentially a text classification puzzle.

They recommend using frequencies of bytes and two-, three-, and four-byte "words" to create a feature vector.  More than that, they recommend the [tf-idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) statistic to normalize these vectors.

My full solution consisted of the following:

### Prepping the data

- First gather training data.
- For each of the 12 architectures, record the top 16 (with the highest tf-idf score) two-, three-, and four-byte words.  (This gives us 12*16 = 192 possible two-, three-, and four-byte words).
- For each set of 192 words, pick the 100 with the highest tf-idf score.
- My feature vector will consist of:
  - 256 tf-idf scores for each possible byte (0-255)
  - 100 tf-idf scores for the top two-byte words
  - 100 tf-idf scores for the top three-byte words
  - 100 tf-idf scores for the top four-byte words
- That gives us a feature vector of size 556, not unmanageable.

### Fitting the model

My first attempt was naive.  I created an average "profile" for each architecture (using the feature vector), then tried to minimize the difference for a sample from each profile to classify it.  This method gave about 65% accuracy.

My second attempt was to use K-nearest neighbors.  Still using the same feature vector, I used a simple Euclidean measure to compare a sample to the 16 nearest neighbors in the training data.  This was time-consuming and only achieved about 70% accuracy.  I also tried a measure called [Normalized Compression Distance](https://en.wikipedia.org/wiki/Normalized_compression_distance) on the binary data itself, which yielded similar accuracy.  I think the idea of NCD is fascinating, and while it didn't end up working for me, I'm sure it will be useful to know about in the future.

I finally decided to try writing a neural network.  I had never written a neural net before, and to make sure I really did understand how it worked, I wrote it from scratch.

I experimented with the number of neurons in the hidden layer, as well as the learning rate, and the number of epochs.  I settled on 20 hidden neurons, a learning rate of 0.01, and 10000 epochs.

When I tested my resulting neural net model against the challenge server, I achieved 98% accuarcy (around 1 or 2 incorrect guesses in every 100).

I was pretty happy with this, although not completely satisfied.  However, I decided I could allow myself to use build-in tools.

After doing some research, I decided to try Python's keras.  I created a neural network with a single dense hidden layer and fitted it to the training data.

Finally, I was getting near-perfect accuracy.  This was the try where I got 500/502 correct guesses, better than 99.6% accuracy.
