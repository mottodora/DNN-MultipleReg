# -*- coding:utf-8 -*-
import cPickle
import gzip
import os
import time
import sys


import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from load_data import load_data

class dA(object):

    def __init__(self, numpy_rng, theano_rng=None, input=None,
                 n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W:
            initial_W = np.asarray(numpy_rng.uniform(
                      low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=np.zeros(n_visible,
                                         dtype=theano.config.floatX),
                                 borrow=True)

        if not bhid:
            bhid = theano.shared(value=np.zeros(n_hidden,
                                                   dtype=theano.config.floatX),
                                 name='b',
                                 borrow=True)

        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T
        self.theano_rng = theano_rng

        if input == None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    # SdA用
    def get_corrupted_input(self, input, corruption_level):

        return  self.theano_rng.binomial(size=input.shape, n=1,
                                         p= 1 - corruption_level,
                                         dtype=theano.config.floatX) * input
        J
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)


#def test_dA(learning_rate=0.2, training_epochs=20,
#            dataset='test2.pkl.gz',
#            batch_size=10):
#
#
#    datasets = load_data(dataset)
#    train_set_x, train_set_y = datasets[0]
#
#    # compute number of minibatches for training, validation and testing
#    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
#
#    # allocate symbolic variables for the data
#    index = T.lscalar()    # index to a [mini]batch
#    x = T.matrix('x')  # the data is presented as rasterized images
#
#    rng = np.random.RandomState(123)
#    theano_rng = RandomStreams(rng.randint(2 ** 30))
#
#    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
#            n_visible=1631, n_hidden=800)
#
#    cost, updates = da.get_cost_updates(corruption_level=0.3,
#                                        learning_rate=learning_rate)
#
#    train_da = theano.function([index], cost, updates=updates,
#         givens={x: train_set_x[index * batch_size:
#                                (index + 1) * batch_size]})
#
#    start_time = time.clock()
#
#    for epoch in xrange(training_epochs):
#        # go through trainng set
#        c = []
#        for batch_index in xrange(n_train_batches):
#            c.append(train_da(batch_index))
#
#        print 'Training epoch %d, cost ' % epoch, np.mean(c)
#
#    end_time = time.clock()
#
#    training_time = (end_time - start_time)
#
#    print >> sys.stderr, ('The no corruption code for file ' +
#                          os.path.split(__file__)[1] +
#                          ' ran for %.2fm' % ((training_time) / 60.))
#
#if __name__ == '__main__':
#    test_dA()
