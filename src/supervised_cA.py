
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
#from load_valid_data import load_valid_data

from multiple_regression import RegressionLayer 
from mlp import HiddenLayer, MLP

class cA(object):

    def __init__(self, numpy_rng, input=None, n_visible=1631, n_hidden=400,
                 n_batchsize=1, W=None, bhid=None, bvis=None):

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize

        if not W:
            initial_W = np.asarray(numpy_rng.uniform(
                      low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
                      high=4 * np.sqrt(6. / (n_hidden + n_visible)),
                      size=(n_visible, n_hidden)),
                                      dtype=theano.config.floatX)
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
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        if input == None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_jacobian(self, hidden, W):
        """Computes the jacobian of the hidden layer with respect to
        the input, reshapes are necessary for broadcasting the
        element-wise product on the right axis

        """
        return T.reshape(hidden * (1 - hidden),
                         (self.n_batchsize, 1, self.n_hidden)) * T.reshape(
                             W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, contraction_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the cA """

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        J = self.get_jacobian(y, self.W)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        self.L_rec = - T.sum(self.x * T.log(z) +
                             (1 - self.x) * T.log(1 - z),
                             axis=1)

        # Compute the jacobian and average over the number of samples/minibatch
        self.L_jacob = T.sum(J ** 2) / self.n_batchsize

        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(self.L_rec) + contraction_level * T.mean(self.L_jacob)

        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - learning_rate * gparam))

        return (cost, updates)

def test_cA(learning_rate=0.002, training_epochs=10,
            dataset='test2.pkl.gz', n_epochs=100,
            batch_size=5, contraction_level=0.001):


    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    # validation/testの時はミニバッチを使わない
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] #/ batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] #/ batch_size

    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.vector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    # 乱数シード
    rng = np.random.RandomState(1234)

    regressor = MLP(rng=rng, input=x, n_in=1631,
                        n_hidden=800)


    ca = cA(numpy_rng=rng, input=x,
            n_visible=1631, n_hidden=800, n_batchsize=batch_size,
            W = regressor.hiddenLayer.W, bhid = regressor.hiddenLayer.b)

    cost, updates = ca.get_cost_updates(contraction_level=contraction_level,
                                        learning_rate=0.001)

    train_ca = theano.function([index], [T.mean(ca.L_rec), ca.L_jacob],
                               updates=updates,
                               givens={x: train_set_x[index * batch_size:
                                                    (index + 1) * batch_size]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############
    print '... training the model'
    epoch = 0

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_ca(batch_index))

        c_array = np.vstack(c)
        print 'Training epoch %d, reconstruction cost ' % epoch, np.mean(
            c_array[0]), ' jacobian norm ', np.mean(np.sqrt(c_array[1]))
        a = regressor.hiddenLayer.W.eval()
        print np.sum(a)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    cost = regressor.mean_square_error(y)


    gparams = []
    for param in regressor.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    updates = []

    for param, gparam in zip(regressor.params, gparams):
        updates.append((param, param - learning_rate * gparam))

    # 学習
    train_model = theano.function(inputs=[index], outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    # validation
    validate_model = theano.function(inputs=[index],
            outputs=regressor.prediction,
            givens={
                x: valid_set_x[index:(index + 1)]})
            
    # validationデータセット
    valid_datasets = load_valid_data(dataset)
    valid = valid_datasets[1][1]

    # 学習データの再現validation
    #valid_datasets = load_valid_data(dataset)
    #valid = valid_datasets[0][1]

    start_time = time.clock()

    best_corrcoef = 0

    epoch = 0

    while epoch < n_epochs:
        epoch = epoch + 1

        for minibatch_index in xrange(n_train_batches):
            # 学習
            train_fn = train_model(minibatch_index)

        if epoch % 1 == 0:
            # validation
            predictions = np.array([validate_model(i) for i in xrange(n_valid_batches)])
            preds = np.reshape(predictions,(47,))

            # ピアソン相関係数の計算
            this_corrcoef = np.corrcoef(preds, valid)[0][1]
            print epoch, this_corrcoef


    end_time = time.clock()

    print ('calculation time %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    test_cA()
