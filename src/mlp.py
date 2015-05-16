# -*- coding:utf-8 -*-
import cPickle
import gzip
import os
import sys
import time

import numpy as np

import theano
import theano.tensor as T

from load_data import load_data

from multiple_regression import RegressionLayer

# 活性化関数
#### rectified linear unit
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
#### tanh
def Tanh(x):
    y = T.tanh(x)
    return(y)


class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, 
                activation, W=None, b=None,
                use_bias=False):

        self.input = input
        self.activation = activation

        # 重みWの乱数初期化
        # 不思議な初期化ですが、Practical Recommendationsに書いてあります 
        if W is None:
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == Sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        # Adagrad
        self.hist_grad_W = theano.shared(np.zeros((n_in,n_out), 
                                            dtype = theano.config.floatX),
                                name='hist_grad_W', borrow=True)

        self.hist_grad_b = theano.shared(np.zeros((n_out,), 
                                            dtype = theano.config.floatX),
                                name='hist_grad_b', borrow=True)

        # Adadelta
        self.accu_grad_W = theano.shared(np.zeros((n_in, n_out), 
                                            dtype = theano.config.floatX),
                                name='accu_grad_W', borrow=True)

        self.accu_grad_b = theano.shared(np.zeros((n_out,), 
                                            dtype = theano.config.floatX),
                                name='accu_grad_b', borrow=True)

        self.accu_delta_W = theano.shared(np.zeros((n_in, n_out), 
                                            dtype = theano.config.floatX),
                                name='accu_delta_W', borrow=True)

        self.accu_delta_b = theano.shared(np.zeros((n_out,), 
                                            dtype = theano.config.floatX),
                                name='accu_delta_b', borrow=True)

        # 隠れ層の出力
        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None \
                                                else activation(lin_output))
    
        # 隠れ層のパラメータ
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]

        if use_bias:
            self.hist_grads = [self.hist_grad_W, self.hist_grad_b]
        else:
            self.hist_grads = [self.hist_grad_W]

        if use_bias:
            self.accu_grads = [self.accu_grad_W, self.accu_grad_b]
            self.accu_deltas = [self.accu_delta_W, self.accu_delta_b]
        else:
            self.accu_grads = [self.accu_grad_W]
            self.accu_deltas = [self.accu_delta_W]


class MLP(object):

    def __init__(self, rng, layer_sizes, activations, use_bias=False):
        self.params = []

        self.x = T.matrix('x')
        self.y = T.vector('y')
        # Ex)
        #   layer_sizes = [1631, 800, 400, 1]
        #   weight_matrix_sizes = [(1631, 800), (800, 400), (400, 1)]
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])

        self.layers = []
        self.hist_grads = []
        self.accu_grads = []
        self.accu_deltas = []


        layer_counter = 0
        next_layer_input = self.x

        # Ex)
        #   weight_matrix_sizes[:-1] = [(1631, 800), (800, 400)]
        for n_in, n_out in weight_matrix_sizes[:-1]:

            next_layer = HiddenLayer(rng = rng, input = next_layer_input,
                                        n_in = n_in, n_out = n_out,
                                        activation = activations[layer_counter],
                                        W = None, b = None, use_bias = use_bias)
            self.params.extend(next_layer.params)
            self.hist_grads.extend(next_layer.hist_grads)
            self.accu_grads.extend(next_layer.accu_grads)
            self.accu_deltas.extend(next_layer.accu_deltas)

            self.layers.append(next_layer)
            next_layer_input = next_layer.output

            layer_counter = layer_counter + 1

        # n_outは1で固定
        n_in = layer_sizes[-2]

        output_layer = RegressionLayer(input = next_layer_input,
                                            n_in = n_in, W = None, b = None)
        self.params.extend(output_layer.params)
        self.hist_grads.extend(output_layer.hist_grads)
        self.accu_grads.extend(output_layer.accu_grads)
        self.accu_deltas.extend(output_layer.accu_deltas)
        self.layers.append(output_layer)

        # build_sgd_functionsのほうに移した方いい(L1, L2正則化)
        self.finetune_cost = self.layers[-1].mean_square_error(self.y)

        self.prediction = self.layers[-1].prediction


    def build_sgd_functions(self, datasets, batch_size, 
                            learning_way,
                            learning_rate = 0.001):

        if learning_way == 'adadelta':
            rho = learning_rate

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]

        
        # トレーニングデータ数
        n_train = train_set_x.get_value(borrow=True).shape[0]
        # validationデータ数
        n_valid = valid_set_x.get_value(borrow=True).shape[0]

        index = T.lscalar('index')
        eps = 1e-8

        # コスト関数を微分
        gparams = T.grad(self.finetune_cost, self.params)

        updates = []

        if learning_way == 'normal':
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - gparam * learning_rate))

        elif learning_way == 'adagrad':
            for param_i, grad_i, hist_grad_i \
                        in zip(self.params, gparams, self.hist_grads):
                new_hist_grad_i = hist_grad_i + T.sqr(grad_i)
                updates.append((hist_grad_i, new_hist_grad_i))
                updates.append((param_i,  param_i -
                    learning_rate /(eps + T.sqrt(new_hist_grad_i)) * grad_i))

        elif learning_way == 'adadelta':
            # Adagradによる学習率調整
            for param_i, grad_i, accu_grad_i, accu_delta_i \
                    in zip(self.params, gparams, \
                            self.accu_grads, self.accu_deltas):
                
                agrad = rho * accu_grad_i + (1 - rho) * grad_i * grad_i
                dx = -T.sqrt((accu_delta_i + eps) / (agrad + eps)) * grad_i
                updates.append((accu_grad_i, agrad))
                updates.append((accu_delta_i, rho * accu_delta_i \
                                                    + (1 - rho) * dx * dx))
                updates.append((param_i, param_i + dx))

        # 学習関数
        # 一回の学習はバッチ数だけ行う #todo
        train_fn = theano.function(inputs=[index],
                outputs=self.finetune_cost,
                updates=updates,
                givens={ 
                    self.x: train_set_x[index * batch_size:
                                                    (index + 1) * batch_size], 
                    self.y: train_set_y[index * batch_size:
                                                    (index + 1) * batch_size]}) 
        # validation用関数 # 学習は行わない 
        valid_fn = theano.function(inputs=[index], 
                outputs=self.prediction, 
                givens={
                    self.x: valid_set_x[index:index+1]})

        # validation用関数
        # 学習データの再現に使う
        valid_train_fn = theano.function(inputs=[index],
                outputs=self.prediction,
                givens={
                    self.x: train_set_x[index:index+1]})

        def valid_score():
            return np.reshape(np.array([valid_fn(i) for i in xrange(n_valid)]),
                    (n_valid,))

        def valid_train_score():
            return np.reshape(
                    np.array([valid_train_fn(i) for i in xrange(n_train)]),
                    (n_train,))

        return train_fn, valid_score, valid_train_score

def test_mlp(learning_way = 'normal',
            learning_rate=0.001,
            n_epochs=100,
            dataset='test2.pkl.gz',
            batch_size=1,
            #L1_reg = 0.00,
            #L2_reg = 0.00,
            layer_sizes=[1631, 800, 400, 1],
            activations=[ReLU,ReLU],
            use_bias=False,
            random_seed=1234):

    datasets = load_data(dataset)

    train_datasets = datasets.get_shared_dataset()

    train_set_x, train_set_y = train_datasets[0]
    valid_set_x, valid_set_y = train_datasets[1]
    test_set_x, test_set_y = train_datasets[2]

    # トレーニングデータ数
    n_train = train_set_x.get_value(borrow=True).shape[0]
    # 繰り返し回数 = データ数/バッチサイズ
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    # validation/testの時は1つ1つのpearson相関係数がほしいのでbatch_sizeは1つ
    n_valid = valid_set_x.get_value(borrow=True).shape[0]

    #n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    valid_datasets = datasets.get_ndarray_dataset()
    valid_dataset = valid_datasets[1][1]
    valid_train_dataset = valid_datasets[0][1]

    rng = np.random.RandomState(random_seed)

    print '... building the model'

    regressor = MLP(rng=rng, layer_sizes = layer_sizes,
                        activations = activations, use_bias = use_bias)


    train_fn, valid_fn, valid_train_fn = regressor.build_sgd_functions(
                                                datasets=train_datasets,
                                                batch_size=batch_size,
                                                learning_way=learning_way,
                                                learning_rate=learning_rate)


    epoch = 0
    best_corrcoef = 0
    best_epoch = 0

    start_time = time.clock()

    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            # 学習
            train_model = train_fn(minibatch_index)

        if epoch % 1 == 0:
            # validation
            valid_preds = valid_fn()
            valid_train_preds = valid_train_fn()

            this_train_corrcoef = np.corrcoef(valid_train_preds, 
                                                    valid_train_dataset)[0][1]
            this_valid_corrcoef = np.corrcoef(valid_preds, 
                                                    valid_dataset)[0][1]
            print epoch, this_train_corrcoef, this_valid_corrcoef
            if best_corrcoef < this_valid_corrcoef:
                best_corrcoef = this_valid_corrcoef
                best_epoch = epoch

    end_time = time.clock()

    print learning_way, learning_rate
    print best_epoch, best_corrcoef
    print ('calculation time %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    test_mlp(
            learning_way = 'adadelta',
            learning_rate = 0.9,
            n_epochs = 1000,
            dataset = 'test2.pkl.gz',
            batch_size = 4,
            layer_sizes = [1631, 800, 1],
            activations = [ReLU],
            use_bias = False,
            random_seed = 1234)

