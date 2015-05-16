# -*- coding:utf-8 -*-
import cPickle
import gzip
import os
import time


import numpy as np

import theano
import theano.tensor as T

from load_data import load_data


class RegressionLayer(object):
    # 線形回帰


    def __init__(self, input, n_in, W=None, b=None):

        
        # 重みの初期化
        if W is None:
            self.W = theano.shared(value=np.zeros((n_in,),
                                                dtype=theano.config.floatX),
                                    name='W', borrow=True)
        else:
            self.W = W

        # バイアスの初期化
        if b is None:
            self.b = theano.shared(value=0.0, name='b')
        else:
            self.b = b


        self.input = input
        # 線形回帰 input * 重み + bias
        self.prediction = T.dot(self.input, self.W) + self.b

        # 微分対象のパラメータ
        self.params = [self.W, self.b]

        # Adagrad
        self.hist_grad_W = theano.shared(np.zeros((n_in,), 
                                            dtype = theano.config.floatX),
                                name='hist_grad_W', borrow=True)

        self.hist_grad_b = theano.shared(value = 0.0, name='hist_grad_W')

        self.hist_grads = [self.hist_grad_W, self.hist_grad_b]

        # Adadelta
        self.accu_grad_W = theano.shared(np.zeros((n_in,), 
                                            dtype = theano.config.floatX),
                                name='accu_grad_W', borrow=True)

        self.accu_grad_b = theano.shared(value = 0.0, name='accu_grad_b')

        self.accu_delta_W = theano.shared(np.zeros((n_in,), 
                                            dtype = theano.config.floatX),
                                name='accu_delta_W', borrow=True)

        self.accu_delta_b = theano.shared(value = 0.0, name='accu_delta_b')

        self.accu_grads = [self.accu_grad_W, self.accu_grad_b]

        self.accu_deltas = [self.accu_delta_W, self.accu_delta_b]

    # 二乗誤差の平均(minibatchならこちらのほうが良さそう)
    def mean_square_error(self, y):

        return T.log(T.mean((self.prediction-y) ** 2))
        #return T.mean((self.prediction-y) ** 2)

class integrate_training_way(object):

    def __init__(self, n_in=1631, L1_reg=0.00, L2_reg=0.00):

        self.params = []
        self.hist_grads = []
        self.accu_grads = []
        self.accu_deltas = []

        self.x = T.matrix('x')
        self.y = T.vector('y')

        self.multiRegLayer = RegressionLayer(
                                        input = self.x,
                                        n_in = n_in)

        self.params.extend(self.multiRegLayer.params)
        self.hist_grads.extend(self.multiRegLayer.hist_grads)
        self.accu_grads.extend(self.multiRegLayer.accu_grads)
        self.accu_deltas.extend(self.multiRegLayer.accu_deltas)

        self.L1 = abs(self.multiRegLayer.W).sum()
        self.L2 = (self.multiRegLayer.W ** 2).sum()

        self.finetune_cost = self.multiRegLayer.mean_square_error(self.y) \
                            + self.L1 * L1_reg \
                            + self.L2 * L2_reg

        self.prediction = self.multiRegLayer.prediction

    def build_sgd_functions(self, datasets, batch_size, 
                            learning_way,
                            learning_rate = 0):

        if learning_way == 'adadelta':
            rho = learning_rate


        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        #test_set_x, test_set_y = datasets[2]

        # トレーニングデータ数
        n_train = train_set_x.get_value(borrow=True).shape[0]
        # 繰り返し回数 = データ数/バッチサイズ
        #n_train_batches = n_train / batch_size
        # validationの時はbatch_sizeは1つ
        n_valid = valid_set_x.get_value(borrow=True).shape[0]

        #print '... building the model'

        # データの形式を宣言
        index = T.lscalar('index')
        eps = 1e-8

        # コスト関数を微分
        gparams = T.grad(self.finetune_cost, self.params)

        # 学習方法の設定
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

def test_multiple_regression(learning_way = 'normal',
                            learning_rate=0.001,
                            n_epochs=100,
                            dataset='test2.pkl.gz',
                            batch_size=1,
                            L1_reg = 0.00,
                            L2_reg = 0.00):

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

    print '... building the model'
    regressor = integrate_training_way(n_in=1631,L1_reg=L1_reg, L2_reg=L2_reg)

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

if __name__ ==  '__main__':
    test_multiple_regression(
            learning_way = 'adadelta',
            learning_rate = 0.9,
            n_epochs = 1000,
            dataset = 'test2.pkl.gz',
            batch_size = 1,
            L1_reg=0.005,
            L2_reg=0.001)










