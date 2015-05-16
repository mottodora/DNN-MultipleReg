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
from mlp import ReLU, Sigmoid, Tanh
from dropout import HiddenLayer, _dropout_from_layer, DropoutHiddenLayer
from dA import dA
from supervised_cA import cA

class DNN(object):

    def __init__(self, rng, theano_rng, layer_sizes, activations, 
                            dropout_rates, pretrain_way, 
                            batch_size, use_bias=False):

        self.x = T.matrix('x')
        self.y = T.vector('y')
        # Ex)
        #   layer_sizes = [1631, 800, 400, 1]
        #   weight_matrix_sizes = [(1631, 800), (800, 400), (400, 1)]
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])

        self.params = []
        self.layers = []
        self.dropout_layers = []
        self.pretrain_layers = []
        self.hist_grads = []
        self.accu_grads = []
        self.accu_deltas = []

        layer_counter = 0
        next_layer_input = self.x
        next_dropout_layer_input = _dropout_from_layer(rng, self.x, \
                                                        p = dropout_rates[0])

        # Ex)
        #   weight_matrix_sizes[:-1] = [(1631, 800), (800, 400)]
        for n_in, n_out in weight_matrix_sizes[:-1]:

            next_dropout_layer = DropoutHiddenLayer(
                    rng=rng, input=next_dropout_layer_input,
                    n_in=n_in, n_out=n_out,
                    activation=activations[layer_counter],
                    dropout_rate=dropout_rates[layer_counter+1],
                    W = None, b = None, use_bias = False)
            
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            self.params.extend(next_dropout_layer.params)
            self.hist_grads.extend(next_dropout_layer.hist_grads)
            self.accu_grads.extend(next_dropout_layer.accu_grads)
            self.accu_deltas.extend(next_dropout_layer.accu_deltas)

            if pretrain_way == 'denoising':
                pretrain_layer = dA(numpy_rng = rng,
                            theano_rng = theano_rng,
                            input = next_layer_input,
                            n_visible = n_in, n_hidden = n_out,
                            W=next_dropout_layer.W,
                            bhid=next_dropout_layer.b, bvis=None)
                self.pretrain_layers.append(pretrain_layer)
            elif pretrain_way == 'contractive':
                pretrain_layer = cA(numpy_rng = rng,
                            input = next_layer_input,
                            n_visible = n_in, n_hidden = n_out,
                            n_batchsize = batch_size,
                            W=next_dropout_layer.W,
                            bhid=next_dropout_layer.b, bvis=None)
                self.pretrain_layers.append(pretrain_layer)

            next_layer = HiddenLayer(rng = rng, 
                    input = next_layer_input,
                    n_in = n_in, n_out = n_out,
                    activation = activations[layer_counter],
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_counter]),
                    b=next_dropout_layer.b,
                    use_bias = use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output


            #self.layers.append(next_layer)
            #next_layer_input = next_layer.output

            layer_counter = layer_counter + 1

        # n_outは1で固定
        n_in = layer_sizes[-2]

        dropout_output_layer = RegressionLayer(
                input = next_dropout_layer_input,
                n_in = n_in)
        self.dropout_layers.append(dropout_output_layer)

        output_layer = RegressionLayer(
                input = next_layer_input,
                W=dropout_output_layer.W * (1 - dropout_rates[-1]),
                b=dropout_output_layer.b,
                n_in=n_in)
        self.layers.append(output_layer)

        self.params.extend(dropout_output_layer.params)
        self.hist_grads.extend(dropout_output_layer.hist_grads)
        self.accu_grads.extend(dropout_output_layer.accu_grads)
        self.accu_deltas.extend(dropout_output_layer.accu_deltas)


        # dropoutの二乗誤差
        self.dropout_mean_square_error = \
                            self.dropout_layers[-1].mean_square_error(self.y)

        # 二乗誤差
        self.mean_squared_error = self.layers[-1].mean_square_error(self.y)

        self.dropout_prediction = self.dropout_layers[-1].prediction
        self.prediction = self.layers[-1].prediction

    def build_pretraining_function(self, train_set_x, batch_size):
        
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        pretrain_fns = []

        for pretrain in self.pretrain_layers:

            cost, updates = pretrain.get_cost_updates(corruption_level, \
                                                            learning_rate)


            fn = theano.function(inputs=[index, corruption_level, \
                                                        learning_rate],
                    outputs=cost, 
                    updates=updates,
                    givens= {
                        self.x: train_set_x[index * batch_size: \
                                                    (index + 1) * batch_size]})

            pretrain_fns.append(fn)

        return pretrain_fns

    def build_sgd_functions(self, datasets, batch_size, 
                            dropout,
                            learning_way,
                            learning_rate = 0.001):

        if learning_way == 'adadelta':
            rho = learning_rate

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        
        # トレーニングデータ数
        n_train = train_set_x.get_value(borrow=True).shape[0]
        # validationデータ数
        n_valid = valid_set_x.get_value(borrow=True).shape[0]
        n_test = test_set_x.get_value(borrow=True).shape[0]

        index = T.lscalar('index')
        eps = 1e-8

        # cost関数
        cost = self.mean_squared_error
        dropout_cost = self.dropout_mean_square_error

        # コスト関数を微分
        gparams = []
        for param in self.params:
            # Use the right cost function here to train with or without dropout.
            gparam = T.grad(dropout_cost if dropout else cost, param)
            gparams.append(gparam)

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
                outputs=self.dropout_mean_square_error,
                updates=updates,
                givens={ 
                    self.x: train_set_x[index * batch_size:
                                                    (index + 1) * batch_size], 
                    self.y: train_set_y[index * batch_size:
                                                    (index + 1) * batch_size]}) 

        valid_fn = theano.function(inputs=[index],
                outputs=self.dropout_mean_square_error,
                givens={ 
                    self.x: valid_set_x[index:
                                                    (index + 1)], 
                    self.y: valid_set_y[index:
                                                    (index + 1)]}) 

        test_fn = theano.function(inputs=[index],
                outputs=self.dropout_mean_square_error,
                givens={ 
                    self.x: test_set_x[index:
                                                    (index + 1)], 
                    self.y: test_set_y[index:
                                                    (index + 1)]}) 


        # validation用関数 # 学習は行わない 
        valid_prediction_fn = theano.function(inputs=[index], 
                outputs=self.prediction, 
                givens={
                    self.x: valid_set_x[index:index+1]})

        # validation用関数
        # 学習データの再現に使う
        valid_train_fn = theano.function(inputs=[index],
                outputs=self.prediction,
                givens={
                    self.x: train_set_x[index:index+1]})

        valid_test_fn = theano.function(inputs=[index],
                outputs=self.prediction,
                givens={
                    self.x: test_set_x[index:index+1]})

        def valid_score():
            return np.reshape(np.array([valid_prediction_fn(i) for i in xrange(n_valid)]),
                    (n_valid,))

        def valid_train_score():
            return np.reshape(
                    np.array([valid_train_fn(i) for i in xrange(n_train)]),
                    (n_train,))

        def test_score():
            return np.reshape(
                    np.array([valid_test_fn(i) for i in xrange(n_test)]),
                    (n_test,))

        return train_fn, valid_fn, valid_score, valid_train_score, test_score

def test_sda(
            pretrain_lr = 0.001,
            pretrain_way = 'denoising',
            pretrain_epochs=10,
            learning_way = 'normal',
            learning_rate=0.001,
            n_epochs=100,
            dataset='test2.pkl.gz',
            batch_size=1,
            #L1_reg = 0.00,
            #L2_reg = 0.00,
            layer_sizes=[1631, 800, 400, 1],
            activations=[ReLU,ReLU],
            dropout = True,
            dropout_rates = [0.2, 0.5, 0.5],
            corruption_levels = [0.1, 0.2, 0.3],
            use_bias=False,
            random_seed=1234):

    os.chdir("./results/" + dataset.split('.')[0].split('_')[0])
    fp = open(dataset + '_' +  str(learning_rate) + '.res', 'w')

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
    test_dataset = valid_datasets[2][1]

    rng = np.random.RandomState(random_seed)

    #print '... building the model'


    dnn = DNN(rng = rng, theano_rng = None,
                layer_sizes = layer_sizes, activations = activations,
                dropout_rates = dropout_rates,
                pretrain_way = pretrain_way, 
                batch_size = batch_size, use_bias = use_bias)



    #########################
    # PRETRAINING THE MODEL #
    #########################
    #print '... getting the pretraining functions'

    pretrain_fns = dnn.build_pretraining_function(train_set_x = train_set_x,
                                                    batch_size=batch_size)

    #print '... pre-training the model'
    start_time = time.clock()

    for i in xrange(len(layer_sizes)-2):

        for epoch in xrange(pretrain_epochs):

            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretrain_fns[i](index=batch_index,
                                corruption=corruption_levels[i],
                                lr=pretrain_lr))

            
            fp.write('Pre-training layer %i, epoch %d, cost %.3f\n' % (i, epoch, np.mean(c)))
            #print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            #Jprint np.mean(c)

    end_time = time.clock()

    #print >> sys.stderr, ('The pretraining code for file ' +
    #                      os.path.split(__file__)[1] +
    #                      ' ran for %.2fm' % ((end_time - start_time) / 60.))

    #print '... building the model'


    train_fn, vl_fn, valid_fn, valid_train_fn, test_fn = dnn.build_sgd_functions(
                                                datasets=train_datasets,
                                                batch_size=batch_size,
                                                dropout = dropout,
                                                learning_way=learning_way,
                                                learning_rate=learning_rate)

    epoch = 0
    best_corrcoef = 0
    best_epoch = 0

    start_time = time.clock()

    while epoch < n_epochs:
        epoch = epoch + 1
        #for minibatch_index in xrange(n_train_batches):
            # 学習
            #train_model = train_fn(minibatch_index)
            #print train_model
        tr_cost = np.asarray([train_fn(minibatch_index) for minibatch_index in xrange(n_train_batches)])
        #print tr_cost.mean()
        if epoch % 1 == 0:
            # validation
            #for valid_index in xrange(n_valid):
            #  valid_model = vl_fn(valid_index)
            vl_cost = np.asarray([vl_fn(valid_index) for valid_index in xrange(n_valid)])

            valid_preds = valid_fn()
            valid_train_preds = valid_train_fn()
            test_preds = test_fn()

            this_train_corrcoef = np.corrcoef(valid_train_preds, 
                                                    valid_train_dataset)[0][1]
            this_valid_corrcoef = np.corrcoef(valid_preds, 
                                                    valid_dataset)[0][1]
            this_test_corrcoef = np.corrcoef(test_preds, 
                                                    test_dataset)[0][1]
            fp.write('%d %.3f %.3f %.3f %.3f %.3f\n' %(epoch, tr_cost.mean(), vl_cost.mean(), this_train_corrcoef, this_valid_corrcoef, this_test_corrcoef))
            if best_corrcoef < this_valid_corrcoef:
                best_corrcoef = this_valid_corrcoef
                best_epoch = epoch
                best_test_pred = this_test_corrcoef

    end_time = time.clock()


    print 'layer', layer_sizes[-2]
    print 'lr', learning_rate
    print 'epoch: %d' %(best_epoch)
    print 'value: ', best_test_pred
    print ('calculation time %.2fm' % ((end_time - start_time) / 60.))
    print '\n'

    fp.write('p_lr: %.3f p_epochs: %d\n' %(pretrain_lr, pretrain_epochs))
    fp.write('way: %s lr: %.3f epochs: %d\n' %(learning_way, learning_rate, n_epochs))
    fp.close()

    fp = open('data_1layer_' + str(layer_sizes[-2]) + '_' + str(learning_rate) + '_' + dataset + '.res', 'w')
    fp.write(' '.join(map(str, valid_dataset)) + '\n')
    fp.write(' '.join(map(str, valid_preds)) + '\n')
    fp.close()

if __name__ == '__main__':

    argvs = sys.argv
    
    #layer = 300 * 2 ** int(argvs[2])
    layer = 1920

    test_sda(
            pretrain_lr = 0.001,
            pretrain_way = 'denoising', # denoising/contractive
            pretrain_epochs = 0,
            learning_way = 'adadelta', # normal/adagrad/adadelta
            learning_rate = 0.2 * float(argvs[2]) + 0.3,
            n_epochs = 200,
            dataset = argvs[1],
            batch_size = 4,
            layer_sizes = [1631, layer, layer, layer, layer, layer, 1],
            activations = [ReLU, ReLU, ReLU, ReLU, ReLU, ReLU], # sigmoid/tanh/ReLU
            dropout = True,
            dropout_rates = [0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            corruption_levels = [0.1, 0.1, 0.1, 0.1],
            use_bias = False,
            random_seed = 1234)
