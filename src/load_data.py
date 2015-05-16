# -*- coding:utf-8 -*-
import cPickle
import gzip
import os

import numpy as np

import theano
import theano.tensor as T


############################################
## **.pkl.gz形式のファイルを読み込んで    ##
## numpyのndarrayとtheanoのshared variable##
## 形式で保持するクラス                   ##
############################################

class load_data(object):

    # datasetは読み込むのデータセット
    # ex) 'test.pkl.gz'
    def __init__(self, dataset):

        # 入力のdatasetをpathとfile名に分割
        data_dir, data_file = os.path.split(dataset)

        # file名のみだったらdata_dirは空になるはず
        if data_dir == "" and not os.path.isfile(dataset):

            # print os.path.dirname(__file__) __file__は相対path?
            # ../datasets/dataset
            new_path = os.path.join(
                    os.path.split(__file__)[0], "..", "datasets", dataset)

            # ('../datasets/*.gz'が存在していたらそこをpathとする)
            if os.path.isfile(new_path):
                dataset = new_path

        #print ' ... loading data'

        # 上で設定したpathのファイルを開く(ex) test.pkl.gz
        f = gzip.open(dataset, 'rb')

        # .pklの中身は(train_set, valid_set, test_set)である
        # train_set, valid_set, test_setはそれぞれtuple(input, target)
        # inputは(入力の次元数 * データ数)のnumpy.ndarrayの行列
        # targetはデータ数のnumpy.ndarrayのベクトル
        self.train_set, self.valid_set, self.test_set = cPickle.load(f)

        f.close()

    # borrowはなにかよくわからない 参照関連?
    def shared_dataset(self, dataset, borrow=True):

        # datasetをshared variableに格納
        # shared varibalに格納することによってGPU memoryに送れる
        data_x, data_y = dataset

        # GPUに格納するときはfloatにする必要がある
        # shared(データ型、borrow?)
        shared_x = theano.shared(np.asarray(data_x,
                                dtype = theano.config.floatX), borrow = borrow)
        shared_y = theano.shared(np.asarray(data_y, 
                                dtype = theano.config.floatX), borrow = borrow)

        
        # 分類ではなく回帰問題なので、floatのままreturnする
        return shared_x, shared_y

    def get_shared_dataset(self):
        # trainとvalidとtestをそれぞれshared variableに格納
        train_set_x, train_set_y = self.shared_dataset(self.train_set)
        valid_set_x, valid_set_y = self.shared_dataset(self.valid_set)
        test_set_x, test_set_y = self.shared_dataset(self.test_set)


        # rvalにそれぞれのデータを格納してreturn
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval
    
    def get_ndarray_dataset(self):
        # trainとvalidとtestをそれぞれndarrayに格納
        train_set_x, train_set_y = self.train_set
        valid_set_x, valid_set_y = self.valid_set
        test_set_x, test_set_y = self.test_set


        # rvalにそれぞれのデータを格納してreturn
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
                (test_set_x, test_set_y)]
        return rval

if __name__ == '__main__':


    d = load_data('test2.pkl.gz')

    datasets = d.get_shared_dataset()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print type(train_set_x), type(train_set_y)
    print type(valid_set_x), type(valid_set_y)
    print type(test_set_x), type(test_set_y)

    datasets = d.get_ndarray_dataset()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print type(train_set_x), type(train_set_y)
    print type(valid_set_x), type(valid_set_y)
    print type(test_set_x), type(test_set_y)
