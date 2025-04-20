#目前在深度学习领域分类两个派别，一派为学院派，研究强大、复杂的模型网络和实验方法，为了追求更高的性能；
# 另一派为工程派，旨在将算法更稳定、高效的落地在硬件平台上，效率是其追求的目标。
# 复杂的模型固然具有更好的性能，但是高额的存储空间、计算资源消耗是使其难以有效的应用在各硬件平台上的重要原因。
# 所以，卷积神经网络日益增长的深度和尺寸为深度学习在移动端的部署带来了巨大的挑战，
# 深度学习模型压缩与加速成为了学术界和工业界都重点关注的研究领域之一

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import keras
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras import regularizers
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.data_utils import get_file
import csv
import time
import json
import pickle
import codecs
from keras.models import model_from_json
from socketIO_client import SocketIO, LoggingNamespace
from fl_server import obj_to_pickle_string, pickle_string_to_obj
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,confusion_matrix,roc_curve,auc
from dataset import gen_train_valid_data
import datetime,time
print("now is {}".format(datetime.datetime.today()))
datasource= gen_train_valid_data()

import threading

class LocalModel(object):
    def __init__(self, model_config, data_collected):
        # model_config:
            # 'model': self.global_model.model.to_json(),
            # 'model_id'
            # 'min_train_size'
            # 'data_split': (0.6, 0.3, 0.1), # train, test, valid
            # 'epoch_per_round'
            # 'batch_size'
        self.model_config = model_config
        self.model = model_from_json(model_config['model_json'])#model_from_json用于从输入的 JSON 格式的模型描述创建了一个新的模型。
        # the weights will be initialized on first pull from server
        self.model.compile(loss=keras.losses.mean_squared_error,
                           optimizer=keras.optimizers.Adam())#从模型文件新创建的模型需要编译之后方可使用。
        #load data
        self.x_train, self.y_train,self.x_test,self.y_test = data_collected
        i = self.x_train.shape[1]
        print("weidu", i)
        print(self.x_test.shape)



        # self.x_train = np.array([tup[0] for tup in train_data])
        # self.y_train = np.array([tup[1] for tup in train_data])
        # self.x_test = np.array([tup[0] for tup in test_data])
        # self.y_test = np.array([tup[1] for tup in test_data])

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, new_weights):
        self.model.set_weights(new_weights)

    # return final weights, train loss, train accuracy
    def train_one_round(self,p,round):
        print('\033[1;35;0m p= \033[0m', p)

        self.model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
        batchsize=2000
        self.loss=self.model.fit(self.x_train[round*batchsize:int(round*batchsize+batchsize*p)], self.x_train[round*batchsize:int(round*batchsize+batchsize*p)],
                  epochs=self.model_config['epoch_per_round'],
                  batch_size=self.model_config['batch_size'],
                  verbose=0)
        print('one round loss',self.loss.history['loss'][0])

        # score= self.model.evaluate(self.x_train[round*2000:int(round*2000+2000*p)], self.x_train[round*2000:int(round*2000+2000*p)], verbose=0)
        # print('Train loss:', score)
        return self.model.get_weights(), self.loss.history['loss'][0]#, score[1]

    # def validate(self):
    #     score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
    #     print('Validate loss:', score[0])
    #     print('Validate accuracy:', score[1])
    #     return score

    def evaluate1(self):
        # score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        # print('Test loss:', score[0])
        # print('Test accuracy:', score[1])
        def calculate_losses(x, preds):
            losses = np.zeros(len(x))
            for i in range(len(x)):
                losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)
            return losses
        # We set the threshold equal to the training loss of the autoencoder
        threshold = self.loss.history['loss'][-1]
        testing_set_predictions = self.model.predict(self.x_test)
        test_losses = calculate_losses(self.x_test, testing_set_predictions)
        testing_set_predictions = np.zeros(len(test_losses))
        testing_set_predictions[np.where(test_losses > threshold)] = 1
        # ==============================Evaluation====================================
        precision = precision_score(self.y_test, testing_set_predictions)
        recall = recall_score(self.y_test, testing_set_predictions)
        f1 = f1_score(self.y_test, testing_set_predictions)
        print("Performance over the testing data set \n")
        print(" Recall:{}, Precision:{}, F1:{}\n".format(recall,precision,f1,'.4f'))
        return f1,precision,recall



# A federated client is a process that can go to sleep / wake up intermittently
# it learns the global model by communication with the server;
# it contributes to the global model by sending its local gradients.

class FederatedClient(object):
    MAX_DATASET_SIZE_KEPT = 1200

    def __init__(self, server_host, server_port, datasource):
        self.local_model = None
        self.datasource = datasource
        self.sio = SocketIO(server_host, server_port, LoggingNamespace)
        self.register_handles()
        print("sent wakeup")
        self.sio.emit('client_wake_up')
        self.sio.emit('------------------------------client_wake_up---------------')
        self.sio.wait()

    ########## Socket Event Handler ##########
    def on_init(self, *args):
        model_config = args[0]
        print('on init', model_config)
        print('preparing local data based on server model_config')
        # ([(Xi, Yi)], [], []) = train, test, valid
        # fake_data, my_class_distr = self.datasource.fake_non_iid_data(
        #     min_train=model_config['min_train_size'],
        #     max_train=FederatedClient.MAX_DATASET_SIZE_KEPT,
        #     data_split=model_config['data_split']
        # )
        self.local_model = LocalModel(model_config, datasource)

        # ready to be dispatched for training
        self.sio.emit('client_ready', {
                'train_size': self.local_model.x_train[:2000].shape[0],
                #'class_distr': my_class_distr  # for debugging, not needed in practice
            })


    def register_handles(self):
        ########## Socket IO messaging ##########
        def on_connect():
            print('connect')

        def on_disconnect():
            print('disconnect')

        def on_reconnect():
            print('reconnect')

        def on_request_update(*args):
            req = args[0]
            # req:
            #     'model_id'
            #     'round_number'
            #     'current_weights'
            #     'weights_format'
            #     'run_validation'
            print("update requested")
            print('round_number:', req['round_number'])

            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])

            self.local_model.set_weights(weights)
            my_weights, train_loss= self.local_model.train_one_round(req['p3'],req['round_number'])

            #print("quanzhong",my_weights[0])
            print("compress start")



            def calculate_losses(x, preds):
                losses = np.zeros(len(x))
                for i in range(len(x)):
                    losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)

                return losses



            resp = {
                'round_number': req['round_number'],
                'weights': obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train[:int(2000*req['p3'])].shape[0],
                #'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': train_loss,
                #'train_accuracy': train_f1,
            }
            # if req['run_validation']:
            #     valid_loss, valid_accuracy = self.local_model.validate()
            #     resp['valid_loss'] = valid_loss
            #     resp['valid_accuracy'] = valid_accuracy

            self.sio.emit('client_update', resp)




        def on_stop_and_eval(*args):
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            self.local_model.evaluate1()
            time_end = time.time()
            print('\033[1;35;0m Time cost = %fs \033[0m' % (time_end - time_start))
            #test_loss, test_accuracy = self.local_model.evaluate()
            resp = {
                'test_size': self.local_model.x_test.shape[0],
                #'test_loss': test_loss,
                #'test_accuracy': test_accuracy
            }
            self.sio.emit('client_eval', resp)


        self.sio.on('connect', on_connect)
        self.sio.on('disconnect', on_disconnect)
        self.sio.on('reconnect', on_reconnect)
        self.sio.on('init', lambda *args: self.on_init(*args))
        self.sio.on('request_update', on_request_update)
        self.sio.on('stop_and_eval', on_stop_and_eval)




        # TODO: later: simulate datagen for long-running train-serve service
        # i.e. the local dataset can increase while training

        # self.lock = threading.Lock()
        # def simulate_data_gen(self):
        #     num_items = random.randint(10, FederatedClient.MAX_DATASET_SIZE_KEPT * 2)
        #     for _ in range(num_items):
        #         with self.lock:
        #             # (X, Y)
        #             self.collected_data_train += [self.datasource.sample_single_non_iid()]
        #             # throw away older data if size > MAX_DATASET_SIZE_KEPT
        #             self.collected_data_train = self.collected_data_train[-FederatedClient.MAX_DATASET_SIZE_KEPT:]
        #             print(self.collected_data_train[-1][1])
        #         self.intermittently_sleep(p=.2, low=1, high=3)

        # threading.Thread(target=simulate_data_gen, args=(self,)).start()

    
    def intermittently_sleep(self, p=.1, low=10, high=100):
        if (random.random() < p):
            time.sleep(random.randint(low, high))


# possible: use a low-latency pubsub system for gradient update, and do "gossip"
# e.g. Google cloud pubsub, Amazon SNS
# https://developers.google.com/nearby/connections/overview
# https://pypi.python.org/pypi/pyp2p

# class PeerToPeerClient(FederatedClient):
#     def __init__(self):
#         super(PushBasedClient, self).__init__()    


if __name__ == "__main__":
    time_start = time.time()
    FederatedClient("127.0.0.1", 5008, datasource)
