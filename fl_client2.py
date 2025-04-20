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

            def save_weights_histogram(w, stage, iter):
                w = w.reshape(-1)
                valid_w = [x for x in w if (x != 0.0)]
                plt.grid(True)
                plt.hist(valid_w, 100, color='blue')
                plt.gca().set_xlim([-3.5, 3.5])
                plt.gca().set_ylim([0, 15])
                plt.xlabel('Weight')
                plt.ylabel('Count')
                plt.savefig('./图片/' + stage + str(iter), dpi=100)
                plt.gcf().clear()

            def open_result_csv(result_file):
                read_path = os.path.join(result_file, '迭代修剪剪枝策略结果.csv')
                result_col = ['satge', 'accuracy', 'recall', 'precision', 'f1', 'true_positive_rate',
                              'false_positve_rate',
                              'AUC',
                              'N_accuracy', 'N_recall', 'N_precision', 'N_f1',
                              'D_accuracy', 'D_recall', 'D_precision', 'D_f1',
                              'R_accuracy', 'R_recall', 'R_precision', 'R_f1',
                              'U_accuracy', 'U_recall', 'U_precision', 'U_f1',
                              'P_accuracy', 'P_recall', 'P_precision', 'P_f1']
                result_pd = pd.DataFrame(columns=result_col)
                result_pd.to_csv(read_path)
                csv_f = open(read_path, "a+", newline='')
                csv_write = csv.writer(csv_f)
                return csv_f, csv_write

            def save_ACCs(stage, resultList, class_resultList):
                temp_list = ['', stage]
                temp_list.extend(resultList)
                for i in range(len(class_resultList)):
                    temp_list.extend(class_resultList[i])
                csv_write.writerow(temp_list)

            def calculate_losses(x, preds):
                losses = np.zeros(len(x))
                for i in range(len(x)):
                    losses[i] = ((preds[i] - x[i]) ** 2).mean(axis=None)

                return losses

            def Acuracy(y_test, testing_set_predictions):
                matrix = confusion_matrix(y_test, testing_set_predictions)
                TP = matrix[0, 0]
                FN = matrix[0, 1]
                FP = matrix[1, 0]
                TN = matrix[1, 1]
                accuracy = accuracy_score(y_test, testing_set_predictions)
                precision = precision_score(y_test, testing_set_predictions)
                recall = recall_score(y_test, testing_set_predictions)
                f1 = f1_score(y_test, testing_set_predictions)
                true_positive_rate = TP * 1.0 / (TP + FN)
                false_positve_rate = FP * 1.0 / (FP + TN)
                fpr, tpr, thresholds = roc_curve(y_test, testing_set_predictions)
                AUC = auc(fpr, tpr)
                resultList = [accuracy, recall, precision, f1, true_positive_rate, false_positve_rate, AUC]

                class_resultList = []
                classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]
                for class_ in classes:
                    class_y = y_test[np.where(test_label == class_)]
                    class_pred = testing_set_predictions[np.where(test_label == class_)]
                    c_accuracy = accuracy_score(class_y, class_pred)
                    c_recall = recall_score(class_y, class_pred)
                    c_precision = precision_score(class_y, class_pred)
                    c_f1 = f1_score(class_y, class_pred)
                    temp_list = [c_accuracy, c_recall, c_precision, c_f1]
                    class_resultList.append(temp_list)

                return resultList, class_resultList

            # 读入数据
            x_train, y_train, x_test, y_test = self.datasource
            test_label = y_test
            # 读入AE-keras的w和b、阈值
            AE_model = my_weights

            print("load weight")
            #print(AE_model)
            encoder_w = AE_model[0]
            print("encoder_w", encoder_w.shape)
            encoder_b = AE_model[1]
            encoder2_w = AE_model[2]
            print("encoder2_w", encoder2_w.shape)
            encoder2_b = AE_model[3]
            print("encoder2_b", encoder2_b.shape)
            # decoder_w = AE_model[4]
            # print("decoder_w", decoder_w.shape)
            # decoder_b = AE_model[5]
            # decoder2_w = AE_model[6]
            # decoder2_b = AE_model[7]
            threshold = 0.009041859193491928

            # 打开存储结果的文件
            result_file = '.\\'
            csv_f, csv_write = open_result_csv(result_file)

            # 重构模型-Keras==========================================
            encoded_mask_init = np.ones(944)
            decoded_mask_init = np.ones(944)
            mask_list_init = [encoded_mask_init]
            init_drop_p = 0.5

            def getModel1(mask_list, drop_p):
                input_img = Input(shape=(118,))
                # # 分支0
                hidden = [64, 32, 16, 1, 16, 32, 64, 118]
                tower0 = input_img
                for nums in hidden:
                    # tower0 = Dense(nums, activation=LeakyReLU(alpha=0.2) if nums != 1 else None)(tower0)
                    tower0 = Dense(nums, activation=None if nums != 1 else None)(tower0)
                    tower0 = LeakyReLU(alpha=0.2)(tower0)
                    tower0 = Dropout(drop_p)(tower0)
                    if nums == 1:
                        z0 = tower0
                # def extract_rec_features(x, x_rec):
                #     x_l2 = keras.backend.l2_normalize(x)
                #     x_rec_l2 = keras.backend.l2_normalize(x_rec)
                #     res_l2 = keras.backend.l2_normalize(x - x_rec)
                #     relative_euclid = res_l2 / x_l2
                #     cos_similarity = keras.backend.sum(x * x_rec, axis=1, keepdims=True) / (x_l2 * x_rec_l2)
                #     return relative_euclid, cos_similarity
                #
                # z_r_seq = extract_rec_features(input_img, tower0)
                # z0 = keras.backend.concatenate([*z_r_seq, z0], axis=1)

                # # 分支1
                hidden1 = [118, 81, 55, 28, 1, 28, 55, 81, 118]
                tower1 = input_img
                for nums in hidden1:
                    # tower1 = Dense(nums, activation=LeakyReLU(alpha=0.2)if nums != 1 else None)(tower1)
                    tower1 = Dense(nums, activation=None if nums != 1 else None)(tower1)
                    tower1 = LeakyReLU(alpha=0.2)(tower1)
                    tower1 = Dropout(drop_p)(tower1)
                    if nums == 1:
                        z1 = tower1

                # # 分支2
                hidden2 = [118, 60, 30, 1, 30, 60, 118]
                tower2 = input_img
                for nums in hidden2:
                    tower2 = Dense(nums, activation=None if nums != 1 else None)(tower2)
                    tower2 = LeakyReLU(alpha=0.2)(tower2)
                    tower2 = Dropout(drop_p)(tower2)
                    if nums == 1:
                        z2 = tower2
                # 拼接output
                loss0 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower0)
                loss1 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower1)
                loss2 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower2)
                # seclect minimum loss
                recons22 = keras.backend.maximum(loss0, loss1)
                loss = keras.backend.maximum(recons22, loss2)
                # loss最小的重建误差
                if loss0 == loss:
                    tower = tower0
                elif loss1 == loss:
                    tower = tower1
                else:
                    tower = tower2
                output = keras.layers.concatenate([z0, z1, z2, tower], axis=1)
                # # GAN
                output1 = Dense(8, activation='relu', kernel_initializer='random_uniform',
                                activity_regularizer=regularizers.l2(10e-5), name='gan1')(output)
                output1 = Dense(118, activation='relu', kernel_initializer='random_uniform',
                                activity_regularizer=regularizers.l2(10e-5), name='gan2')(output1)
                # output1 = Dense(64, activation='relu', kernel_initializer='random_uniform',name='encoded7')(output1)

                # 把前面的计算逻辑，分别指定input和output，并构建成网络
                model = Model(inputs=input_img, outputs=output1)
                # 编译model
                #adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
                adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
                # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
                # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
                model.compile(loss=keras.losses.mean_squared_error, optimizer=adam,metrics=['accuracy'])
                return model

            autoencoder = getModel1(mask_list_init, init_drop_p)
            autoencoder.get_layer(index=1).set_weights([encoder_w, encoder_b])
            #autoencoder.get_layer(index=3).set_weights([encoder2_w, encoder2_b])
            # autoencoder.get_layer('decoded1').set_weights([decoder_w, decoder_b])
            # autoencoder.get_layer('decoded2').set_weights([decoder2_w, decoder2_b])

            # 预测
            testing_set_predictions = autoencoder.predict(x_test)
            test_losses = calculate_losses(x_test, testing_set_predictions)
            testing_set_predictions = np.zeros(len(test_losses))
            testing_set_predictions[np.where(test_losses > threshold)] = 1

            # 检测
            resultList, class_resultList = Acuracy(y_test, testing_set_predictions)
            save_ACCs('input', resultList, class_resultList)

            # 剪枝============================================================================
            def prune_weights(w_data, w_shape, threshold):
                hi = np.max(np.abs(w_data.flatten()))
                hi = np.sort(-np.abs(w_data.flatten()))[int((len(w_data.flatten()) - 1) * threshold)]
                pruning_mask_data = (np.abs(w_data) > (np.abs(hi))).astype(np.float32)
                return pruning_mask_data, (pruning_mask_data * w_data).reshape(w_shape)

            def prune(prune_th, iter, encoder_w, encoder_b, threshold,drop_p):
                # 剪枝--更新权重
                encoded_mask, encoder_w = prune_weights(encoder_w, encoder_w.shape, prune_th)
                # encoded1_mask, encoder2_w = prune_weights(encoder2_w, encoder2_w.shape, prune_th)
                # decoded_mask, decoder_w = prune_weights(decoder_w, decoder_w.shape, prune_th)
                # decoded1_mask, decoder2_w = prune_weights(decoder2_w, decoder2_w.shape, prune_th)

                save_weights_histogram(encoder_w, 'pruned', iter)
                # save_weights_histogram(encoder2_w, 'pruned', iter)
                # 重建模型
                autoencoder = getModel1([encoded_mask.reshape(-1)], drop_p)

                # 重新赋值
                autoencoder.get_layer(index=1).set_weights([encoder_w, encoder_b])
                # autoencoder.get_layer('encoded2').set_weights([encoder2_w, encoder2_b])
                # autoencoder.get_layer('decoded1').set_weights([decoder_w, decoder_b])
                # autoencoder.get_layer('decoded2').set_weights([decoder2_w, decoder2_b])

                # 预测
                testing_set_predictions = autoencoder.predict(x_test)
                test_losses = calculate_losses(x_test, testing_set_predictions)
                testing_set_predictions = np.zeros(len(test_losses))
                testing_set_predictions[np.where(test_losses > threshold)] = 1

                # 检测
                resultList, class_resultList = Acuracy(y_test, testing_set_predictions)
                save_ACCs(str(prune_th) + 'after pruned', resultList, class_resultList)

                return encoded_mask, encoder_w

            def retrain(encoded_mask,encoder_w, encoder_b, drop_p):
                # 重构模型
                autoencoder = getModel1([encoded_mask.reshape(-1)], drop_p)

                # 重新赋值
                autoencoder.get_layer(index=1).set_weights([encoder_w, encoder_b])
                # autoencoder.get_layer('encoded2').set_weights([encoder2_w, encoder2_b])
                # autoencoder.get_layer('decoded1').set_weights([decoder_w, decoder_b])
                # autoencoder.get_layer('decoded2').set_weights([decoder2_w, decoder2_b])

                # 重训练
                history = autoencoder.fit(x_train, x_train, epochs=1, batch_size=100, shuffle=True,
                                          validation_split=0.1)
                threshold = history.history["loss"][-1]

                                testing_set_predictions = autoencoder.predict(x_test)
                test_losses = calculate_losses(x_test, testing_set_predictions)
                testing_set_predictions = np.zeros(len(test_losses))
                testing_set_predictions[np.where(test_losses > threshold)] = 1

                # 再检测
                testing_set_predictions = autoencoder.predict(x_test)
                test_losses = calculate_losses(x_test, testing_set_predictions)
                testing_set_predictions = np.zeros(len(test_losses))
                testing_set_predictions[np.where(test_losses > threshold)] = 1

                resultList, class_resultList = Acuracy(y_test, testing_set_predictions)
                save_ACCs(str(prune_th) + 'after retrained', resultList, class_resultList)

                # 获得权重并返回
                encoder_w, encoder_b = autoencoder.get_layer(index=1).get_weights()
                # encoder2_w, encoder2_b = autoencoder.get_layer('encoded2').get_weights()
                # decoder_w, decoder_b = autoencoder.get_layer('decoded1').get_weights()
                # decoder2_w, decoder2_b = autoencoder.get_layer('decoded2').get_weights()

                return encoder_w, encoder_b, threshold

            # 主程序 边剪枝边调整剪枝率，边重训练
            old_drop_p = init_drop_p
            for i in range(0, 100, 80):
                if (i == 0):
                    continue
                prune_th = (100 - i) / 100
                encoded_mask, encoder_w = prune(prune_th, 100 - i, encoder_w, encoder_b, threshold,old_drop_p)
                #encoded_mask, encoder_w = prune(prune_th, 100 - i, encoder_w, encoder_b, threshold, old_drop_p)
                new_drop_p = init_drop_p * (prune_th ** 0.5)
                old_drop_p = new_drop_p
                encoder_w, encoder_b, threshold = retrain(encoded_mask, encoder_w, encoder_b, new_drop_p)

             #c重新赋值本地权重，后上传到服务端
            my_weights[0]=encoder_w
            my_weights[1] = encoder_b

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
