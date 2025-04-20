import warnings
warnings.filterwarnings("ignore")
#author:chunjiong zhang
#date:2020/07/16
import os
from collections import defaultdict

import pickle#python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
import keras
import uuid#  UUID是128位的全局唯一标识符，通常由32字节的字符串表示。它可以保证时间和空间的唯一性，也称为GUID，全称为：
            #UUID —— Universally Unique IDentifier      Python 中叫 UUID
    #它通过MAC地址、时间戳、命名空间、随机数、伪随机数来保证生成ID的唯一性。
from keras.models import Sequential,Model,Input
from keras.layers import Dense, Dropout, Flatten,regularizers
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from keras import backend as K
from multi_AE import MLP_AE,GRU_AE,LSTM_AE,CNN_AE,MLP_AE1,MLP_AE2
import msgpack#信息压缩
import random
import codecs#Python中用codecs处理各种字符编码的文件
import numpy as np
import json
#import msgpack_numpy#信息编码格式
# https://github.com/lebedov/msgpack-numpy
import logging
import torch
from keras.models import load_model
from keras_compressor.compressor import compress

logging.basicConfig(
    level=logging.INFO,
)
from keras_compressor.layers import custom_layers
import sys
#from ranger import Ranger
import time

from flask import *
from flask_socketio import SocketIO#SocketIO是大名鼎鼎的实时通讯库,可以在服务器和页面之间轻松的实现双向实时通讯, 兼容性好,使用方便.多用来制作聊天/直播室, 等需要实时传输数据的地方.
from flask_socketio import *
# https://flask-socketio.readthedocs.io/en/latest/

from cosinelayer import CosineLayer
       

class GlobalModel(object):#类文档字符串
    """docstring for GlobalModel"""
    def __init__(self):#__init__()方法是一种特殊的方法，被称为类的构造函数或初始化方法，当创建了这个类的实例时就会调用该方法
        self.model = self.build_model()#self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数。
        self.current_weights = self.model.get_weights()
        # for convergence check
        self.prev_train_loss = None

        # all rounds; losses[i] = [round#, timestamp, loss]
        # round# could be None if not applicable
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.training_start_time = int(round(time.time()))#以cluster节点的时间为基准
    
    # def build_model(self):
    #     raise NotImplementedError()#raise可以实现报出错误的功能,而此时产生的问题分类是NotImplementedError。
    #
    # # client_updates = [(w, n)..]
    cosine1=[]
    def update_weights(self, client_weights, client_sizes):#全局模型参数更新（可修改）
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_sizes)

        node_weights1= np.array(client_weights[0]).reshape(1, -1)
        node_weights2 = np.array(client_weights[1]).reshape(1, -1)
        #print(node_weights1[0][0][0])
        #节点间预先相似度计算


        node1 = np.array(node_weights1[0][0][0]).reshape(1, -1)
        node2 = np.array(node_weights2[0][0][0]).reshape(1, -1)

        cosine=cosine_similarity(node1, node2)
        print(cosine)
        # self.cosine1.append(cosine)
        # np.savetxt("cosine.txt",self.cosine1)


        #这里修改权重计算方式
        for c in range(len(client_weights)):
            #print("client_weights",len(client_weights))
            #print("client_weights_shape",client_weights[c].shape)
            for i in range(len(new_weights)):
                new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
        #new_weights[1]=new_weights[0]
        self.current_weights = new_weights

    max1=[]
    min1 = []
    f1=[]
    def cosine111(self, client_weights,COMMUNICATION_ROUNDS,f1,idc,max_round):#全局模型参数更新（可修改）
        from helper import ExperimentLogger, display_train_stats
        import numpy as np
        from fl_devices import Server, Client
        import math
        server=Server
        client1=Client
        clients=client_weights

        EPS_1 =math.log(0.0008,10)
        EPS_2 =math.log(0.1,10)

        cfl_stats = ExperimentLogger()

        cluster_indices = [np.arange(len(clients)).astype("int")]
        client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        #for c_round in range(1, COMMUNICATION_ROUNDS + 1):
        c_round=int(COMMUNICATION_ROUNDS+1)
        if c_round == 1:
            client1.synchronize_with_server(self, server)

        participating_clients = server.select_clients(clients, frac=1.0)

        for client in participating_clients:
            train_stats = client.compute_weight_update(epochs=1)
            client.reset()

        similarities = server.compute_pairwise_similarities(self, clients)  # 修通

        print("cosine", similarities)

        cluster_indices_new = []

        max_norm = server.compute_max_update_norm(self, clients)
        print("zuidazhi   bianhua ma ", max_norm)
        mean_norm = server.compute_mean_update_norm(self, clients)

        if mean_norm < EPS_1 and max_norm > EPS_2 and len(idc) > 2 and c_round > 6:

            c1, c2 = server.cluster_clients(self, similarities[idc][:, idc])
            cluster_indices_new += [c1, c2]

            cfl_stats.log({"split": c_round})

        else:
            cluster_indices_new += [idc]

        cluster_indices = cluster_indices_new
        #client_clusters = [[clients[i] for i in idcs] for idcs in cluster_indices]

        server.aggregate_clusterwise(self,clients)

        acc_clients = f1#[client1.evaluate(self,client2) for client2 in clients]
        print(f1)
        self.f1.append(f1)
        self.max1.append(math.log(max_norm,10))
        print(self.max1)

        self.min1.append(math.log(mean_norm,10))
        print(self.min1)

        cfl_stats.log({"acc_clients": acc_clients, "mean_norm": mean_norm, "max_norm": max_norm,
                           "rounds": c_round, "clusters": cluster_indices})

        if c_round==max_round:
            display_train_stats(cfl_stats,self.f1,c_round, self.max1,self.min1,EPS_1, EPS_2, c_round)



        from sklearn import cluster
        from sklearn.metrics import adjusted_rand_score
        import numpy as np
        import matplotlib.pyplot as plt


        def plot_data(*data):
            X, labels_true = data
            labels = np.unique(labels_true)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            colors = 'rgbycm'
            for i, label in enumerate(labels):
                position = labels_true == label
                ax.scatter(X[position, 0], X[position, 1], label="cluster %d" % label),
                color = colors[i % len(colors)]

            ax.legend(loc="best", framealpha=0.5)
            ax.set_xlabel("X[0]")
            ax.set_ylabel("Y[1]")
            ax.set_title("data")
            plt.show()

        """
            测试函数
        """

        def test_AgglomerativeClustering(*data):
            X, labels_true = data
            clst = cluster.AgglomerativeClustering()
            predicted_labels = clst.fit_predict(X)
            print("ARI:%s" % adjusted_rand_score(labels_true, predicted_labels))

        """
            考察簇的数量对于聚类效果的影响
        """

        def test_AgglomerativeClustering_nclusters(*data):
            X, labels_true = data
            nums = range(1, 30)
            ARIS = []
            for num in nums:
                clst = cluster.AgglomerativeClustering(affinity="cosine", n_clusters=num, linkage="complete")
                predicted_lables = clst.fit_predict(-X)
                ARIS.append(adjusted_rand_score(labels_true, predicted_lables))

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(nums, ARIS, marker="+")
            ax.set_xlabel("n_clusters")
            ax.set_ylabel("ARI")
            fig.suptitle("AgglomerativeClustering")
            plt.show()

        """
            考察链接方式对聚类结果的影响
        """

        def test_agglomerativeClustering_linkage(*data):
            X, labels_true = data
            nums = range(1, 50)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            linkages = ['ward', 'complete', 'average']
            markers = "+o*"
            for i, linkage in enumerate(linkages):
                ARIs = []
                for num in nums:
                    clst = cluster.AgglomerativeClustering(n_clusters=num, linkage=linkage)
                    predicted_labels = clst.fit_predict(X)
                    ARIs.append(adjusted_rand_score(labels_true, predicted_labels))
                ax.plot(nums, ARIs, marker=markers[i], label="linkage:%s" % linkage)

            ax.set_xlabel("n_clusters")
            ax.set_ylabel("ARI")
            ax.legend(loc="best")
            fig.suptitle("AgglomerativeClustering")
            plt.show()

        from dataset import gen_train_valid_data
        self.x_train, self.y_train, self.x_test, self.y_test = gen_train_valid_data()


        # test_AgglomerativeClustering(X, labels_true)
        #plot_data(self.x_test, self.y_test)
        # test_AgglomerativeClustering_nclusters(X, labels_true)





    def aggregate_loss_accuracy(self, client_losses, client_sizes):
        total_size = np.sum(client_sizes)
        print("-----client number:-------",client_sizes)
        print('\033[1;35;0m client_sizes \033[0m', client_sizes)
        print('\033[1;35;0m client_losses \033[0m',client_losses)  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')

        # weighted sum
        aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
                for i in range(len(client_sizes)))
        from sklearn.preprocessing import MinMaxScaler

        f11=client_losses[0]
        f12= client_losses[1]
        f13 = client_losses[2]

        x1 = [[f11], [f12], [f13]]

        # norm2=Normalizer(norm='l2')
        # x1=norm2.fit_transform(x1)
        # x1=normalize(x1, norm='l1', axis=1, copy=True, return_norm=False)
        x1 = MinMaxScaler(feature_range=(0.003, 0.005), copy=True).fit_transform(x1)
        print(x1)

        f11 = x1[0]
        f12 = x1[1]
        f13 = x1[2]
        print(f11, f12, f13)

        from scipy.optimize import minimize
        import math

        x0 = np.asarray((0.33, 0.33, 0.33))

        # demo 2
        # citidu    计算  (2+x1)/(1+x2) - 3*x1+4*x3 的最小值  x1,x2,x3的范围都在0.1到0.9 之间
        def fun(args):
            l11, l12, l13 = args
            lambda1 = 0
            for n in range(20):
                v = lambda x0: x0[0] * l11 + x0[1] * l12 + x0[2] * l13 + lambda1 * (
                            x0[0] + x0[1] + x0[2] - 1);
                rho = math.pow(0.8, n)
                # print(n)
                # print("xdezhi",x0[0])
                lambda1 = lambda1 + rho * (x0[0] + x0[1] + x0[2] - 1)
                # print(lambda1)
                # print(tf.cast(v,dtype=tf.float32))
            return v

        def con(args1):
            # 约束条件 分为eq 和ineq
            # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
            x1min, x1max, x2min, x2max, x3min, x3max= args1
            cons = ({'type': 'ineq', 'fun': lambda x0: x0[0] - x1min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[0] + x1max},
                    {'type': 'ineq', 'fun': lambda x0: x0[1] - x2min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[1] + x2max},
                    {'type': 'ineq', 'fun': lambda x0: x0[2] - x3min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[2] + x3max},
                    #{'type': 'ineq', 'fun': lambda x0: x0[3] - x4min},
                    # {'type': 'ineq', 'fun': lambda x0: -x0[3] + x4max},
                    {'type': 'eq', 'fun': lambda x0: x0[0] + x0[1] + x0[2] - 1})
            return cons

        # if __name__ == "__main__":
        # 定义常量值
        args = (f11, f12, f13)  # a,b,c,d
        # 设置参数范围/约束条件
        args1 = (0.18, 0.99, 0.015, 0.99, 0.0152, 0.99)  # x1min, x1max, x2min, x2max
        # cons = con(args1)
        # 设置初始猜测值
        res = minimize(fun(args), x0, method='SLSQP', constraints=con(args1))  #
        print("f1 min", res.fun)
        print(res.success)
        # print(res.x)
        p1, p2, p3= res.x
        print("p1, p2, p3", p2, p1, p3)

        #这个位置修改loss 函数

        # aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
        #         for i in range(len(client_sizes)))
        return aggr_loss,res.x

    # cur_round coule be None    , cur_round
    def aggregate_train_loss_accuracy(self, client_losses,client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, p = self.aggregate_loss_accuracy(client_losses, client_sizes)
        self.train_losses += [[cur_round,cur_time, aggr_loss]]
        #self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        return aggr_loss,p

    # cur_round coule be None
    # def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
    #     cur_time = int(round(time.time())) - self.training_start_time
    #     aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
    #     self.valid_losses += [[cur_round, cur_time, aggr_loss]]
    #     self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
    #     with open('stats.txt', 'w') as outfile:
    #         json.dump(self.get_stats(), outfile)
    #     return aggr_loss, aggr_accuraries

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies
        }
        
import keras.backend as K
from keras.layers.core import Lambda
class GlobalModel_KDD_AE(GlobalModel):

    def cosine(self,x1, x2):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        output_shape = (1,)
        value = Lambda(_cosine, output_shape=output_shape)([x1, x2])
        return value

    def relative_euclid(self,x, x_rec):
        def _cosine(x):
            min_val = 1e-3

            def euclid_norm(x):
                return K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))  # tf.reduce_sum

            x_l2 = euclid_norm(x[0])
            res_l2 = euclid_norm(x[0] - x[1])
            return res_l2 / (x_l2 + min_val)

        output_shape = (1,)
        relative_euclid = Lambda(_cosine, output_shape=output_shape)([x, x_rec])  #
        return relative_euclid
    def __init__(self):
        super(GlobalModel_KDD_AE, self).__init__()
        #其中的super类的作用是继承的时候，
        # 调用含super的各个的基类__init__函数，
        # 如果不使用super，就不会调用这些类的__init__函数，
        # 除非显式声明。而且使用super可以避免基类被重复调用

    def build_model(self):
        # ~35MB worth of parameters
        #怎么获得模型大小
        # input数据接口
        input_img = Input(shape=(118,))
        # # 分支0
        x = Dense(64, activation='relu', kernel_initializer='random_uniform',
                  activity_regularizer=regularizers.l2(10e-5),
                  name='encoded1')(input_img)
        # x=Dropout(0.5)(x)
        x = Dense(32, activation='relu', kernel_initializer='random_uniform', name='net0_encoded2')(x)
        x = Dense(16, activation='relu', kernel_initializer='random_uniform', name='net0_encoded3')(x)
        z0 = Dense(1, kernel_initializer='random_uniform', name='net_encodedz')(x)
        x = Dense(16, activation='relu', kernel_initializer='random_uniform', name='net0_decoded1')(x)
        x = Dense(32, activation='relu', kernel_initializer='random_uniform', name='net0_decoded2')(x)
        x = Dense(64, activation='relu', kernel_initializer='random_uniform', name='net0_decoded3')(x)
        x = Dense(118, activation=None, kernel_initializer='random_uniform', name='net0_decoded4')(x)
        #
        # #cosine0 = self.(input_img, x)
        # relative_euclid0 = self.relative_euclid(input_img, x)
        # z0 = keras.layers.concatenate([cosine0, relative_euclid0, z0], axis=1)

        # # 分支1
        tower1 = Dense(118, activation='relu', kernel_initializer='random_uniform', name='net1_encoded0')(input_img)
        # tower1 = Dropout(0.5)(tower1)
        tower1 = Dense(112, activation='relu', kernel_initializer='random_uniform', name='net1_encoded1')(tower1)
        tower1 = Dense(81, activation='relu', kernel_initializer='random_uniform', name='net1_encoded2')(tower1)
        tower1 = Dense(55, activation='relu', kernel_initializer='random_uniform', name='net1_encoded3')(tower1)
        tower1 = Dense(28, activation='relu', kernel_initializer='random_uniform', name='net1_encoded4')(tower1)
        z1 = Dense(1, kernel_initializer='random_uniform', name='net1_encodedz')(tower1)
        tower1 = Dense(28, activation='relu', kernel_initializer='random_uniform', name='net1_decoded0')(tower1)
        tower1 = Dense(55, activation='relu', kernel_initializer='random_uniform', name='net1_decoded1')(tower1)
        tower1 = Dense(81, activation='relu', kernel_initializer='random_uniform', name='net1_decoded2')(tower1)
        tower1 = Dense(112, activation='relu', kernel_initializer='random_uniform', name='net1_decoded3')(tower1)
        tower1 = Dense(118, activation=None, kernel_initializer='random_uniform', name='net1_decoded4')(tower1)

        # cosine1 = CosineLayer(input_img, tower1)
        # relative_euclid1 = self.relative_euclid(input_img, tower1)
        # z1 = keras.layers.concatenate([cosine1, relative_euclid1, z1], axis=1)

        # # 分支2
        tower2 = Dense(118, activation='relu', kernel_initializer='random_uniform', name='net2_encoded0')(input_img)
        # tower2 = Dropout(0.5)(tower2)
        tower2 = Dense(60, activation='relu', kernel_initializer='random_uniform', name='net2_encoded1')(tower2)
        tower2 = Dense(30, activation='relu', kernel_initializer='random_uniform', name='net2_encoded2')(tower2)
        tower2 = Dense(10, activation='relu', kernel_initializer='random_uniform', name='net2_encoded3')(tower2)
        z2 = Dense(1, kernel_initializer='random_uniform', name='net2_encoded4')(tower2)
        tower2 = Dense(10, activation='relu', kernel_initializer='random_uniform', name='net2_decoded0')(tower2)
        tower2 = Dense(30, activation='relu', kernel_initializer='random_uniform', name='net2_decoded1')(tower2)
        tower2 = Dense(60, activation='relu', kernel_initializer='random_uniform', name='net2_decoded2')(tower2)
        tower2 = Dense(118, activation=None, kernel_initializer='random_uniform', name='net2_decoded3')(tower2)

        # hidden2 = [118, 60, 30, 1, 30, 60, 118]
        # tower2 = input_img
        # n_layer = 0
        # for nums in hidden2:
        #     tower2 = Dense(nums, activation='elu' if nums != 1 else None, name="net2_{}".format(n_layer))(tower2)
        #     #tower2=LeakyReLU(alpha=0.2)(tower2)
        #     #tower2 = Dropout(0.5)(tower2)
        #     n_layer += 1
        #     if nums == 1:
        #         z2 = tower2

        # cosine2 = CosineLayer(input_img, tower2)
        # relative_euclid2 = self.relative_euclid(input_img, tower2)
        # z2 = keras.layers.concatenate([cosine2, relative_euclid2, z2], axis=1)

        # 拼接output
        loss0 = keras.losses.mean_squared_error(y_pred=input_img, y_true=x)
        loss1 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower1)
        loss2 = keras.losses.mean_squared_error(y_pred=input_img, y_true=tower2)
        # seclect minimum loss
        recons22 = keras.backend.maximum(loss0, loss1)
        loss = keras.backend.maximum(recons22, loss2)
        # loss最小的重建误差
        if loss0 == loss:
            tower = x
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
        model.summary()
        for layer in model.layers:
            print(layer.name)


        #model = load_model('demo/model_raw.h5')
        # 编译model
        #adam = keras.optimizers.Adam(lr=0.0005, beta_1=0.95, beta_2=0.999, epsilon=1e-08)
        adam = keras.optimizers.Adam(lr = 0.001, beta_1=0.95, beta_2=0.999,epsilon=1e-08)
        # sgd = keras.optimizers.SGD(lr = 0.001, decay = 1e-06, momentum = 0.9, nesterov = False)
        # reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.1, patience = 2,verbose = 1, min_lr = 0.00000001, mode = 'min')
        model.compile(loss=keras.losses.mean_squared_error, optimizer=adam, metrics=['accuracy'])
        #model.save("global_m.h5")

        #model = load_model('demo/model_raw.h5', custom_layers)
        #model = load_model('demo/model_compress.h5', custom_layers)
        #model = load_model('demo/model_compress_finetuned.h5',custom_layers)

        for layer in model.layers:
            print(layer.name)
        #模型压缩
        #model = compress(model, 7e-1)

        return model


######## Flask server with Socket IO ########

# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):

    
    MIN_NUM_WORKERS = 3
    MAx_NUM_ROUNDS = 15#设定联邦循环次数
    NUM_CLIENTS_CONTACTED_PER_ROUND = 3#设置节点数量，作用，多少比例的掉队。
    ROUNDS_BETWEEN_VALIDATIONS = 2

    def __init__(self, global_model, host, port):
        self.global_model = global_model()


        self.ready_client_sids = set()#???

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port
        #将 Flask-SocketIO 添加到 Flask 应用程序

        self.model_id = str(uuid.uuid4())#uuid4()——基于随机数,由伪随机数得到，有一定的重复概率，该概率可以计算出来。

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        #####

        # socket io messages
        self.register_handles()


        @self.app.route('/')
        def dashboard():
            """测试页面"""
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())

    def register_handles(self):#Flask-SocketIO 调度连接和断开事件。以下示例显示如何为它们注册处理程序
        # single-threaded async, no need to lock

        @self.socketio.on('connect')# """客户端连接"""
        def handle_connect():
            print(request.sid, "connected")# # request.sid,,,io客户端的sid, socketio用此唯一标识客户端.
            print('\033[1;35;0m connected \033[0m')


        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_reconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            #print(self.global_model.model.to_json())
            # site = self.global_model.model.to_json()
            # site= site.replace('class_name','')
            # site = site.replace('FactorizedDense', '')# 删除要删除的键值对，如{'name':'我的博客地址'}这个键值对

            emit('init', {
                    'model_json': self.global_model.model.to_json(),
                    'model_id': self.model_id,

                    #'data_split': (0.6, 0.3, 0.1), # train, test, valid
                    'epoch_per_round': 1,
                    'batch_size': 32
                })#定义的 SocketIO 事件处理程序可以使用send()和emit() 函数将回复消息发送到连接的客户端。

        @self.socketio.on('client_ready')
        def handle_client_ready(data):#大于设置最少的节点个数时开始全局训练
            print("client ready for training")#request.sid, data
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= FLServer.MIN_NUM_WORKERS and self.current_round == -1:
                self.train_next_round(None)

        @self.socketio.on('client_update')
        def handle_client_update(data):
            #接收多少字节
            print("received client update of bytes: ", sys.getsizeof(data))
            print("handle client_update", request.sid)
            print('\033[1;35;0m handle client_update \033[0m')
            # for x in data:
            #     if x != 'weights':
            #         #print(x, data[x])

            # data:
            #   weights
            #   train_size
            #   valid_size
            #   train_loss
            #   train_accuracy
            #   valid_loss?
            #   valid_accuracy?


            # discard outdated update
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
               # print("shangchuanneirong",self.current_round_client_updates)

                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                
                # tolerate 30% unresponsive clients
                if len(self.current_round_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )

                    aggr_train_loss, p = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        #[x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )
                    print("aggr_train_loss", aggr_train_loss)
                    #print("aggr_train_accuracy", aggr_train_accuracy)


                    self.global_model.cosine111(
                        [x['weights'] for x in self.current_round_client_updates], self.current_round,
                        data['train_accuracy'],self.MIN_NUM_WORKERS,self.MAx_NUM_ROUNDS
                    )

                    # emit('p', {
                    #     'p1': p[0],
                    #     'p2': p[1],
                    #     'p3': p[2],}
                    #      )这里不好传送


                    # if self.global_model.prev_train_loss is not None and \
                    #         (self.global_model.prev_train_loss - aggr_train_loss) / self.global_model.prev_train_loss < .1:#我修改了
                    #     # converges
                    #     print("converges! starting test phase..")#判断收敛性
                    #     print('\033[1;35;0m converges! starting test phase.. \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
                    #     #self.stop_and_eval()
                    #     return
                    
                    self.global_model.prev_train_loss = aggr_train_loss

                    if self.current_round >= FLServer.MAx_NUM_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round(p)



        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            print("handle client_eval", request.sid)
            #print("eval_resp", data)
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            if len(self.eval_client_updates) > FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                # aggr_test_loss, aggr_test_accuracy = self.global_model.aggregate_loss_accuracy(
                #     #[x['test_loss'] for x in self.eval_client_updates],
                #     #[x['test_accuracy'] for x in self.eval_client_updates],
                #     [x['test_size'] for x in self.eval_client_updates],
                # );
                #print("\naggr_test_loss", aggr_test_loss)
                #print("aggr_test_accuracy", aggr_test_accuracy)
                print('\033[1;35;0m == done == \033[0m')  # 有高亮 或者 print('\033[1;35m字体有色，但无背景色 \033[0m')
                time_end = time.time()
                print('totally cost', time_end - time_start)

                self.eval_client_updates = None  # special value, forbid evaling again

    
    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self,p):
        if p is None:
            p=[1,1,1]#防止在handle_client_ready(data) 中的self.train_next_round(None)出现NONE
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        print("### Round ", self.current_round, "###")
        client_sids_selected = random.sample(list(self.ready_client_sids), FLServer.NUM_CLIENTS_CONTACTED_PER_ROUND)#为了提取出N个不同元素的样本用来(所有内容，需要的数量)
        print("request updates from", client_sids_selected)

        # by default each client cnn is in its own "room"
        for rid in client_sids_selected:
            emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'p1': p[0],
                    'p2': p[1],
                    'p3': p[2],
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),

                    'weights_format': 'pickle',
                    'run_validation': self.current_round % FLServer.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)

    def stop_and_eval(self):
        #self.global_model.save("global_model.h5")
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                    'model_id': self.model_id,
                    'current_weights': obj_to_pickle_string(self.global_model.current_weights),
                    'weights_format': 'pickle'
                }, room=rid)
    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)



def obj_to_pickle_string(x):
    return codecs.encode(pickle.dumps(x), "base64").decode()
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO

def pickle_string_to_obj(s):
    return pickle.loads(codecs.decode(s.encode(), "base64"))
    # return msgpack.unpackb(s, object_hook=msgpack_numpy.decode)


if __name__ == '__main__':
    # When the application is in debug mode the Werkzeug development server is still used
    # and configured properly inside socketio.run(). In production mode the eventlet web server
    # is used if available, else the gevent web server is used.
    time_start = time.time()
    server = FLServer(GlobalModel_KDD_AE, "127.0.0.1", 5011)
    print("listening on 127.0.0.1:5011");
    server.start()
