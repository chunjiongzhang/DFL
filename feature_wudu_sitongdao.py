from multi_AE import MLP_AE,MLP_AE1,GRU_AE,LSTM_AE,CNN_AE,MLP_AE4,MLP_AE2

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import itertools
import matplotlib.pyplot as plt
import numpy as np
import time
import numpy
import pandas as pd
from tensorflow.python.ops import resources
#from keras.layers import Input,Dropout,Dense,Convolution1D,LSTM,MaxPooling1D,AlphaDropout,Flatten
#from keras import regularizers
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
#from keras_radam import RAdam
#from dataset_multitest import gen_train_valid_data
#from gmm import GMM

from dataset import gen_train_valid_data

import datetime
print("now is {}".format(datetime.datetime.today()))

batch_size=1
training_set,X_test,y_train,y_test= gen_train_valid_data()

dataset_size=training_set.shape[0]
dataset_size1=X_test.shape[0]
X=tf.placeholder(shape=(None,training_set.shape[1]),dtype=tf.float32,name='X')


i=training_set.shape[1]
print("weidu",i)

zc = MLP_AE.encoder(X[:,0:28], 14)
recons = MLP_AE.decoder(zc, 28)

zc1 = MLP_AE.encoder(X[:,29:57], 14)
recons1 = MLP_AE.decoder(zc1, 28)

zc2 = MLP_AE2.encoder(X[:,58:85], 14)
recons2 = MLP_AE2.decoder(zc2, 27)


zc3 = MLP_AE.encoder(X[:,86:i], 16)
recons3 = MLP_AE.decoder(zc2, 32)

#集成AE，对每两个维度特征训练

saver=tf.concat([recons,recons1],1)
saver1=tf.concat([recons2,recons3],1)
saver2=tf.concat([saver,saver1],1)
#将每个AE训练后的tensor拼接起来，用于输入最后一个所有维度的AE
print(saver2.shape)

#最后总的AE
zc = MLP_AE4.encoder(saver2,i)
recons33 = MLP_AE4.decoder(zc,i)
# print(recons)




loss = tf.losses.mean_squared_error(labels=X, predictions=recons33)
# print(loss)
# loss1 = tf.losses.mean_squared_error(labels=X, predictions=recons1)
# print(loss1)
# loss2 = tf.losses.mean_squared_error(labels=X, predictions=recons2)
# print(loss2)
# loss=tf.cast(loss, dtype=float, name=None)
# loss1=tf.cast(loss1, dtype=float, name=None)
# loss2=tf.cast(loss2, dtype=float, name=None)
#
# #min(loss,loss1,loss2)
# if loss<loss1:
#     if loss<loss2:
#         train_op = tf.train.AdamOptimizer(0.007).minimize(loss)
# if loss1<loss:
#     if loss1<loss2:
#         train_op = tf.train.AdamOptimizer(0.007).minimize(loss1)
# if loss2<loss1:
#     if loss2<loss:
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
# saver=tf.train.Saver()
# tf.add_to_collection("predict", recons33)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_plot = []
    for k in range(int(1*int(dataset_size / batch_size))-20000):
        if k >dataset_size-1:
            for k in range(int(1 * int(dataset_size / batch_size))):
                start = k * batch_size
                end = start + batch_size
                loss_val1, en, loss_val = sess.run([recons33, train_op, loss],
                                                   feed_dict={X: training_set[start:end, :]})
                loss_plot.append(loss_val)

                print(k, "di yi ceng Loss: ", loss_val)
                if k > dataset_size-1:
                    for k in range(int(1 * int(dataset_size / batch_size))):
                        start = k * batch_size
                        end = start + batch_size
                        loss_val1, en, loss_val = sess.run([recons33, train_op, loss],
                                                           feed_dict={X: training_set[start:end, :]})
                        loss_plot.append(loss_val)
                        print(k, "zuihou yiceng Loss: ", loss_val)
        start = k * batch_size
        end = start + batch_size
        loss_val1,en, loss_val = sess.run([recons33,train_op,loss],feed_dict={X: training_set[start:end, :]})
        loss_plot.append(loss_val)
        print(k,"Loss: ", loss_val)
        # saver.save(sess, "Model/model.ckpt", global_step=3)
        # print("save the model")
    print("***************************************************************")
    print("Autoencoder_fullly Performance over the testing data set")
    threshold = loss_plot[-1]

    test_losses1 = []
    k = 0
    while k < dataset_size1:

        start = k*batch_size
        end = start + batch_size
        test_losses = sess.run(loss, feed_dict={X: X_test[start:end, :]})
        test_losses1.append(test_losses)
        k = k + 1
    y_est_np = np.zeros(len(test_losses1))
    y_est_np[np.where(test_losses1 > threshold)] = 1
    # np.savetxt("test.txt",y_est_np)
    # np.savetxt("y_test.txt", y_test)
    accuracy = accuracy_score(y_test, y_est_np)
    precision = precision_score(y_test, y_est_np,average="binary")
    recall = recall_score(y_test, y_est_np, average="binary")
    f1 = f1_score(y_test, y_est_np, average="binary")
    print("kdd Accuracy : {:.4f} , Recall : {:.4f} , Precision : {:.4f} , F1 : {:.4f}".format(accuracy, recall,precision, f1))

