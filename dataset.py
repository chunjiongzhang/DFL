#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def gen_train_valid_data():
    training_df = pd.read_csv("KDD/train.csv")
    testing_df = pd.read_csv("KDD/test.csv")
    #training_df, testing_df = train_test_split(training_df, test_size=0.3, random_state=123)
    # ===================replace the label to Normal/Dos/R2L/U2R/Probe=========================
    # A list ot attack names that belong to each general attack type
    dos_attacks = ["snmpgetattack.", "back.", "land.", "neptune.", "smurf.", "teardrop.", "pod.", "apache2.",
                   "udpstorm.", "processtable.", "mailbomb."]
    r2l_attacks = ["snmpguess.", "worm.", "httptunnel.", "named.", "xlock.", "xsnoop.", "sendmail.", "ftp_write.",
                   "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."]
    u2r_attacks = ["sqlattack.", "buffer_overflow.", "loadmodule.", "perl.", "rootkit.", "xterm.", "ps."]
    probe_attacks = ["ipsweep.", "nmap.", "portsweep.", "satan.", "saint.", "mscan."]

    # Our new labels
    classes = ["Normal", "Dos", "R2L", "U2R", "Probe"]

    # Helper function to label samples to 5 classes
    def label_attack(row):
        if row["label"] in dos_attacks:
            return classes[1]
        if row["label"] in r2l_attacks:
            return classes[2]
        if row["label"] in u2r_attacks:
            return classes[3]
        if row["label"] in probe_attacks:
            return classes[4]
        return classes[0]

    # We combine the datasets temporarily to do the labeling
    test_samples_lengtr = len(training_df)
    test_samples_length = len(testing_df)

    df = pd.concat([training_df, testing_df])
    df["Class"] = df.apply(label_attack, axis=1)

    # The old outcome field is dropped since it was replaced with the Class field, the difficulty field will be dropped as well.
    df = df.drop("label", axis=1)

    # we again split the data into training and test sets.
    training_df = df.iloc[:test_samples_lengtr, :]
    testing_df = df.iloc[test_samples_lengtr:test_samples_lengtr +test_samples_length, :]

    # ==============================Preparing the Features====================================
    # Helper function for scaling continous values
    def minmax_scale_values(training_df, testing_df, col_name):
        # print(col_name)
        scaler = MinMaxScaler()
        scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
        train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
        training_df[col_name] = train_values_standardized

        test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
        testing_df[col_name] = test_values_standardized


    # Helper function for one hot encoding
    def encode_text(training_df, testing_df,  name):
        training_set_dummies = pd.get_dummies(training_df[name])
        testing_set_dummies = pd.get_dummies(testing_df[name])


        for x in training_set_dummies.columns:
            dummy_name = "{}_{}".format(name, x)
            training_df[dummy_name] = training_set_dummies[x]

            if x in testing_set_dummies.columns:
                testing_df[dummy_name] = testing_set_dummies[x]
            else:
                testing_df[dummy_name] = np.zeros(len(testing_df))
        training_df.drop(name, axis=1, inplace=True)
        testing_df.drop(name, axis=1, inplace=True)

    sympolic_columns = ["protocol_type", "service", "flag"]
    label_column = "Class"
    for column in df.columns[:]:
        if column in sympolic_columns:
            encode_text(training_df, testing_df, column)
        elif not column == label_column:
            minmax_scale_values(training_df, testing_df,  column)

    x, y = training_df, training_df.pop("Class").values
    X_train = x.values
    # testing_df= testing_df.iloc[:,1:]
    x_test, y_test = testing_df, testing_df.pop("Class").values
    X_test = x_test.values
    print(x_test.shape)
    #print(test_samples_length)

    y_train = np.ones(len(y), np.int8)
    y_train[np.where(y == classes[0])] = 0

    y_test1 = np.ones(len(y_test), np.int8)
    y_test1[np.where(y_test == classes[0])] = 0

    return X_train[np.where(y_train==0)], y_train, X_test, y_test1
