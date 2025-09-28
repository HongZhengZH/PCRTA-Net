import glob
import os
import numpy as np
import  csv
import pandas as pd
########################################################################
def split_data(traindata_str, testdata_str,root):
    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息

    file = open(root+traindata_str, 'r', encoding='utf-8', newline="")
    reader = csv.reader(file)
    train_data = []
    for line in reader:
        train_data.append(line)
    file.close()

    file_test = open(root+testdata_str, 'r', encoding='utf-8', newline="")
    reader_test = csv.reader(file_test)
    test_data = []
    for line in reader_test:
        test_data.append(line)
    file_test.close()

    for i, row in enumerate(test_data):
        val_images_path.append(row[0])  # 存储验证集的所有图片路径
        val_images_label.append(int(row[1]))  # 存储验证集图片对应索引信息
    for i, row in enumerate(train_data):
        train_images_path.append(row[0])  # 存储训练集的所有图片路径
        train_images_label.append(int(row[1]))  # 存储训练集图片对应索引信息


    return train_images_path, train_images_label, val_images_path, val_images_label


