#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:49:32 2020

@author: Tao Shi
"""

# Facil Emotion Recognition using CNNS
# CNN is a collection of two types of layers: 
# 1 The hidden layers/Feature extraction part(include convolutions and pollings)
# 2 the classifier part
# Dropout: randomly selected neurons are ignored during the training, which means
# they are dropped out randomly. This technique is used to reduce overfitting
 
   
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
 
# We used the Kaggle's FER2013 dataset
data = pd.read_csv('/Users/apple/Downloads/2020毕业设计/基于深度学习的人脸表情识别/fer2013.csv')
width, height = 48, 48 # The scale of image is 48 * 48 pixel
datapoints = data['pixels'].tolist()

# Getting features for training: convert to numpy arrays
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X) 
X = np.expand_dims(X, -1) # add an additional dimension to feature vector

# Getting labels for training: convert to numpy arrays
y = pd.get_dummies(data['emotion']).values

if __name__ == '__main__':
    print('数据预处理完成')
    print('样本的特征数量: {}'.format(len(X[0])))
    print('样本数量: {}'.format(len(X)))
    print('标签数量：{}'.format(len(y))) 
    # 一共有7种标签：(0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Sad, 5:Surprise, 6:Neutral)
    
    # 对输入样本进行正则化处理
    X -= np.mean(X, axis = 0) 
    X /= np.std(X, axis = 0)
    
    for xx in range(10): # 打印前10张图片看一看 
        plt.figure(xx)
        plt.imshow(X[xx].reshape((48, 48)), interpolation='none', cmap='gray')
        plt.show()
        
        # 把数据集划分成训练集、验证集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = 0.1, random_state = 41)
    
    # 保存划分好的测试集
    np.save('X_train', X_train)
    np.save('y_train', y_train)
    np.save('X_valid', X_valid)
    np.save('y_valid', y_valid)
    np.save('X_test', X_test)
    np.save('y_test', y_test)
    

    



