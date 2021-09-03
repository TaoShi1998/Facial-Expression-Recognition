#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:47:57 2020

@author: Tao Shi
"""  
 
# Train the model
import numpy as np
from model import model
import matplotlib.pyplot as plt

BATCH_SIZE = 64
EPOCHS = 1
 
X_train = np.load('./X_train.npy')
y_train = np.load('./y_train.npy')
X_valid = np.load('./X_valid.npy')
y_valid = np.load('./y_valid.npy')

model.fit(X_train, y_train,
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          verbose=1,
          validation_data = (X_valid, y_valid), 
          shuffle = True)

print('模型训练完成')

# Test the model on the test set
X_test = np.load('./X_test.npy')
y_test = np.load('./y_test.npy')
 
test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 2)
print('模型在测试集上的损失值为: {:5.2f}'.format(test_loss))
print('模型在测试集上的准确率为: {:5.2f}%'.format(test_acc * 100))


expression_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Visualize the result
predictions = model.predict(X_test)
print(predictions[0]) 
# print(np.argmax(predictions[0]))
# print(np.argmax(y_test[0]))

def plot_image(i, predictions, true_label, image):
    predictions, true_label, image = predictions, true_label, image
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(image, cmap = 'gray')
    
    predicted_label = np.argmax(predictions)
    true_label = np.argmax(true_label)
    if predicted_label == true_label:
        color = 'blue' # 预测正确的话用蓝色表示
    else:
        color = 'red' # 预测错误的话用红色表示
    
    plt.xlabel("Predicted: {}, True: {}".format(expression_names[predicted_label],
                                expression_names[true_label]),
                                color=color)
 
def plot_value_array(i, predictions, true_label):
  predictions, true_label = predictions, true_label
  plt.grid(False)
  plt.xticks(range(7))
  plt.xlabels = [x for x in expression_names]
  plt.yticks([])
  x = range(7) 
  y = predictions
  thisplot = plt.bar(x, y, color="#777777")
  print(x, y)
  _ = plt.xticks(range(7),expression_names) ## 可以设置坐标字
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions)
  true_label = np.argmax(true_label)
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  for a,b in zip(x, y):
      plt.text(a, b+0.05, '{:2.2f}%'.format(b * 100), ha='center', va= 'bottom',fontsize=7)
                    

i = 2
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], y_test[0], X_test[0].reshape((48, 48)))
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], y_test[0])
plt.show()
 

num_rows = 4
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize = (9 * num_cols, 3 * num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
  plot_image(i, predictions[i], y_test[i], X_test[i].reshape((48, 48)))
  plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
  plot_value_array(i, predictions[i], y_test[i])
plt.tight_layout()
#plt.show()
plt.savefig('result.png')



