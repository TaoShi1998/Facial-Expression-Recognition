# Facial-Expression-Recognition


## Overview
Human face expression recognition is one of the most powerful and challenging tasks in social communication. In this project, I aim to use Deep Neural Network(DNN) to accurately classify the expressions of face images.


## Related Dataset
I used FER-2013 dataset for this project, which includes 48x48 pixel grayscale images of faces. 

Please visit https://www.kaggle.com/msambare/fer2013 for more information about FER-2013.


## Model Architecture
To achieve facial expression recognition, I constructed a VGG16 using Keras, added Dropout on Fully Connected Layers to reduce overfitting and Batch Normalization for faster and more stable training. 

Please visit https://arxiv.org/abs/1409.1556 for more information about FER-2013.


## Model Evaulation
The model was evaluated on the test sets, achieved an accuracy of 65.00%, which was the 7th place in the *Kaggle Facial Expression Recognition Challenge*.
![result](https://user-images.githubusercontent.com/37060800/132017869-f1595de8-3aea-45d6-9e7b-1f36f1848348.png)



