if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras import initializers
import argparse
import torch
import torch.nn as nn
import numpy as np

def LeNet():
    model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(6, 5,
                                    padding='same',
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),
                                    input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AveragePooling2D(2, 2),
            tf.keras.layers.Conv2D(16, 5,
                                    padding='valid',
                                    kernel_initializer=initializers.RandomNormal(stddev=0.01),),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AveragePooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, kernel_initializer=initializers.RandomNormal(stddev=0.01),),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(84, kernel_initializer=initializers.RandomNormal(stddev=0.01),),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(10, kernel_initializer=initializers.RandomNormal(stddev=0.01),),
    ])
    return model

def Logistic_TF():
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(120, batch_input_shape=(1, 23), kernel_initializer=initializers.RandomNormal(stddev=0.01),),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(2, kernel_initializer=initializers.RandomNormal(stddev=0.01),),
    ])
    return model

def Bank_TF():
    model = tf.keras.Sequential([
            tf.keras.layers.Dense(250, batch_input_shape=(1, 20), kernel_initializer=initializers.RandomNormal(stddev=0.01),),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.Dense(2, kernel_initializer=initializers.RandomNormal(stddev=0.01),),
    ])
    return model

class Bank_torch(nn.Sequential):
    def __init__(self):
        super(Bank_torch, self).__init__()
        self.fc1 = nn.Linear(20, 250)
        self.fc2 = nn.Linear(250, 2)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class Logistic(nn.Sequential):
    def __init__(self):
        super(Logistic, self).__init__()
        self.fc1 = nn.Linear(23, 120)
        self.fc2 = nn.Linear(120, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def training(args):
    batch_size = args.batch_size
    epochs = args.epoch_num

    img_rows, img_cols = 28, 28
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = (x_train - 0.1307) / 0.3081
    x_test = (x_test - 0.1307) / 0.3081

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    model = LeNet()

    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save(args.model_path)

def loading_credit(args):
    with tf.Session() as sess:
        temp_inp = np.zeros((1, 23))
        model_tf = Logistic_TF()
        model_tf.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                metrics=['accuracy'])
        sess.run(tf.global_variables_initializer())
        print(sess.run(model_tf(temp_inp)))
    
        model_torch = Logistic()
        model_torch.load_state_dict(torch.load('./model/Credit.pth', map_location=torch.device('cpu')))
        model_torch.eval()

        lr1 = [model_torch.fc1.weight.data.numpy().transpose(), model_torch.fc1.bias.data.numpy()]
        lr2 = [model_torch.fc2.weight.data.numpy().transpose(), model_torch.fc2.bias.data.numpy()]
        model_tf.layers[0].set_weights(lr1)
        model_tf.layers[2].set_weights(lr2)

        print(model_torch(torch.from_numpy(temp_inp).type(torch.float)))
        print(sess.run(model_tf(temp_inp)))
        model_tf.save('./model/Credit.h5')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_tf1 = tf.keras.models.load_model('./model/Credit.h5')
        print(sess.run(model_tf1(temp_inp)))

def loading_bank(args):
    with tf.Session() as sess:
        temp_inp = np.zeros((1, 20))
        model_tf = Bank_TF()
        model_tf.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                metrics=['accuracy'])
        sess.run(tf.global_variables_initializer())
        print(sess.run(model_tf(temp_inp)))
    
        model_torch = Bank_torch()
        model_torch.load_state_dict(torch.load('./model/Bank.pth', map_location=torch.device('cpu')))
        model_torch.eval()

        lr1 = [model_torch.fc1.weight.data.numpy().transpose(), model_torch.fc1.bias.data.numpy()]
        lr2 = [model_torch.fc2.weight.data.numpy().transpose(), model_torch.fc2.bias.data.numpy()]
        model_tf.layers[0].set_weights(lr1)
        model_tf.layers[2].set_weights(lr2)

        print(model_torch(torch.from_numpy(temp_inp).type(torch.float)))
        print(sess.run(model_tf(temp_inp)))
        model_tf.save('./model/Bank.h5')
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model_tf1 = tf.keras.models.load_model('./model/Bank.h5')
        print(sess.run(model_tf1(temp_inp)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default='./model/MNIST_LeNet_Sigmoid.h5')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    loading_bank(args)