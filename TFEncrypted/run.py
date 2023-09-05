# pylint:  disable=redefined-outer-name
"""An example of performing secure inference with MNIST.

Also performs plaintext training.
"""

if __name__ == '__main__':
    import os
    import sys
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    p = os.path.abspath('../..')
    if p not in sys.path:
        sys.path.append(p)
    import warnings
    warnings.filterwarnings("ignore")

import logging, tqdm
from pyexpat import features

import tensorflow as tf
import tensorflow.keras as keras

import tf_encrypted as tfe
from convert import decode

from train_plaintext import LeNet
import numpy as np
import torch
from tf2torch import LeNet
import torchvision
from torchvision import datasets, transforms

if len(sys.argv) > 1:
    config_file = sys.argv[1]
    config = tfe.RemoteConfig.load(config_file)
    tfe.set_config(config)
    tfe.set_protocol(tfe.protocol.Pond())

session_target = sys.argv[2] if len(sys.argv) > 2 else None


class ModelOwner:
    """Contains code meant to be executed by the model owner.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the model owner.
    local_data_file: filepath to MNIST data.
  """

    BATCH_SIZE = 128
    NUM_CLASSES = 10
    EPOCHS = 1

    ITERATIONS = 60000 // BATCH_SIZE

    IMG_ROWS = 28
    IMG_COLS = 28
    FLATTENED_DIM = IMG_ROWS * IMG_COLS

    def __init__(self, player_name, local_data_file):
        self.player_name = player_name
        self.local_data_file = local_data_file

    def _build_data_pipeline(self):
        """Build a reproducible tf.data iterator."""

        def normalize(image, label):
            image = (tf.cast(image, tf.float32) / 255.0 - 0.1307) / 0.3081
            return image, label

        def flatten(image, label):
            image = tf.reshape(image, shape=[self.FLATTENED_DIM])
            return image, label
        def reshaping(image, label):
            image = tf.reshape(image, shape=[1, IMG_ROWS, IMG_COLS, 1])
            return image, label
        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        dataset = dataset.map(reshaping)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator

    def _build_training_graph(self, training_data):
        """Build a graph for plaintext model training."""

        model = LeNet()
        # model = keras.Sequential()
        # model.add(keras.layers.Dense(512, input_shape=[self.FLATTENED_DIM]))
        # model.add(keras.layers.Activation("relu"))
        # model.add(keras.layers.Dense(self.NUM_CLASSES, activation=None))

        # optimizer and data pipeline
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

        def loss(model, inputs, targets):
            logits = model(inputs)
            per_element_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=targets, logits=logits
            )
            return tf.reduce_mean(per_element_loss)

        def grad(model, inputs, targets):
            loss_value = loss(model, inputs, targets)
            return loss_value, tf.gradients(loss_value, model.trainable_variables)

        def loop_body(i):
            x, y = training_data.get_next()
            _, grads = grad(model, x, y)
            update_op = optimizer.apply_gradients(zip(grads, model.trainable_variables))
            with tf.control_dependencies([update_op]):
                return i + 1

        loop = tf.while_loop(
            lambda i: i < self.ITERATIONS * self.EPOCHS, loop_body, loop_vars=(0,)
        )

        with tf.control_dependencies([loop]):
            print_op = tf.print("Training complete")
        with tf.control_dependencies([print_op]):
            return [tf.identity(x) for x in model.trainable_variables]

    @tfe.local_computation
    def provide_weights(self):
        with tf.name_scope("loading"):
            training_data = self._build_data_pipeline()

        with tf.name_scope("training"):
            parameters = self._build_training_graph(training_data)

        return parameters


class PredictionClient:
    """
  Contains code meant to be executed by a prediction client.

  Args:
    player_name: `str`, name of the `tfe.player.Player`
                 representing the data owner
    build_update_step: `Callable`, the function used to construct
                       a local federated learning update.
  """

    BATCH_SIZE = 1

    def __init__(self, player_name, local_data_file):
        self.player_name = player_name
        self.local_data_file = local_data_file

    def _build_data_pipeline(self):
        """Build a reproducible tf.data iterator."""

        def normalize(image, label):
            image = (tf.cast(image, tf.float32) / 255.0 - 0.1307) / 0.3081
            return image, label
        def reshaping(image, label):
            image = tf.reshape(image, shape=[1, 28, 28, 1])
            return image, label

        dataset = tf.data.TFRecordDataset([self.local_data_file])
        dataset = dataset.map(decode)
        dataset = dataset.map(normalize)
        # dataset = dataset.map(reshaping)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.BATCH_SIZE)

        iterator = dataset.make_one_shot_iterator()
        return iterator
    def public_inputs(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
        features_np = []
        labels_np = []
        for feature, label in testloader:
            feature = torch.transpose(feature, 1, 3)
            feature = torch.transpose(feature, 1, 2)
            feature = feature.clone().numpy()
            label = label.clone().numpy()
            features_np.append(feature)
            labels_np.append(label)
        return features_np, labels_np

    @tfe.local_computation
    def provide_input(self, prediction_input):
        """Prepare input data for prediction."""
        return prediction_input

    @tfe.local_computation
    def receive_output(self, logits: tf.Tensor):
        with tf.name_scope("post-processing"):
            prediction = tf.argmax(logits, axis=1)
            op = tf.print("Result", prediction)
            return op


if __name__ == "__main__":
    prediction_client = PredictionClient(
        player_name="prediction-client", local_data_file="./data/test.tfrecord"
    )

    features_np, labels_np = prediction_client.public_inputs()
    with tfe.protocol.SecureNN():
        x_tf = tf.placeholder(dtype=tf.float32, shape=(1, 28, 28, 1))
        x = prediction_client.provide_input(x_tf)
        model = tfe.keras.Sequential([
            tfe.keras.layers.Conv2D(6, 5,
                                    padding='same',
                                    batch_input_shape=(1, 28, 28, 1)),
            tfe.keras.layers.BatchNormalization(),
            tfe.keras.layers.Activation('sigmoid'),
            tfe.keras.layers.AveragePooling2D(2, 2),
            tfe.keras.layers.Conv2D(16, 5,
                                    padding='valid',),
            tfe.keras.layers.BatchNormalization(),
            tfe.keras.layers.Activation('sigmoid'),
            tfe.keras.layers.AveragePooling2D(2, 2),
            tfe.keras.layers.Flatten(),
            tfe.keras.layers.Dense(120),
            tfe.keras.layers.BatchNormalization(axis=1),
            tfe.keras.layers.Activation('sigmoid'),
            tfe.keras.layers.Dense(84),
            tfe.keras.layers.BatchNormalization(axis=1),
            tfe.keras.layers.Activation('sigmoid'),
            tfe.keras.layers.Dense(10),
        ])
        logits = model(x)

    plaintext_model = keras.models.load_model('../../model/MNIST_LeNet_Sigmoid.h5')
    weights = plaintext_model.get_weights()
    model_torch = LeNet()
    model_torch.load_state_dict(torch.load('../../model/MNIST_LeNet.pth', map_location=torch.device('cpu')))
    model_torch.eval()
    sess_tf = tf.Session()
    with tfe.Session(target=session_target) as sess:
        sess.run(tf.global_variables_initializer(), tag="init")

        model.set_weights(weights, sess)
        total_num = 0
        acc1 = 0
        acc2 = 0
        AE_num = 0
        for i in tqdm.tqdm(range(10000)):
            current_x = features_np[i]
            real_labels = labels_np[i]
            logits_dec = sess.run(logits.reveal().to_native(), feed_dict={x_tf: current_x})
            prediction = np.argmax(logits_dec, axis=1)
            acc1 += (prediction == real_labels).sum()
            total_num += len(prediction)
            current_x = np.transpose(current_x, (0, 3, 1, 2))
            with torch.no_grad():
                torch_logits = model_torch(torch.from_numpy(current_x))
                prediction_torch = torch.argmax(torch_logits, dim=1).numpy()
            acc2 += (prediction_torch == real_labels).sum()
            if prediction_torch == real_labels and prediction != real_labels:
                AE_num += 1
        acc1 /= total_num
        acc2 /= total_num
        print('accuracy: ', acc1, acc2, AE_num)
