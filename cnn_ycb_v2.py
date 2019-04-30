from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import os
import cv2
import numpy as np
import tensorflow as tf
import random

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    input_layer = tf.reshape(features, [-1, 28, 28, 3])
    tf.summary.image(name="images", tensor=input_layer, max_outputs=1000)
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_input_fn(data_paths):
    """An input function for training"""

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28], method=tf.image.ResizeMethod.AREA)
        return image_resized, label

    filenames = []
    labels = []
    for data_dir, object_dir, img in data_paths:
        filenames += [os.path.join(data_dir, object_dir, img)]
        labels += [int(object_dir[:3]) - 1]
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.shuffle(buffer_size=len(data_paths) * 10)
    dataset = dataset.map(parse_function)
    dataset = dataset.repeat()
    dataset = dataset.batch(100)

    return dataset.make_one_shot_iterator().get_next()

def eval_input_fn(data_paths):
    """An input function for evaluating"""

    # Reads an image from a file, decodes it into a dense tensor, and resizes it
    # to a fixed shape.
    def parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [28, 28], method=tf.image.ResizeMethod.AREA)
        return image_resized, label

    filenames = []
    labels = []
    for data_dir, object_dir, img in data_paths:
        filenames += [os.path.join(data_dir, object_dir, img)]
        labels += [int(object_dir[:3]) - 1]

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(len(data_paths))

    return dataset.make_one_shot_iterator().get_next()


def main(unused_argv):

    config = tf.estimator.RunConfig(
        save_summary_steps = 1
    )
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="every_image_cropped_model",
        config=config)

    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    data_dir = "ycb-data"
    data_paths = []
    for object_dir in os.listdir(data_dir):
        if not object_dir.startswith("."):
            for img in os.listdir(os.path.join(data_dir, object_dir)):
                # if img.endswith(".jpg") and (img.startswith("NP1") or img.startswith("NP2")):
                if img.endswith(".jpg"):
                    data_paths += [[data_dir, object_dir, img]]
    random.shuffle(data_paths)

    train_percentage = int(len(data_paths) * 0.9)
    train_data_paths = data_paths[:train_percentage]
    eval_data_paths = data_paths[train_percentage:]

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input_fn(train_data_paths), max_steps=200)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(eval_data_paths), throttle_secs=60)
    tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

    # Train the model
    # mnist_classifier.train(
    #     input_fn=lambda:train_input_fn(train_data_paths),
    #     steps=1000)
    
    # Evaluate the model and print results
    # eval_results = mnist_classifier.evaluate(input_fn=lambda:eval_input_fn(eval_data_paths))
    # print(eval_results)


if __name__ == "__main__":
    tf.app.run()
