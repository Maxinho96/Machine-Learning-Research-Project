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


# def load_data(img_dir):
#     return np.array([cv2.imread(os.path.join(img_dir, img)).flatten() for img in os.listdir(img_dir) if img.endswith(".jpg")])

# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])
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
    # sigmoid = tf.nn.sigmoid

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


def main(unused_argv):

    # run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1}))

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="every_image_model_v1") # , config=run_config)

    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    # train_data = mnist.train.images  # Returns np.array
    # train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    data_dir = "ycb-data-cropped"
    data_paths = []
    for object_dir in os.listdir(data_dir):
        if not object_dir.startswith("."):
            for img in os.listdir(os.path.join(data_dir, object_dir)):
                # if img.endswith(".jpg") and (img.startswith("NP1") or img.startswith("NP2")):
                if img.endswith(".jpg"):
                    data_paths += [[data_dir, object_dir, img]]
    random.shuffle(data_paths)

    train_data = np.empty((0, 28 * 28 * 3), dtype="float32")
    train_labels = np.empty(0, dtype="int64")
    eval_data = np.empty((0, 28 * 28 * 3), dtype="float32")
    eval_labels = np.empty(0, dtype="int64")
    test_percentage = 0.1
    test_length = int(test_percentage * len(data_paths))
    c = 0
    for data_dir, object_dir, img in data_paths:
        image = cv2.imread(os.path.join(data_dir, object_dir, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        # image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image', cv2.resize(image_resized, (500, 500)))
        # cv2.waitKey(1)
        eval_data = np.concatenate(
            (
                eval_data,
                np.expand_dims(image_resized.flatten(), axis=0)
            ), axis=0)
        eval_labels = np.concatenate(
            (
                eval_labels,
                np.array([int(object_dir[:3]) - 1])
            ), axis=0)
        data_paths.remove([data_dir, object_dir, img])
        c+=1
        if(c >= test_length):
            break

    c = 0
    for data_dir, object_dir, img in data_paths:
        image = cv2.imread(os.path.join(data_dir, object_dir, img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype('float32')
        # image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX)
        image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow('image', cv2.resize(image_resized, (500, 500)))
        # cv2.waitKey(1)
        train_data = np.concatenate(
            (
                train_data,
                np.expand_dims(image_resized.flatten(), axis=0)
            ), axis=0)
        train_labels = np.concatenate(
            (
                train_labels,
                np.array([int(object_dir[:3]) - 1])
            ), axis=0)
        c += 1
        if c == 100:
            # Train the model
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_data},
                y=train_labels,
                batch_size=100,
                num_epochs=None,
                shuffle=False)
            mnist_classifier.train(
                input_fn=train_input_fn,
                steps=400)

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)

            train_data = np.empty((0, 28 * 28 * 3), dtype="float32")
            train_labels = np.empty(0, dtype="int64")
            c = 0

    # # Train and evaluate the model
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": train_data},
    #     y=train_labels,
    #     batch_size=100,
    #     num_epochs=None,
    #     shuffle=True)
    # train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=20000)
    #
    # eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={"x": eval_data},
    #     y=eval_labels,
    #     num_epochs=1,
    #     shuffle=False)
    # eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=5)
    #
    # tf.estimator.train_and_evaluate(mnist_classifier, train_spec, eval_spec)

if __name__ == "__main__":
    tf.app.run()
