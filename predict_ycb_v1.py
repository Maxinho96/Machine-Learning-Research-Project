from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import cnn_ycb_v1
import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    data_dir = "ycb-data-cropped"
    class_to_name = {}
    for object_dir in os.listdir(data_dir):
        if not object_dir.startswith("."):
            class_to_name[int(object_dir[:3]) - 1] = object_dir
    classes = list(class_to_name.keys())
    classes.sort()

    classifier = tf.estimator.Estimator(
        model_fn=cnn_ycb_v1.cnn_model_fn, model_dir="every_image_model_v1")

    cap = cv2.VideoCapture(0)

    while True:

        ret, image = cap.read()

        if ret:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype('float32')
            # image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX)
            image_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": np.expand_dims(image_resized.flatten(), axis=0)},
                num_epochs=1,
                shuffle=False)
            predictions = list(classifier.predict(input_fn=predict_input_fn))

            template = '{} ({:.1f}%)'
            for prediction in predictions:
                class_id = prediction["classes"]
                probability = prediction["probabilities"][class_id]
                image_to_display = image_resized.astype('uint8')
                image_to_display = cv2.cvtColor(image_to_display, cv2.COLOR_RGB2BGR)
                image_to_display = cv2.resize(image_to_display, (500, 500))
                plt.ion()
                # plt.hist(class_to_name.keys(), 10, weights=prediction["probabilities"])
                plt.bar(x=[x+1 for x in classes], height=prediction["probabilities"])
                names = list(class_to_name.values())
                names.sort()
                plt.text(x=8, y=0.2, s="\n".join(names))
                plt.xticks([x+1 for x in classes])
                plt.draw()
                plt.ioff()
                plt.cla()
                cv2.putText(image_to_display, template.format(class_to_name[class_id], 100 * probability), (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('image', image_to_display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    tf.app.run()
