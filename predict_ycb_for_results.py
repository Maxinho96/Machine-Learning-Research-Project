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
    id_to_name = {}
    name_to_id = {}
    for object_dir in os.listdir(data_dir):
        if not object_dir.startswith("."):
            id_to_name[int(object_dir[:3]) - 1] = object_dir[4:]
            name_to_id[object_dir[4:]] = int(object_dir[:3]) - 1
    classes = list(id_to_name.keys())
    classes.sort()

    results = []

    classifier = tf.estimator.Estimator(
        model_fn=cnn_ycb_v1.cnn_model_fn, model_dir="every_image_model_v1")

    prediction_dir = 'prediction_images'
    for object_dir in os.listdir(prediction_dir):
        if not object_dir.startswith("."):
            tot_predictions = 0
            summed_correct_predictions = 0
            correct_predictions = 0
            summed_incorrect_predictions = 0
            incorrect_predictions = 0
            # summed_predictions = 0
            object_name = object_dir.split("_")[0]
            for img in os.listdir(os.path.join(prediction_dir, object_dir)):
                if not img.startswith(".") and img.endswith(".jpg"):
                    image = cv2.imread(os.path.join(prediction_dir, object_dir, img))
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
                        predicted_class_id = prediction["classes"]
                        predicted_probability = prediction["probabilities"][predicted_class_id]
                        predicted_object = id_to_name[predicted_class_id]
                        # current_class_id = name_to_id[object_name]

                        tot_predictions += 1
                        # summed_predictions += prediction["probabilities"][current_class_id] * 100
                        if predicted_object == object_name:
                            summed_correct_predictions += prediction["probabilities"][predicted_class_id] * 100
                            correct_predictions += 1
                        else:
                            summed_incorrect_predictions += prediction["probabilities"][predicted_class_id] * 100
                            incorrect_predictions += 1

                        print(str(tot_predictions) + " " + str(summed_correct_predictions) + " " + str(correct_predictions) + " " + str(summed_incorrect_predictions) + " " + str(incorrect_predictions))

                        image_to_display = image_resized.astype('uint8')
                        image_to_display = cv2.cvtColor(image_to_display, cv2.COLOR_RGB2BGR)
                        image_to_display = cv2.resize(image_to_display, (500, 500))
                        # plt.ion()
                        # # plt.hist(class_to_name.keys(), 10, weights=prediction["probabilities"])
                        # plt.bar(x=[x+1 for x in classes], height=prediction["probabilities"])
                        # plt.xticks([x+1 for x in classes])
                        # plt.draw()
                        # plt.ioff()
                        # plt.cla()
                        cv2.putText(image_to_display, template.format(predicted_object, 100 * predicted_probability), (10, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow('image', image_to_display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            results += [(
                object_dir,
                (correct_predictions / tot_predictions) * 100,
                (summed_correct_predictions / correct_predictions) if correct_predictions != 0 else 0,
                (summed_incorrect_predictions / incorrect_predictions) if incorrect_predictions != 0 else 0
            )]

    for name, accuracy, avg_correct_percentage, avg_incorrect_percentage in results:
        template = '{}: accuracy = {:.1f} % avg correct percentage = {:.1f} % avg incorrect percentage = {:.1f}'
        print(template.format(name, accuracy, avg_correct_percentage, avg_incorrect_percentage))

        # spam_cluttered: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 85.5
        # sugar_blu: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 99.8
        # rubik_cluttered: accuracy = 10.0 % avg
        # correct
        # percentage = 93.6 % avg
        # incorrect
        # percentage = 85.5
        # spam_blu: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 93.7
        # tomato_cluttered: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 94.5
        # mustard_cluttered: accuracy = 75.0 % avg
        # correct
        # percentage = 89.4 % avg
        # incorrect
        # percentage = 94.6
        # rubik_blu: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 97.5
        # banana_bianco: accuracy = 25.0 % avg
        # correct
        # percentage = 98.9 % avg
        # incorrect
        # percentage = 97.1
        # tuna_cluttered: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 94.8
        # chips_cluttered: accuracy = 40.0 % avg
        # correct
        # percentage = 99.0 % avg
        # incorrect
        # percentage = 92.3
        # baseball_bianco: accuracy = 10.0 % avg
        # correct
        # percentage = 72.9 % avg
        # incorrect
        # percentage = 76.6
        # tomato_blu: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 97.7
        # banana_cluttered: accuracy = 20.0 % avg
        # correct
        # percentage = 88.7 % avg
        # incorrect
        # percentage = 92.7
        # timer_bianco: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 66.6
        # sugar_bianco: accuracy = 10.0 % avg
        # correct
        # percentage = 62.9 % avg
        # incorrect
        # percentage = 90.8
        # chips_blu: accuracy = 30.0 % avg
        # correct
        # percentage = 89.6 % avg
        # incorrect
        # percentage = 95.6
        # baseball_blu: accuracy = 25.0 % avg
        # correct
        # percentage = 82.6 % avg
        # incorrect
        # percentage = 83.1
        # tomato_bianco: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 99.1
        # tuna_blu: accuracy = 5.0 % avg
        # correct
        # percentage = 100.0 % avg
        # incorrect
        # percentage = 93.6
        # timer_cluttered: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 82.9
        # baseball_cluttered: accuracy = 65.0 % avg
        # correct
        # percentage = 89.1 % avg
        # incorrect
        # percentage = 82.3
        # sugar_cluttered: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 93.7
        # timer_blu: accuracy = 35.0 % avg
        # correct
        # percentage = 86.1 % avg
        # incorrect
        # percentage = 97.6
        # rubik_bianco: accuracy = 20.0 % avg
        # correct
        # percentage = 92.2 % avg
        # incorrect
        # percentage = 93.8
        # spam_bianco: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 80.8
        # tuna_bianco: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 69.7
        # mustard_blu: accuracy = 25.0 % avg
        # correct
        # percentage = 100.0 % avg
        # incorrect
        # percentage = 98.0
        # chips_bianco: accuracy = 68.4 % avg
        # correct
        # percentage = 99.9 % avg
        # incorrect
        # percentage = 96.2
        # mustard_bianco: accuracy = 40.0 % avg
        # correct
        # percentage = 79.4 % avg
        # incorrect
        # percentage = 83.1
        # banana_blu: accuracy = 0.0 % avg
        # correct
        # percentage = 0.0 % avg
        # incorrect
        # percentage = 100.0

    cv2.destroyAllWindows()

if __name__ == "__main__":
    tf.app.run()
