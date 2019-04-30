from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
from tkinter import *

# IP_CAM
import urllib

import subprocess
import time
import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn import svm
from sklearn.externals import joblib
import sklearn.preprocessing

# import matplotlib
# matplotlib.use('Qt5Agg')
# print(matplotlib.get_backend())

import matplotlib.pyplot as plt


def load_graph(model_file):

    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image(image, input_height=299, input_width=299, input_mean=0, input_std=255):

    """
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img, np.uint8)

    float_caster = tf.cast(img_tensor, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    sess = tf.Session()
    result = sess.run(normalized)
    """

    img = cv2.resize(image, (input_width, input_height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # input_mean = np.mean(img)
    # input_std = np.std(img)
    img = img.astype(np.float32)
    img = np.divide(np.subtract(img, input_mean), input_std)
    img = np.expand_dims(img, axis=0)

    '''
    immagine = np.add(np.multiply(t, input_std), input_mean)
    immagine = immagine.astype(np.uint8)
    immagine = np.squeeze(immagine)
    immagine = cv2.cvtColor(immagine, cv2.COLOR_RGB2BGR)
    cv2.imshow("predicting", immagine)
    '''

    # return result
    return img


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


'''
# se non esiste crea una cartella in [dir_path], altrimenti la svuota
def initialize_folder(dir_path):

    if not Path.exists(Path(dir_path)):
        os.makedirs(dir_path)
    else:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as err:
                print(err)
'''


'''
# salva nella cartella [dir_path] un'immagine presa dalla webcam [cap] e ne restituisce il percorso
def save_img_form_webcam_and_get_path(cap, dir_path):

    ret, img = cap.read()

    height, width, channels = img.shape

    if width > height:
        cropped_img = img[0:height, int((width - height) / 2):int(((width - height) / 2) + height)]
    elif width < height:
        cropped_img = img[0:height, int((width - height) / 2):int(((width - height) / 2) + height)]
    else:
        cropped_img = img

    path = str(dir_path)+"frame.jpg"

    if ret:
        try:
            cv2.imwrite(path, cropped_img)
        except IOError as err:
            print(err)
            path = "error"
    else:
        path = "error"

    return path
'''


# prende un'immagine dalla ipcam
def get_img_from_ipcam():

    url = "http://192.168.67.39:8080/shot.jpg"
    ret = True
    image = None

    try:
        img_resp = urllib.request.urlopen(url)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        image = cv2.imdecode(img_np, -1)
    except Exception as e:
        print(e)
        ret = False

    return ret, image


# prende un'immagine e la restituisce dopo averla resa quadrata
def crop_img(img):

    height, width, channels = img.shape

    if width > height:
        cropped_img = img[0:height, int((width - height) / 2):int(((width - height) / 2) + height)]
    elif width < height:
        cropped_img = img[int((height - width) / 2):(int((height - width) / 2) + width), 0:width]
    else:
        cropped_img = img
    return cropped_img


def learn_new_class(nome):

    path = os.path.join('retrain_files/retrain_dataset/', nome, '')

    index = 1

    if not Path.exists(Path(path)):
        os.makedirs(path)
    else:
        index += len(os.listdir(path))

    # PC_CAM
    cap = cv2.VideoCapture(camera)

    while True:

        # PC_CAM
        ret, image = cap.read()

        # IP_CAM
        # ret, image = get_img_from_ipcam()

        if ret:
            cropped_img = crop_img(image)
            cv2.imshow('capturing: ' + nome, cropped_img)
            pressed_key = cv2.waitKey(1)
            if pressed_key == ord('s'):
                try:
                    cv2.imwrite(path + str(index) + ".jpg", cropped_img)
                    cv2.setWindowTitle('capturing: ' + nome, 'capturing: ' + nome + ', ' + str(index))
                    index += 1
                except IOError as e:
                    print(e)
            elif pressed_key == 27:
                break
        else:
            print("error retrieving img from webcam, ret = ", ret)

    # PC_CAM
    # cap.release()

    cv2.destroyAllWindows()

    subprocess.call(['python3', 'retrain_v4.py'])

#
# def callback(event=None):
#
#     name = e.get()
#     master.destroy()
#     learn_new_class(name)


def take_input_class():

    # global master       # SISTEMARE: non dovresti usare global
    # global e
    #
    # master = Tk()
    # master.title("Insert class name")
    # master.geometry("200x60")
    #
    # e = Entry(master)
    # e.focus_set()
    # e.bind("<Return>", callback)
    # e.pack()
    #
    # b = Button(master, text="OK", width=10, command=callback)
    # b.pack()
    #
    # master.mainloop()
    print("Insert class name:")
    learn_new_class(input())


# def check_if_unknown(last_five_best_logit, last_five_best_softmax):
#
#     ret = False
#
#     for softmax in last_five_best_softmax:
#         if softmax < 0.9:
#             ret = True
#             break
#
#     return ret

def check_if_unknown(model_file, bottleneck):
    normalized_bottleneck = sklearn.preprocessing.normalize([bottleneck], norm="max")
    svm_classifier = joblib.load(model_file)
    return svm_classifier.predict(normalized_bottleneck) == -1


if __name__ == "__main__":

    global camera
    camera = 0

    model_file = "retrain_files/retrained_graphs/ycb_from_webcam_retrained_graph.pb"
    label_file = "retrain_files/retrained_graphs/ycb_from_webcam_retrained_labels.txt"
    svm_model_file = "retrain_files/svm_models/svm_model.joblib.pkl"
    input_height = 224
    input_width = 224
    input_mean = 127.5
    input_std = 127.5
    input_layer = "input"
    output_layer = "final_result"   # softmax
    bottleneck_layer = "MobilenetV1/Predictions/Reshape"
    # logits_layer = "final_training_ops/Wx_plus_b/add"

    graph = load_graph(model_file)

    ''' # code to load imgs from folders
    data_dir = "/Users/Massi/Downloads"
    data_paths = []
    for object_dir in os.listdir(data_dir):
        if not object_dir.startswith("."):
            for img in os.listdir(os.path.join(data_dir, object_dir)):
                if not img.startswith(".") and img.endswith(".jpg"):
                    data_paths += [os.path.join(data_dir, object_dir, img)]
    '''

    ''' # code to initialize folder to store webcam's prediction img
    dir_path = str(Path.home())+"/Immagini/from_webcam/"
    initialize_folder(dir_path)
    '''

    # PC_CAM
    cap = cv2.VideoCapture(camera)

    last_five_best_softmax = []

    for i in range(5):
        last_five_best_softmax += [-1]

    while True:

        inizio = time.time()

        # PC_CAM
        ret, image = cap.read()

        # IP_CAM
        # ret, image = get_img_from_ipcam()

        if ret:

            cropped_img = crop_img(image)
            resized_image = cv2.resize(cropped_img, (input_width, input_height))

            t = read_tensor_from_image(resized_image, input_height, input_width, input_mean, input_std)

            input_name = "import/" + input_layer
            output_name = "import/" + output_layer
            bottleneck_name = "import/" + bottleneck_layer
            input_operation = graph.get_operation_by_name(input_name)
            output_operation = graph.get_operation_by_name(output_name)
            bottleneck_operation = graph.get_operation_by_name(bottleneck_name)

            with tf.Session(graph=graph) as sess:
                start = time.time()
                results = sess.run((output_operation.inputs[0], output_operation.outputs[0], bottleneck_operation.outputs[0]), {input_operation.outputs[0]: t})
                end = time.time()

            results = (np.squeeze(results[0]), np.squeeze(results[1]), np.squeeze(results[2]))
            logits_results = results[0]
            softmax_results = results[1]
            bottleneck_results = results[2]

            top_k = softmax_results.argsort()[-5:][::-1]
            labels = load_labels(label_file)

            print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))

            template = '{} ({:.1f}%, {:.1f})'
            for i in top_k:
                print(template.format(labels[i], softmax_results[i] * 100, logits_results[i]))

            '''

            first_result_exp = np.exp(logits_results[top_k[0]])
            other_result_exp = 0

            for i in range(0, len(top_k)):
                other_result_exp += np.exp(logits_results[top_k[i]])

            softmax = first_result_exp / other_result_exp

            '''

            class_id = labels[top_k[0]]
            softmax_top_result = softmax_results[top_k[0]]
            logits_top_result = logits_results[top_k[0]]

            image_to_display = cv2.resize(resized_image, (720, 720))

            last_five_best_softmax += [softmax_top_result]
            last_five_best_softmax.pop(0)

            # if check_if_unknown(None, last_five_best_softmax):
            #     text = "sconosciuto"
            if check_if_unknown(svm_model_file, bottleneck_results):
                text = "sconosciuto"
            else:
                text = template.format(class_id, 100 * softmax_top_result, logits_top_result)

            font = cv2.FONT_HERSHEY_SIMPLEX
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX = (cropped_img.shape[1] - textsize[0]) / 2
            textY = (cropped_img.shape[0] + textsize[1]) / 10

            cv2.putText(image_to_display, text, (int(textX), int(textY)), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('predicting', image_to_display)

            key_pressed = cv2.waitKey(1)

            if key_pressed == ord('n'):     # inserisci nuova class
                # PC_CAM
                cap.release()
                cv2.destroyAllWindows()
                take_input_class()
                graph = load_graph(model_file)
                # PC_CAM
                cap = cv2.VideoCapture(camera)

            elif key_pressed == 27:         # esci dal programma
                break

            plt.ion()
            plt.bar(x=[label[:3] for label in labels], height=softmax_results)
            plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            text_x = labels[int(0.7 * len(labels))][:3]
            text_y = 0.5
            plt.text(x=text_x, y=text_y, s="\n".join(labels))
            # plt.show()
            plt.draw()
            plt.ioff()
            plt.cla()

            print("tempo di ciclo: ", time.time()-inizio)


    # PC_CAM
    cap.release()
    cv2.destroyAllWindows()
