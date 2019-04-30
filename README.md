# Research in object classification and novelty detection
This is a cutting-edge project in the field of machine learning, developed at the ISTC-CNR research centre. The system is designed to work on a robot and it allows to classify everyday objects which are in the environment using a convolutional neural network. Furthermore, I started a research activity to let the robot autonomously detect when the object is unknown to him, so that he can increase his knowledge capturing photos of it. To do this I developed a novelty detection system based on support vector machines.

Technologies involved:
* Python
* TensorFlow
* Scikit-Learn
* OpenCV
* NumPy
* Matplotlib

We got the real objects and the dataset here: http://www.ycbbenchmarks.com

# Structure

## Model built from scratch
* <strong>cnn_ycb_v1.py</strong>: model definition from scratch and first version of training pipeline
* <strong>cnn_ycb_v2.py</strong>: same model definition, but second version of training pipeline
* <strong>predict_ycb_v1.py</strong>: simulation of the robot behaviour during prediction using the laptop webcam or smartphone camera
* <strong>predict_ycb_for_results.py</strong>: script to get scientific results of the model on the test set
## Model built using transfer learning and Support Vector Machine
* <strong>retrain.py</strong>: model definition using transfer learning (CNN), SVM definition and training pipeline
* <strong>prediction_mode</strong>: simulation of the robot behaviour during prediction using the laptop webcam or smartphone camera
* <strong>prediction_mode_for_results.py</strong>: script to get scientific results of the model on the test set
## Utilities
* <strong>show_img_from_ip_webcam.py</strong>: script to use the smartphone camera connected in the network
* <strong>crop_images.py</strong>: script to clean the dataset, which crops all images at the centre
