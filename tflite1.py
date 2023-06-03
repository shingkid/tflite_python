from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import zipfile
import urllib.request
from pathlib import Path
import os

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)

  ordered = np.argpartition(-output, 1)
  return [(i, output[i]) for i in ordered[:top_k]][0]


def classifyImagesFromFolder():
  exts = [".png", ".jpg"]
  # Same directory
  global test_dir
  print("Will classify images from ",test_dir)
  files = [p for p in Path(test_dir).iterdir() if p.suffix in exts]
  for filename in files:
    # do your stuff
    print("classifying " + str(filename))
    image = Image.open(filename).convert('RGB').resize((width, height))
    # Classify the image.
    time1 = time.time()
    label_id, prob = classify_image(interpreter, image)
    time2 = time.time()
    classification_time = np.round(time2 - time1, 3)
    #print("Classification Time =", classification_time, "seconds.")

    # Read class labels.
    labels = load_labels(label_path)

    # Return tpwdhe classification label of the image.
    classification_label = labels[label_id]
    print("This is a ", classification_label, ", classified with Accuracy :", np.round(prob * 100, 2), "%." , "in ",str(classification_time)+" secpmds")


model_dir = None
test_dir = None
model_path = None
label_path = None
interpreter = None
height = None
width = None


def setup() :
  current_dir = "."
  global model_dir
  model_dir = current_dir + "/models"
  if not os.path.exists(model_dir):
     os.makedirs(model_dir)

  global test_dir
  test_dir = current_dir + "/test/"

  if not os.path.exists(test_dir):
     os.makedirs(test_dir)

  model_url="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip"

  # Downloading the mobilenet TFLite model
  zip_path, _ = urllib.request.urlretrieve(model_url)
  with zipfile.ZipFile(zip_path, "r") as f:
      f.extractall(model_dir)
  global model_path
  model_path = model_dir + "/mobilenet_v1_1.0_224_quant.tflite"
  global label_path
  label_path = model_dir + "/labels_mobilenet_quant_v1_224.txt"

  global  interpreter
  interpreter = Interpreter(model_path)
  print("Model Loaded Successfully.")

  interpreter.allocate_tensors()
  global height, width
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  print("Image Shape (", width, ",", height, ")")

setup();
classifyImagesFromFolder();
