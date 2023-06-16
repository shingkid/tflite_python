from tflite_runtime.interpreter import Interpreter
from PIL import Image
import numpy as np
import time
import zipfile
import urllib.request
from pathlib import Path
import os

# Optional/Advanced import Pi GPIO and time modules
"""
import RPi.GPIO as GPIO
import time
"""

model_dir = None
test_dir = None
model_path = None
label_path = None
interpreter = None
height = None
width = None

def load_labels(path): # Read the labels from the text file as a Python list.
  with open(path, 'r') as f:
    return [line.strip() for i, line in enumerate(f.readlines())]

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Resetting the input tensor with the current image's RGB values 
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)

  #Running inference
  interpreter.invoke()
  
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  #Quantization = mapping continuous infinite values to a smaller set of discrete finite values (approximating real-world values with a digital representation) 
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  ordered = np.argpartition(-output, 1)

  # Getting output image's label as the first output and the probability as the second output  
  return [(i, output[i]) for i in ordered[:top_k]][0]

"""
Attempts to classify all images in the test directory 
"""
def classifyImagesFromFolder():
  allowed_file_extensions = [".png", ".jpg"]
  # Same directory
  global test_dir
  print("Will classify images from ",test_dir)
  # Read class labels.
  labels = load_labels(label_path)

  files = [p for p in Path(test_dir).iterdir() if p.suffix in allowed_file_extensions]
  for filename in files:
    # do your stuff
    
    print("classifying " + str(filename))

    #Images from test folder need to be resized to training data size 
    image = Image.open(filename).convert('RGB').resize((width, height))
   
    time1 = time.time()
    
    #Classifies the image 
    label_id, prob = classify_image(interpreter, image)
    
    time2 = time.time()
    classification_time = np.round(time2 - time1, 3)
    #print("Classification Time =", classification_time, "seconds.")

   
    # Return the classification label of the image.
    classification_label = labels[label_id]
    probability =  np.round(prob * 100, 2)
    print("This is a ", classification_label, ", classified with Accuracy :", probability, "%." , "in ",str(classification_time)+" seconds")

    # Optional/Advanced
    """ 
    if(probability > 90): 
      print("going to light up LEDs")
      #lightUpLEDs()
    """

"""
creates required directories to store models and test images (if reqd.)
downloads and unzips the models  
loads interpreter and labels
"""
def setup() :
  current_dir = "."
  global model_dir
  model_dir = current_dir + "/models_1"
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

  #Loading TFliteModel 
  global  interpreter
  interpreter = Interpreter(model_path)
  print("Model Loaded Successfully.")

  #Allocate tensors to ensure enough memory is preserved for inference
  interpreter.allocate_tensors()

  #Get dimensions of input tensors
  global height, width
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  #print("Image Shape (", width, ",", height, ")")

"""
def lightUpLEDs():
  
  GPIO.setmode(GPIO.BOARD)
  GPIO.setwarnings(False)

  #set up which pin to control the LED from and set it to output
  ledPin = 12
  GPIO.setup(ledPin, GPIO.OUT)
  GPIO.output(ledPin,GPIO.HIGH)
  time.sleep(2)
  GPIO.output(ledPin,GPIO.LOW)
"""
setup();
classifyImagesFromFolder();

