from flask import Flask, jsonify, request, json
from datetime import datetime
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_jwt_extended import (create_access_token, create_refresh_token, jwt_required, jwt_refresh_token_required, get_jwt_identity, get_raw_jwt)
from imutils import paths
from pyzbar.pyzbar import decode
import imutils
import numpy as np


# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.preprocessing.image import img_to_array

    

import pandas as pd  
import cv2

app = Flask(__name__)


bcrypt = Bcrypt(app)
jwt = JWTManager(app)

CORS(app)

def codabar_detection(path):
    image = cv2.imread(path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    

    # blur and threshold the image
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    

    # construct a closing kernel and apply it to the thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    
    
    # find the contours in the thresholded image, then sort the contours
    # by their area, keeping only the largest one
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    cv2.imwrite("/home/ilyes/bachman/Tunihack/out.jpg", image) 
    
    return (path.split("/")[-1])
def codabar_extraction(path):
    
    bgr = (8, 70, 208)

    image = cv2.imread(path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray_img)

    x=""
    for bc in barcodes:

        x= x+bc.data.decode("utf-8") 
        
    return(x)


def classif(path):
          #chanels last so -1 for tensorflow , channels first so 1 for theano

    EPOCHS = 75
    INIT_LR = 1e-3
    BS = 32
    chanDim=-1
    IMAGE_DIMS = (100,150,3)      

    model=Sequential()

    # CONV => RELU => POOL
    model.add(Conv2D(32, (3, 3), padding="same",input_shape=IMAGE_DIMS))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # (CONV => RELU) * 2 => POOL
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # use a *softmax* activation for single-label classification
    # and *sigmoid* activation for multi-label classification
    model.add(Dense(2))
    model.add(Activation("sigmoid"))
 

    model.load_weights("/home/ilyes/Tunihack/classification/datasetV2_SSize.h5")

    image = cv2.imread(path)

    # pre-process the image for classification
    image = cv2.resize(image, (150, 100))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    proba = model.predict(image)[0]
    
    if (proba[0]>proba[1]):
        return ("Lotion")
    else :
        return("Parfume")



	
@app.route('/buisness', methods=['POST'])
def buisness():
    Matricule = request.get_json()['Matricule']
    Brand = request.get_json()['Brand']
    Sector = request.get_json()['Sector']

   

    print(Matricule)
    print(Brand)
    print(Sector)

    identity = {'path_out':" path_out",'codabar': "codabar",'nature': "nature" }
    return jsonify(identity)
	
@app.route('/codabar', methods=['POST'])
def codabar():
    path = request.get_json()['image_path']
    price = request.get_json()['Price']

    path_out= codabar_detection(path)
    codabar=codabar_extraction(path)
    nature=classif(path)

    print(path_out)
    print(codabar)
    print(price)
    print(nature)

    identity = {'path_out': path_out,'codabar': codabar,'nature': nature }
    return jsonify(identity)
    
@app.route("/test")
def hello():
    return jsonify({'text':'Hello World!'})

"""
@app.route('/abckup', methods=['POST'])
def login():
    name = request.get_json()['Name']
    result = name


    access_token = create_access_token(identity = {'first_name': email,'password ': password })
    result = jsonify({"token":access_token})
    
    print (jsonify({"token":access_token}))
"""	
if __name__ == '__main__':
    app.run(debug=True)