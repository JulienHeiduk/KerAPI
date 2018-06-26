#system level operations (like loading files)
import sys 
#for reading operating system data
import os
#tell our app where our saved model is
sys.path.append(os.path.abspath("/var/www/FlaskApp/FlaskApp/model"))
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
from flask import Flask, render_template, session
import io
import flask
import requests
from urllib2 import Request
import sys, requests, shutil, os
import urllib2 
import keras as K
import tensorflow as tf
import time
from keras import backend as K
from load import * 
from nlp import *
import json
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
 
# initialize our Flask application and the Keras model
app = Flask(__name__)

#global vars for easy reusability
global model_r, graph_r, model_v, graph_v
global model_m, graph_m, model_d, graph_d
global model_v2, graph_v2, model_x, graph_x
global model_v3, graph_v3, model_ir, graph_ir

#initialize these variables
model_r, graph_r = resnet()
#model_v, graph_v = vgg16()
#model_m, graph_m = mobnet()
#model_d, graph_d = densnet()
#model_v2, graph_v2 = vgg19()
#model_x, graph_x = xception()
#model_v3, graph_v3 = inceptionv3()
#model_ir, graph_ir = inceptionres()

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

#class ReusableForm(Form):
#    name = TextField('Name:', validators=[validators.required()])

@app.route("/")
def hello():
    return render_template('index.html')

#@app.route("/NLP", methods=['GET', 'POST'])
#def nlp():
#    form = ReusableForm(request.form)
# 
#    if request.method == 'POST':
#        name=request.form['Sentence']
# 
#        if form.validate():
            # Save the comment here.
#            flash('Sentence: ' + name)
 #       else:
 #           flash('All the form fields are required. ')
 
#    return render_template('nlp.html', form=dependency())

@app.route("/ImageRecog/InceptionResNetV2")
def inceptres():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_ir.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)

@app.route("/ImageRecog/InceptionV3")
def inceptv3():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_v3.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)

@app.route("/ImageRecog/Xception")
def xcept():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_x.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)


@app.route("/ImageRecog/ResNet50")
def resnet():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_r.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return render_template('display.html', image = val, data = data["predictions"])

@app.route("/ImageRecog/VGG16")
def vgg16():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_v.as_default():
            preds = model_v.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)

@app.route("/ImageRecog/VGG19")
def vgg19():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_v2.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)

@app.route("/ImageRecog/MobileNet")
def mobnet():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_m.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
           # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)

@app.route("/ImageRecog/DenseNet")
def densenet():
    data = {"success": False}
    val = flask.request.args['image']
    response = urllib2.urlopen(str(val))
    image_data = response.read()
    image = Image.open(io.BytesIO(image_data))
    image = prepare_image(image, target=(224, 224))

    with graph_r.as_default():
            preds = model_d.predict(image)
            #preds = ml_predict(image, model)
            results = imagenet_utils.decode_predictions(preds)
            data["predictions"] = []
            # loop over the results and add them to the list of
            # returned predictions
            for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

            data["success"] = True
            return flask.jsonify(data)

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	load_model()

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
                #print("Method Post detected !")
		if flask.request.files.get("image"):
                        #print("Image detected !")
			# read the image in PIL format
			image = flask.request.files["image"].read()
			#print("IMAGE",image)
                        #image = requests.get('http://example.com').content
                        image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))
			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	load_model_r()
	app.run()
