# Keras-API
Keras + Flask + PreTrained Models

Flask configuration: https://pythonprogramming.net/basic-flask-website-tutorial/?completed=/practical-flask-introduction/

# Set up the server at Digital Ocean

https://www.youtube.com/watch?v=qZNL4Ku1UQg&t=96s

# Pre-trained models available
The models available are the ones that you can find through Keras. The following list contain the model used through the project:

    InceptionResNetV2
    InceptionV3
    Xception
    ResNet50
    VGG16
    VGG19
    MobileNet
    DenseNet


# Example

To use a model you can use the kind of url: http://ip-server/ImageRecog/NameOfModel?image=LinkOfYourJPG

Example with the Pre-trained model ResNet50 with my own server: http://138.68.21.27/ImageRecog/ResNet50?image=https://www.uncompagnon.fr/upload/annonces/41972/middle/01-55637871_1.jpg

The output of the pretrained model ResNet50:

![alt text](https://github.com/JulienHeiduk/Keras-API/blob/master/LittleDog.png)
