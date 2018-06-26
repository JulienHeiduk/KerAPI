from keras.applications import ResNet50, VGG16, VGG19, MobileNet, DenseNet201, NASNetMobile, Xception, InceptionV3, InceptionResNetV2
import tensorflow as tf

def resnet():
    model = ResNet50(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def vgg16():
    model = VGG16(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def vgg19():
    model = VGG19(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def mobnet():
    model = MobileNet(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def densnet():
    model = DenseNet201(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def nasnmob():
    model = NASNetMobile(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def xception():
    model = Xception(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def inceptionv3():
    model = InceptionV3(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

def inceptionres():
    model = InceptionResNetV2(weights="imagenet")
    graph = tf.get_default_graph()
    return model, graph

