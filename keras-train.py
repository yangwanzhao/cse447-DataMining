from sklearn.metrics import classification_report, roc_auc_score
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import os
import sys
import numpy as np

sys.path.append('..')


PIE_PATH = 'PIE_32x32'
YALE_PATH = 'YaleB_32x32'
INIT_LR = 1e-3
BS = 32
norm_size = 32
NumFILES = 10


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--network", default='CNN',
                    help="neural network mode")
    ap.add_argument("-e", "--epoch", default='10',
                    help="epoch")
    ap.add_argument("-d", "--datapath", default='PIE',
                    help="datapath")
    args = vars(ap.parse_args())
    return args


class LeNet:
    def build(width, height, depth, classes):
        print("[INFO] Building CNN model...")
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
        # if we are using "channels last", update the input shape
        if K.image_data_format() == "channels_first":  # for tensorflow
            inputShape = (depth, height, width)
        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

class DNN:
    def build(input_dim, classes):
        print("[INFO] Building DNN model...")
        # initialize the model
        model = Sequential()
        model.add(Dense(512, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

def load_data(path, mode, class_num):

    X, y = load_face_file(path)

    data = []
    labels = []

    for index, pic in enumerate(X):
        pic = np.array(pic)
        if mode == 'CNN':
            pic = pic.reshape(32, 32, 1)
        data.append(pic)

        # extract the class label from the image path and update the
        # labels list
        label = int(y[index])
        labels.append(label)

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=class_num)
    return data, labels

def load_face_file(path):
    raw_file = np.loadtxt(path)
    X = raw_file[:, :-1]
    y = raw_file[:, -1].astype(int)
    return X, y-1


def predict(testX, testy):
    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model('./keras_dnn.model')

    # classify the input image
    prob_y = model.predict(testX)
    pred_y = np.argmax(prob_y, axis=1)
    testy = np.argmax(testy, axis=1)

    report = classification_report(testy, pred_y, output_dict=True)
    report['auc'] = roc_auc_score(testy, prob_y, multi_class='ovr')

    print("Accuracy:", report['accuracy'])
    print("Precision:", report['macro avg']['precision'])
    print("Recall:", report['macro avg']['recall'])
    print("F1-Score:", report['macro avg']['f1-score'])
    print("AUC:", report['auc'])


def train_dnn(aug, trainX, trainY, testX, testY, mode, epoch, class_num):
    # initialize the model
    print("[INFO] compiling model...")

    if mode == 'CNN':
        model = LeNet.build(width=norm_size, height=norm_size, depth=1, classes=class_num)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / epoch)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])

        # train the network
        print("[INFO] training network...")
        H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                                validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                                epochs=epoch, verbose=1)
    else:
        model = DNN.build(input_dim= norm_size * norm_size, classes=class_num)
        opt = Adam(lr=INIT_LR, decay=INIT_LR / epoch)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        print("[INFO] training network...")
        H = model.fit(trainX, trainY, epochs=epoch, batch_size=BS, validation_data=(testX, testY))


    # save the model to disk
    print("[INFO] serializing network...")
    model.save('./keras_dnn.model')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = epoch
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on traffic-sign classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig('./keras_dnn.png')

if __name__=='__main__':
    args = args_parse()

    if args['datapath'] == 'PIE':
        data_path = PIE_PATH
        classes = 68
        print("[INFO] loading PIE images...")
    else:
        data_path = YALE_PATH
        classes = 38
        print("[INFO] loading YALE images...")

    for i in range(NumFILES):
        train_file_path = data_path + f'/StTrainFile{i + 1}.txt'
        test_file_path = data_path + f'/StTestFile{i + 1}.txt'

        trainX, trainY = load_data(train_file_path, mode=args['network'], class_num=classes)
        testX, testY = load_data(test_file_path, mode=args['network'], class_num=classes)

        aug = ImageDataGenerator()
        train_dnn(aug,trainX,trainY,testX,testY, args['network'], int(args['epoch']), class_num=classes)
        predict(testX, testY)

