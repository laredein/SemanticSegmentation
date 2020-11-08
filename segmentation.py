import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from keras.models import *
from keras.layers import *
from keras import layers
import random
traintitles=pd.read_csv("train/ImageSets/Segmentation/train.txt",header=None,dtype=str,names=["traintitles"])
segarr=[]
imarr=[]

def crop(im,segim):
    w=random.randint(0,len(im[0])-320)
    h=random.randint(0,len(im)-320)
    return [im[h:h+320,w:w+320],segim[h:h+320,w:w+320]]

c=0
for i in traintitles["traintitles"]:
    segim=plt.imread("train/SegmentationClass/"+i+".png")
    im=plt.imread("train/JPEGImages/"+i+".jpg")
    if len(segim)>=325 and len(segim[0])>=325 and len(im)>=325 and len(im[0])>=325:
        c=c+1
        arr=crop(im,segim)
        segarr.append(arr[1])
        imarr.append(arr[0])
    if c==500:
        break


def types(segarr):
    pixeltypes=[]
    for im in segarr:
        for j in range(len(im)):
            for k in range(len(im[0])):
                if [im[j][k][0],im[j][k][1],im[j][k][2]] not in pixeltypes:
                    pixeltypes.append([im[j][k][0],im[j][k][1],im[j][k][2]])
        if len(pixeltypes)==22:
            break
    return np.array(pixeltypes)

def getitem(segarr,pixeltypes,count,numb):
    arr22=[]
    for i in range(count):
        im=segarr[i]
        im22=np.zeros((len(im),len(im[0]),numb))
        print(count-i)
        for c in range(numb):
            for j in range(len(im)):
                for k in range(len(im[0])):
                    if im[j][k][0]==pixeltypes[c][0] and im[j][k][1]==pixeltypes[c][1] and im[j][k][2]==pixeltypes[c][2]:
                        im22[j][k][c]=1;
        arr22.append(im22)
    return np.array(arr22)
def topict(im22):
    im=np.zeros((len(im22),len(im22[0]),3))
    for j in range(len(im)):
        for k in range(len(im[0])):
            cons=0
            for c in range(22):
                if im22[j][k][c]>=cons:
                    im[j][k][0]=pixeltypes[c][0]
                    im[j][k][1]=pixeltypes[c][1]
                    im[j][k][2]=pixeltypes[c][2]
                    cons=im22[j][k][c]
    return im


pixeltypes=types(segarr)
arr22=getitem(segarr,pixeltypes,50,22)
imarr=np.array(imarr)


def get_model(img_size,num_classes):
    inputs=keras.Input(shape=img_size+(3,))
    # x=layers.experimental.preprocessing.RandomCrop(160,160)(inputs)
    x=layers.Conv2D(32,3,strides=2,padding="same")(inputs)
    x=layers.BatchNormalization()(x)
    x=layers.Activation("relu")(x)

    previous_block_activation=x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64,128,256]:
        x=layers.Activation("relu")(x)
        x=layers.SeparableConv2D(filters,3,padding="same")(x)
        x=layers.BatchNormalization()(x)

        x=layers.Activation("relu")(x)
        x=layers.SeparableConv2D(filters,3,padding="same")(x)
        x=layers.BatchNormalization()(x)

        x=layers.MaxPooling2D(3,strides=2,padding="same")(x)

        # Project residual
        residual=layers.Conv2D(filters,1,strides=2,padding="same")(previous_block_activation)
        x=layers.add([x,residual])  # Add back residual
        previous_block_activation=x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256,128,64,32]:
        x=layers.Activation("relu")(x)
        x=layers.Conv2DTranspose(filters,3,padding="same")(x)
        x=layers.BatchNormalization()(x)

        x=layers.Activation("relu")(x)
        x=layers.Conv2DTranspose(filters,3,padding="same")(x)
        x=layers.BatchNormalization()(x)

        x=layers.UpSampling2D(2)(x)

        # Project residual
        residual=layers.UpSampling2D(2)(previous_block_activation)
        residual=layers.Conv2D(filters,1,padding="same")(residual)
        x=layers.add([x,residual])  # Add back residual
        previous_block_activation=x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs=layers.Conv2D(num_classes,3,activation="softmax",padding="same")(x)
    # Define the model
    model=keras.Model(inputs,outputs)
    return model



keras.backend.clear_session()
model=get_model((320,320),22)
model.compile(optimizer='adam',loss="categorical_crossentropy")

a=model.predict(imarr[0:10])
for i in range(1):
    b=topict(a[i])
    plt.imshow(b)
    plt.show()
    plt.imshow(imarr[i])
    plt.show()


model.fit(imarr[:len(arr22)],arr22[:],epochs=5,batch_size=50)
keras.backend.clear_session()
print(i)



a=model.predict(imarr[0:10])
for i in range(10):
    b=topict(a[i])
    plt.imshow(b)
    plt.show()
    plt.imshow(imarr[i])
    plt.show()

